"""Convert ReasonRank RL parquet records to E2Rank's fixed-size listwise JSONL.

The ReasonRank prompt contains the query and the passage texts, while ``label`` is a
1-indexed teacher permutation over ``initial_list``.  To keep candidate selection
independent of that teacher label, this converter takes the first N candidates from the
original retrieval order and filters the teacher permutation to those same positions.
``relevant_docids`` is intentionally ignored: this project trains from teacher order.

Example:

    uv run python scripts/convert_reasonrank.py \
        --input data/reasonrank/train.parquet \
        --output data/reasonrank_train_slate16.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq


PASSAGE_MARKER = re.compile(r"(?m)^\[(\d+)\]\s+")
LABEL_FORMAT = re.compile(r"^\s*\[\d+\](?:\s*>\s*\[\d+\])*\s*$")
SEARCH_QUERY_MARKER = "\nSearch Query: "
RANK_INSTRUCTION_MARKER = "\nRank the "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="ReasonRank parquet file")
    parser.add_argument("--output", required=True, type=Path, help="output JSONL file")
    parser.add_argument(
        "--slate-size",
        type=int,
        default=16,
        help="number of candidates to retain from the original retrieval order (default: 16)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace --output if it already exists",
    )
    return parser.parse_args()


def iter_parquet_rows(path: Path, batch_size: int = 128) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def _user_prompt(messages: Any) -> str:
    if not isinstance(messages, list):
        raise ValueError("'prompt' must be a list of chat messages")
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content:
                return content
            break
    raise ValueError("'prompt' does not contain a non-empty user message")


def parse_query_and_passages(prompt: str) -> tuple[str, list[str]]:
    passage_section, separator, footer = prompt.rpartition(SEARCH_QUERY_MARKER)
    if not separator:
        raise ValueError("user prompt is missing the final 'Search Query:' marker")

    query, separator, _ = footer.partition(RANK_INSTRUCTION_MARKER)
    query = query.strip()
    if not separator or not query:
        raise ValueError("user prompt has a malformed final query/ranking instruction")

    matches = list(PASSAGE_MARKER.finditer(passage_section))
    passage_numbers = [int(match.group(1)) for match in matches]
    expected_numbers = list(range(1, len(matches) + 1))
    if passage_numbers != expected_numbers:
        raise ValueError(
            f"passage identifiers must be consecutive from 1, got {passage_numbers}"
        )

    passages = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(passage_section)
        passage = passage_section[match.end() : end].strip()
        if not passage:
            raise ValueError(f"passage [{index + 1}] is empty")
        passages.append(passage)
    return query, passages


def parse_teacher_ranking(label: Any, candidate_count: int) -> list[int]:
    if not isinstance(label, str) or not LABEL_FORMAT.fullmatch(label):
        raise ValueError(f"malformed teacher label: {label!r}")
    ranking = [int(value) for value in re.findall(r"\[(\d+)\]", label)]
    expected = list(range(1, candidate_count + 1))
    if sorted(ranking) != expected:
        raise ValueError(
            f"teacher label must be a permutation of {expected}, got {ranking}"
        )
    return ranking


def convert_record(record: dict[str, Any], slate_size: int) -> dict[str, Any] | None:
    initial_list = record.get("initial_list")
    if not isinstance(initial_list, list) or not all(
        isinstance(doc_id, str) for doc_id in initial_list
    ):
        raise ValueError("'initial_list' must be a list of document IDs")
    candidate_count = len(initial_list)
    if candidate_count < slate_size:
        return None

    query, passages = parse_query_and_passages(_user_prompt(record.get("prompt")))
    if len(passages) != candidate_count:
        raise ValueError(
            f"prompt has {len(passages)} passages but initial_list has {candidate_count} IDs"
        )

    ranking = parse_teacher_ranking(record.get("label"), candidate_count)
    final_list = record.get("final_list")
    if final_list is not None:
        expected_final_list = [initial_list[position - 1] for position in ranking]
        if final_list != expected_final_list:
            raise ValueError("'final_list' does not match label applied to initial_list")

    # Candidate selection uses only the original retriever order. Filtering the teacher
    # permutation then gives a valid teacher ordering over exactly those retained passages.
    retained_ranking = [position for position in ranking if position <= slate_size]
    if sorted(retained_ranking) != list(range(1, slate_size + 1)):
        raise ValueError("filtered teacher ranking is not a valid fixed-size permutation")

    source = record.get("dataset")
    if not isinstance(source, str) or not source:
        raise ValueError("'dataset' must be a non-empty source name")
    return {
        "query": query,
        "document": passages[:slate_size],
        "ranking": retained_ranking,
        "source": source,
    }


def convert_file(input_path: Path, output_path: Path, slate_size: int, overwrite: bool) -> None:
    if slate_size < 2:
        raise ValueError("--slate-size must be at least 2")
    if not input_path.is_file():
        raise FileNotFoundError(f"input parquet does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--input and --output must be different files")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists (pass --overwrite to replace it): {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    converted = 0
    skipped_too_short = 0
    source_counts: Counter[str] = Counter()
    candidate_counts: Counter[int] = Counter()

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            for row_index, record in enumerate(iter_parquet_rows(input_path)):
                total += 1
                initial_list = record.get("initial_list")
                if isinstance(initial_list, list):
                    candidate_counts[len(initial_list)] += 1
                try:
                    converted_record = convert_record(record, slate_size)
                except ValueError as exc:
                    raise ValueError(f"invalid row {row_index}: {exc}") from exc
                if converted_record is None:
                    skipped_too_short += 1
                    continue
                output_file.write(json.dumps(converted_record, ensure_ascii=False) + "\n")
                converted += 1
                source_counts[converted_record["source"]] += 1

            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"Converted {converted}/{total} rows to {output_path}")
    print(f"Skipped {skipped_too_short} rows with fewer than {slate_size} candidates")
    print(
        "Original candidate counts: "
        + ", ".join(f"{size}:{count}" for size, count in sorted(candidate_counts.items()))
    )
    print(
        "Converted sources: "
        + ", ".join(f"{source}:{count}" for source, count in sorted(source_counts.items()))
    )


def main() -> None:
    args = parse_args()
    try:
        convert_file(args.input, args.output, args.slate_size, args.overwrite)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
