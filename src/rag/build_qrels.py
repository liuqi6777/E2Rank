"""Build corpus-aligned binary qrels for DPR NQ and FlashRAG HotpotQA.

The builder deliberately keeps label construction separate from retrieval. NQ uses
only DPR contexts with ``score == 1000`` (the human-positive relevance level), while
HotpotQA maps every annotated supporting fact to the frozen FlashRAG Wikipedia
corpus. The resulting JSONL contains only queries with at least one fully mapped
human-evidence passage.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, TextIO

from rag.data import iter_jsonl, validate_flashrag_record
from rag.metrics import best_window_token_f1, extract_evidence_groups, normalize_answer


DPR_NQ_TRAIN_URL = (
    "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz"
)
FLASHRAG_REPO_ID = "RUC-NLPIR/FlashRAG_datasets"
FLASHRAG_HOTPOTQA_TRAIN_PATH = "hotpotqa/train.jsonl"
EXPECTED_DPR_NQ_TRAIN = 58_880
EXPECTED_HOTPOTQA_TRAIN = 90_447
EXPECTED_DPR_W100_PASSAGES = 21_015_324


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path) -> None:
    """Download to an atomic partial file, resuming when the server supports it."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "E2Rank-RL-qrels-builder/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        append = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as writer:
            shutil.copyfileobj(response, writer, length=8 << 20)
            writer.flush()
            os.fsync(writer.fileno())
    os.replace(partial, target)


def _flashrag_url(revision: str, relative_path: str) -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in FLASHRAG_REPO_ID.split("/"))
    revision = urllib.parse.quote(revision, safe="")
    path = "/".join(urllib.parse.quote(part, safe="") for part in relative_path.split("/"))
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    """Stream objects from a top-level JSON array without loading the file in memory."""
    decoder = json.JSONDecoder()
    with _open_text(path) as handle:
        buffer = ""
        position = 0
        started = False
        finished = False

        def refill() -> bool:
            nonlocal buffer, position
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = handle.read(chunk_size)
            if not chunk:
                return False
            buffer += chunk
            return True

        while True:
            while position >= len(buffer) and refill():
                pass
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer) and refill():
                    continue
                if position >= len(buffer) or buffer[position] != "[":
                    raise ValueError(f"Expected a top-level JSON array in {path}")
                position += 1
                started = True

            while True:
                while position >= len(buffer):
                    if not refill():
                        raise ValueError(f"Unterminated JSON array in {path}")
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                    if position >= len(buffer) and not refill():
                        raise ValueError(f"Unterminated JSON array in {path}")
                if position < len(buffer) and buffer[position] == "]":
                    position += 1
                    finished = True
                    break
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if not refill():
                        raise ValueError(f"Invalid JSON array in {path}")
                    continue
                if not isinstance(value, dict):
                    raise ValueError(f"Every item in {path} must be an object")
                position = end
                yield value
                break
            if finished:
                trailing = buffer[position:] + handle.read()
                if trailing.strip():
                    raise ValueError(f"Unexpected content after JSON array in {path}")
                return


def _strip_wrapping_quotes(text: str) -> str:
    value = text.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value


def split_corpus_contents(contents: str) -> tuple[str, str]:
    title, separator, text = contents.partition("\n")
    if not separator:
        raise ValueError("Corpus contents must use the title\\ntext format")
    return _strip_wrapping_quotes(title), text


def _canonical_component(text: str, *, relaxed: bool) -> str:
    value = unicodedata.normalize("NFKC" if relaxed else "NFC", str(text))
    value = " ".join(value.split())
    if relaxed:
        value = value.replace("“", '"').replace("”", '"').replace("''", '"')
        while '""' in value:
            value = value.replace('""', '"')
    return value


def passage_fingerprint(title: str, text: str, *, relaxed: bool = False) -> str:
    payload = (
        _canonical_component(_strip_wrapping_quotes(title), relaxed=relaxed)
        + "\0"
        + _canonical_component(text, relaxed=relaxed)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _human_dpr_contexts(record: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for context in record.get("positive_ctxs") or []:
        try:
            human_positive = float(context.get("score")) == 1000.0
        except (TypeError, ValueError):
            human_positive = False
        if not human_positive:
            continue
        title, text = context.get("title"), context.get("text")
        if not isinstance(title, str) or not isinstance(text, str) or not title or not text:
            continue
        key = passage_fingerprint(title, text)
        if key not in seen:
            seen.add(key)
            result.append(
                {
                    "strict_key": key,
                    "relaxed_key": passage_fingerprint(title, text, relaxed=True),
                    "dpr_passage_id": str(context.get("passage_id", "")),
                }
            )
    return result


def load_dpr_nq(path: Path, expected_count: int | None) -> tuple[list[dict[str, Any]], dict]:
    rows = []
    statistics = Counter()
    for index, record in enumerate(iter_json_array(path)):
        statistics["raw"] += 1
        question = record.get("question")
        answers = record.get("answers")
        if not isinstance(question, str) or not question.strip():
            statistics["invalid_question"] += 1
            continue
        if not isinstance(answers, list) or not answers or not all(
            isinstance(answer, str) and answer for answer in answers
        ):
            statistics["invalid_answers"] += 1
            continue
        positives = _human_dpr_contexts(record)
        if not positives:
            statistics["missing_human_positive"] += 1
            continue
        rows.append(
            {
                "query_id": f"dpr_train_{index}",
                "source": "nq",
                "question": question.strip(),
                "golden_answers": answers,
                "positive_targets": positives,
            }
        )
    if expected_count is not None and statistics["raw"] != expected_count:
        raise ValueError(
            f"Unexpected DPR NQ train count: {statistics['raw']}, expected {expected_count}"
        )
    statistics["label_ready"] = len(rows)
    return rows, dict(statistics)


def load_hotpotqa(path: Path, expected_count: int | None) -> tuple[list[dict[str, Any]], dict]:
    rows = []
    statistics = Counter()
    for record in iter_jsonl(path):
        statistics["raw"] += 1
        validate_flashrag_record(record, "hotpotqa")
        groups = extract_evidence_groups(record)
        if not groups or any(not group.get("title") or not group.get("sentence") for group in groups):
            statistics["missing_supporting_fact_text"] += 1
            continue
        fact_keys = [
            (
                normalize_answer(group["title"]),
                normalize_answer(group["sentence"]),
            )
            for group in groups
        ]
        if any(not title or not sentence for title, sentence in fact_keys):
            statistics["empty_normalized_supporting_fact"] += 1
            continue
        rows.append(
            {
                "query_id": record["id"],
                "source": "hotpotqa",
                "question": record["question"],
                "golden_answers": record["golden_answers"],
                "fact_keys": fact_keys,
            }
        )
    if expected_count is not None and statistics["raw"] != expected_count:
        raise ValueError(
            f"Unexpected HotpotQA train count: {statistics['raw']}, expected {expected_count}"
        )
    statistics["label_ready"] = len(rows)
    return rows, dict(statistics)


def align_to_corpus(
    corpus_path: Path,
    nq_rows: list[dict[str, Any]],
    hotpot_rows: list[dict[str, Any]],
    minimum_window_f1: float,
    expected_count: int | None,
) -> tuple[dict[str, list[int]], dict[tuple[str, str], tuple[float, int]], dict]:
    nq_strict_targets: dict[str, set[str]] = defaultdict(set)
    nq_relaxed_targets: dict[str, set[str]] = defaultdict(set)
    for row in nq_rows:
        for target in row["positive_targets"]:
            nq_strict_targets[target["strict_key"]].add(target["strict_key"])
            nq_relaxed_targets[target["relaxed_key"]].add(target["strict_key"])

    facts_by_title: dict[str, set[str]] = defaultdict(set)
    for row in hotpot_rows:
        for title, sentence in row["fact_keys"]:
            facts_by_title[title].add(sentence)

    nq_matches: dict[str, list[int]] = defaultdict(list)
    best_hotpot: dict[tuple[str, str], tuple[float, int]] = {}
    statistics = Counter()
    corpus_digest = hashlib.sha256()
    with corpus_path.open("rb") as handle:
        for ordinal, raw_line in enumerate(handle):
            corpus_digest.update(raw_line)
            if not raw_line.strip():
                raise ValueError(f"Blank corpus line at physical line {ordinal + 1}")
            record = json.loads(raw_line)
            try:
                passage_id = int(record["id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid corpus id at line {ordinal + 1}") from exc
            if passage_id != ordinal:
                raise ValueError(
                    f"Corpus ids must equal ordinals: id={passage_id}, ordinal={ordinal}"
                )
            contents = record.get("contents")
            if not isinstance(contents, str):
                raise ValueError(f"Invalid corpus contents at ordinal {ordinal}")
            title, text = split_corpus_contents(contents)

            strict_key = passage_fingerprint(title, text)
            if strict_key in nq_strict_targets:
                for target_key in nq_strict_targets[strict_key]:
                    nq_matches[target_key].append(ordinal)
                statistics["nq_strict_passage_matches"] += 1
            else:
                relaxed_key = passage_fingerprint(title, text, relaxed=True)
                if relaxed_key in nq_relaxed_targets:
                    for target_key in nq_relaxed_targets[relaxed_key]:
                        nq_matches[target_key].append(ordinal)
                    statistics["nq_relaxed_passage_matches"] += 1

            normalized_title = normalize_answer(title)
            fact_sentences = facts_by_title.get(normalized_title)
            if not fact_sentences:
                continue
            normalized_contents = normalize_answer(contents)
            for sentence in fact_sentences:
                if sentence in normalized_contents:
                    score = 1.0
                else:
                    score = best_window_token_f1(sentence, contents)
                    if score < minimum_window_f1:
                        continue
                fact_key = (normalized_title, sentence)
                current = best_hotpot.get(fact_key)
                if current is None or score > current[0] or (
                    score == current[0] and ordinal < current[1]
                ):
                    best_hotpot[fact_key] = (score, ordinal)
    count = ordinal + 1 if "ordinal" in locals() else 0
    if expected_count is not None and count != expected_count:
        raise ValueError(f"Unexpected corpus count: {count}, expected {expected_count}")
    statistics["corpus_count"] = count
    statistics["corpus_sha256"] = corpus_digest.hexdigest()
    statistics["hotpot_fact_targets"] = sum(len(values) for values in facts_by_title.values())
    statistics["hotpot_fact_matches"] = len(best_hotpot)
    return dict(nq_matches), best_hotpot, dict(statistics)


def materialize_qrels(
    nq_rows: list[dict[str, Any]],
    hotpot_rows: list[dict[str, Any]],
    nq_matches: dict[str, list[int]],
    hotpot_matches: dict[tuple[str, str], tuple[float, int]],
) -> tuple[list[dict[str, Any]], dict, list[dict[str, Any]]]:
    output = []
    unresolved = []
    statistics: dict[str, Counter] = {"nq": Counter(), "hotpotqa": Counter()}
    for row in nq_rows:
        ordinals = []
        failures = []
        for target in row["positive_targets"]:
            matches = nq_matches.get(target["strict_key"], [])
            if not matches:
                failures.append(
                    {
                        "dpr_passage_id": target["dpr_passage_id"],
                        "match_count": 0,
                    }
                )
            else:
                # Content-identical corpus duplicates are equally relevant. Keeping
                # every matching ordinal avoids treating a duplicate of a labeled
                # DPR positive as a negative during retrieval.
                ordinals.extend(matches)
        ordinals = sorted(set(ordinals))
        if failures or not ordinals:
            statistics["nq"]["dropped_unmapped"] += 1
            unresolved.append(
                {"query_id": row["query_id"], "source": "nq", "failures": failures}
            )
            continue
        output.append(
            {
                "query_id": row["query_id"],
                "source": "nq",
                "question": row["question"],
                "golden_answers": row["golden_answers"],
                "qrel_passage_ids": ordinals,
                "qrel_relevance": [1] * len(ordinals),
                "evidence_passage_groups": [[ordinal] for ordinal in ordinals],
                "label_origin": "dpr_human_positive_score_1000",
            }
        )
        statistics["nq"]["kept"] += 1

    for row in hotpot_rows:
        groups = []
        failures = []
        for title, sentence in row["fact_keys"]:
            match = hotpot_matches.get((title, sentence))
            if match is None:
                failures.append({"title": title, "sentence": sentence})
            else:
                groups.append([match[1]])
        if failures or not groups:
            statistics["hotpotqa"]["dropped_incomplete_evidence_mapping"] += 1
            unresolved.append(
                {
                    "query_id": row["query_id"],
                    "source": "hotpotqa",
                    "failures": failures,
                }
            )
            continue
        ordinals = sorted({ordinal for group in groups for ordinal in group})
        output.append(
            {
                "query_id": row["query_id"],
                "source": "hotpotqa",
                "question": row["question"],
                "golden_answers": row["golden_answers"],
                "qrel_passage_ids": ordinals,
                "qrel_relevance": [1] * len(ordinals),
                "evidence_passage_groups": groups,
                "label_origin": "hotpotqa_supporting_facts",
            }
        )
        statistics["hotpotqa"]["kept"] += 1
    return output, {key: dict(value) for key, value in statistics.items()}, unresolved


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def build(args: argparse.Namespace) -> tuple[Path, Path]:
    corpus_path = Path(args.corpus_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_path = output_dir / "nq_hotpotqa_train.jsonl"
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    unresolved_path = output_dir / "nq_hotpotqa_train.unresolved.jsonl"
    for path in (output_path, manifest_path, unresolved_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}; pass --overwrite to replace it")
    if not corpus_path.is_file():
        raise FileNotFoundError(
            f"Missing frozen FlashRAG corpus: {corpus_path}. Run `scripts/rag_pipeline.sh prepare` first."
        )

    raw_dir = Path(args.raw_dir).resolve()
    dpr_path = Path(args.dpr_nq_train).resolve() if args.dpr_nq_train else raw_dir / "biencoder-nq-train.json.gz"
    hotpot_path: Path | None = Path(args.hotpotqa_train).resolve() if args.hotpotqa_train else None
    flashrag_manifest = Path(args.flashrag_manifest).resolve()
    flashrag_revision = args.flashrag_revision
    if hotpot_path is None and flashrag_manifest.is_file():
        with flashrag_manifest.open(encoding="utf-8") as handle:
            source_manifest = json.load(handle)
        entry = source_manifest.get("files", {}).get("hotpotqa/train")
        if entry:
            candidate = Path(entry["path"])
            hotpot_path = candidate if candidate.is_absolute() else flashrag_manifest.parent / candidate
            flashrag_revision = source_manifest.get("resolved_revision", flashrag_revision)
    if hotpot_path is None:
        hotpot_path = raw_dir / "hotpotqa-train.jsonl"

    if not dpr_path.is_file():
        if not args.download_missing:
            raise FileNotFoundError(f"Missing DPR NQ train input: {dpr_path}")
        print(f"Downloading {DPR_NQ_TRAIN_URL} -> {dpr_path}")
        _download(DPR_NQ_TRAIN_URL, dpr_path)
    if not hotpot_path.is_file():
        if not args.download_missing:
            raise FileNotFoundError(f"Missing HotpotQA train input: {hotpot_path}")
        url = _flashrag_url(flashrag_revision, FLASHRAG_HOTPOTQA_TRAIN_PATH)
        print(f"Downloading {url} -> {hotpot_path}")
        _download(url, hotpot_path)

    print("Reading DPR NQ train labels...")
    nq_rows, nq_input_stats = load_dpr_nq(dpr_path, args.expected_nq_count)
    print("Reading FlashRAG HotpotQA supporting facts...")
    hotpot_rows, hotpot_input_stats = load_hotpotqa(hotpot_path, args.expected_hotpotqa_count)
    print("Scanning the frozen corpus once to align both label sources...")
    nq_matches, hotpot_matches, corpus_stats = align_to_corpus(
        corpus_path,
        nq_rows,
        hotpot_rows,
        args.hotpot_minimum_f1,
        args.expected_corpus_count,
    )
    rows, output_stats, unresolved = materialize_qrels(
        nq_rows, hotpot_rows, nq_matches, hotpot_matches
    )
    if not rows:
        raise RuntimeError("No qrel-aligned training records were produced")

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(output_path, rows)
    _atomic_jsonl(unresolved_path, unresolved)
    manifest = {
        "format_version": 1,
        "artifact_origin": "E2Rank-RL",
        "artifact_type": "rag_binary_passage_qrels",
        "label_policy": {
            "relevance": "binary",
            "nq": "DPR positive_ctxs with score == 1000 only",
            "hotpotqa": "all annotated supporting facts mapped by title and sentence",
            "retrieval_time_relabeling": False,
        },
        "inputs": {
            "dpr_nq_train": {
                "path": str(dpr_path),
                "url": DPR_NQ_TRAIN_URL,
                "sha256": sha256_file(dpr_path),
            },
            "hotpotqa_train": {
                "path": str(hotpot_path),
                "repo_id": FLASHRAG_REPO_ID,
                "revision": flashrag_revision,
                "sha256": sha256_file(hotpot_path),
            },
            "corpus": {
                "path": str(corpus_path),
                "count": corpus_stats["corpus_count"],
                "sha256": corpus_stats["corpus_sha256"],
            },
        },
        "statistics": {
            "input": {"nq": nq_input_stats, "hotpotqa": hotpot_input_stats},
            "alignment": corpus_stats,
            "output": output_stats,
            "total_kept": len(rows),
            "total_unresolved": len(unresolved),
        },
        "qrels_path": output_path.name,
        "qrels_sha256": sha256_file(output_path),
        "unresolved_path": unresolved_path.name,
        "unresolved_sha256": sha256_file(unresolved_path),
    }
    partial_manifest = manifest_path.with_suffix(manifest_path.suffix + ".partial")
    with partial_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial_manifest, manifest_path)
    return output_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one frozen binary-qrels training set from DPR NQ and FlashRAG HotpotQA"
    )
    parser.add_argument("--corpus-path", default="data/rag/corpus/wiki18_100w.jsonl")
    parser.add_argument("--flashrag-manifest", default="data/rag/flashrag_manifest.json")
    parser.add_argument("--flashrag-revision", default="main")
    parser.add_argument("--dpr-nq-train", default=None)
    parser.add_argument("--hotpotqa-train", default=None)
    parser.add_argument("--raw-dir", default="data/rag/raw")
    parser.add_argument("--output-dir", default="data/rag/qrels")
    parser.add_argument("--hotpot-minimum-f1", type=float, default=0.8)
    parser.add_argument("--expected-nq-count", type=int, default=EXPECTED_DPR_NQ_TRAIN)
    parser.add_argument("--expected-hotpotqa-count", type=int, default=EXPECTED_HOTPOTQA_TRAIN)
    parser.add_argument("--expected-corpus-count", type=int, default=EXPECTED_DPR_W100_PASSAGES)
    parser.add_argument(
        "--download-missing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.hotpot_minimum_f1 <= 1.0:
        parser.error("--hotpot-minimum-f1 must be in [0, 1]")
    for name in ("expected_nq_count", "expected_hotpotqa_count", "expected_corpus_count"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    output_path, manifest_path = build(args)
    print(output_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
