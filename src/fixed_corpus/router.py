"""Build a validated source-to-index router for ReasonRank-based G1 training."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from fixed_corpus.encode import normalize_document
from fixed_corpus.index import sha256_file


# The BRIGHT identity remains shared by post-training evaluation.
BRIGHT_DATASET = "xlangai/BRIGHT"
BRIGHT_CONFIG = "documents"
BRIGHT_REVISION = "3066d29c9651a576c8aba4832d249807b181ecae"
REASONRANK_DATASET = "liuwenhan/reasonrank_data_13k"
REASONRANK_CONFIG = "id_doc"
REASONRANK_REVISION = "09c3ac0f8dee374207592e118866d9bf943e74cc"
REASONRANK_TRAINING_SOURCES = (
    "biology",
    "earth_science",
    "economics",
    "leetcode",
    "math-qa",
    "math-theorem",
    "robotics",
    "stackoverflow",
    "sustainable_living",
)


def _training_documents_by_route(training_path: Path) -> tuple[dict[str, dict[str, str]], int]:
    expected: dict[str, dict[str, str]] = {}
    candidate_count = 0
    with training_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            source = record.get("source")
            if source not in REASONRANK_TRAINING_SOURCES:
                raise ValueError(
                    f"No canonical ReasonRank documents route for source {source!r} "
                    f"at {training_path}:{line_number}"
                )
            route = source
            ids = record.get("document_ids")
            texts = record.get("document")
            if not isinstance(ids, list) or not isinstance(texts, list) or len(ids) != len(texts):
                raise ValueError(f"Candidate ID/text mismatch at {training_path}:{line_number}")
            route_documents = expected.setdefault(route, {})
            for document_id, text in zip(ids, texts):
                if not isinstance(document_id, str) or not isinstance(text, str):
                    raise ValueError(f"Candidate ID/text must be strings at {training_path}:{line_number}")
                normalized = normalize_document(text)
                previous = route_documents.setdefault(document_id, normalized)
                if previous != normalized:
                    raise ValueError(
                        f"Training source {source!r} has conflicting text for document ID {document_id!r}"
                    )
                candidate_count += 1
    return expected, candidate_count


def _iter_json_mapping(path: Path, chunk_size: int = 1024 * 1024):
    """Stream a top-level JSON object without loading large id_doc files into RAM."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def refill():
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            block = handle.read(chunk_size)
            if block:
                buffer += block
            else:
                eof = True

        def decode_value():
            nonlocal position
            while True:
                try:
                    value, position = decoder.raw_decode(buffer, position)
                    return value
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()

        refill()
        while not eof and not buffer.strip():
            refill()
        position = len(buffer) - len(buffer.lstrip())
        if position >= len(buffer) or buffer[position] != "{":
            raise ValueError(f"Expected a JSON object: {path}")
        position += 1
        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                if eof:
                    raise ValueError(f"Unterminated JSON object: {path}")
                refill()
            if buffer[position] == "}":
                return
            key = decode_value()
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                refill()
            if buffer[position] != ":":
                raise ValueError(f"Expected ':' after key in {path}")
            position += 1
            while position >= len(buffer) and not eof:
                refill()
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            value = decode_value()
            yield key, value
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                refill()
            if buffer[position] == ",":
                position += 1
            elif buffer[position] == "}":
                return
            else:
                raise ValueError(f"Expected ',' or '}}' in {path}")


def materialize_reasonrank_documents(
    training_path: str | os.PathLike[str],
    documents_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> dict[str, int]:
    """Extract canonical full texts for the document IDs referenced by G1."""
    training_path = Path(training_path).resolve()
    documents_dir = Path(documents_dir).resolve()
    output_dir = Path(output_dir).resolve()
    expected, candidate_count = _training_documents_by_route(training_path)
    source_hashes = {
        route: sha256_file(documents_dir / f"{route}.json")
        for route in sorted(expected)
    }
    identity = {
        "format_version": 1,
        "artifact_type": "reasonrank_canonical_documents",
        "source_dataset": REASONRANK_DATASET,
        "source_config": REASONRANK_CONFIG,
        "source_revision": REASONRANK_REVISION,
        "training_data_sha256": sha256_file(training_path),
        "source_sha256": source_hashes,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if all(existing.get(key) == value for key, value in identity.items()):
            return existing["statistics"]
        raise ValueError(f"Canonical ReasonRank documents are stale: {output_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete canonical documents: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    unique_count = 0
    try:
        for route, expected_documents in sorted(expected.items()):
            source_path = documents_dir / f"{route}.json"
            remaining = set(expected_documents)
            with (stage / f"{route}.jsonl").open("w", encoding="utf-8") as output:
                for document_id, text in _iter_json_mapping(source_path):
                    if document_id not in remaining:
                        continue
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(
                            f"Canonical ReasonRank document {document_id!r} in {source_path} is empty"
                        )
                    output.write(json.dumps(
                        {"id": document_id, "content": text},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ) + "\n")
                    remaining.remove(document_id)
            if remaining:
                preview = ", ".join(sorted(remaining)[:3])
                raise ValueError(
                    f"G1 references document IDs absent from ReasonRank {route!r}: {preview}"
                )
            unique_count += len(expected_documents)
        statistics = {
            "candidate_count": candidate_count,
            "unique_document_ids": unique_count,
            "route_count": len(expected),
        }
        identity["statistics"] = statistics
        identity["artifacts"] = {
            route: {
                "path": f"{route}.jsonl",
                "sha256": sha256_file(stage / f"{route}.jsonl"),
            }
            for route in sorted(expected)
        }
        (stage / "manifest.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        stage.rename(output_dir)
        return statistics
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def write_reasonrank_index_router(
    output_dir: str | os.PathLike[str],
    training_path: str | os.PathLike[str],
    *,
    revision: str = REASONRANK_REVISION,
) -> Path:
    """Validate canonical ReasonRank route indexes and write the router."""
    output_dir = Path(output_dir).resolve()
    training_path = Path(training_path).resolve()
    expected, candidate_count = _training_documents_by_route(training_path)
    routes: dict[str, dict[str, str]] = {}

    for route, expected_documents in sorted(expected.items()):
        manifest_path = output_dir / route / "index_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing BRIGHT route index: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        identity = {
            "source_name": route,
            "source_dataset": REASONRANK_DATASET,
            "source_config": REASONRANK_CONFIG,
            "source_revision": revision,
        }
        actual = {key: manifest.get(key) for key in identity}
        if actual != identity:
            raise ValueError(
                f"ReasonRank route {route!r} has wrong source identity: {actual!r} != {identity!r}"
            )

        remaining = dict(expected_documents)
        corpus_path = Path(manifest["corpus_path"])
        if not corpus_path.is_absolute():
            corpus_path = manifest_path.parent / corpus_path
        with corpus_path.open(encoding="utf-8") as corpus:
            for line in corpus:
                record = json.loads(line)
                document_id = str(record["document_key"])
                remaining.pop(document_id, None)
        if remaining:
            preview = ", ".join(sorted(remaining)[:3])
            raise ValueError(
                f"G1 references document IDs absent from canonical ReasonRank {route!r}: {preview}"
            )
        routes[route] = {
            "manifest": os.path.relpath(manifest_path, output_dir),
            "sha256": sha256_file(manifest_path),
        }

    manifest = {
        "format_version": 1,
        "artifact_type": "frozen_document_index_router",
        "source_dataset": REASONRANK_DATASET,
        "source_config": REASONRANK_CONFIG,
        "source_revision": revision,
        "training_data_path": os.path.relpath(training_path, output_dir),
        "training_data_sha256": sha256_file(training_path),
        "training_candidate_count": candidate_count,
        "training_unique_document_ids": sum(len(values) for values in expected.values()),
        "routes": routes,
        "aliases": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "index_router_manifest.json"
    temporary = output.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return output
