"""Build a validated source-to-index router for BRIGHT-based G1 training."""

from __future__ import annotations

import json
import os
from pathlib import Path

from bright import (
    BRIGHT_CONFIG,
    BRIGHT_DATASET,
    BRIGHT_REVISION,
    BRIGHT_TRAINING_SOURCE_ROUTES,
)
from fixed_corpus.encode import normalize_document
from fixed_corpus.index import sha256_file


def _training_documents_by_route(training_path: Path) -> tuple[dict[str, dict[str, str]], int]:
    expected: dict[str, dict[str, str]] = {}
    candidate_count = 0
    with training_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            source = record.get("source")
            if source not in BRIGHT_TRAINING_SOURCE_ROUTES:
                raise ValueError(
                    f"No official BRIGHT documents route for source {source!r} "
                    f"at {training_path}:{line_number}"
                )
            route = BRIGHT_TRAINING_SOURCE_ROUTES[source]
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


def validate_training_against_bright_documents(
    training_path: str | os.PathLike[str],
    documents_dir: str | os.PathLike[str],
) -> dict[str, int]:
    """Fail before GPU encoding if G1 IDs/text do not match official BRIGHT parquet."""
    import pyarrow.parquet as pq

    training_path = Path(training_path).resolve()
    documents_dir = Path(documents_dir).resolve()
    expected, candidate_count = _training_documents_by_route(training_path)
    unique_count = 0
    for route, expected_documents in sorted(expected.items()):
        parquet_path = documents_dir / f"{route}-00000-of-00001.parquet"
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Missing official BRIGHT documents: {parquet_path}")
        remaining = dict(expected_documents)
        parquet = pq.ParquetFile(parquet_path)
        if not {"id", "content"}.issubset(parquet.schema.names):
            raise ValueError(f"BRIGHT documents parquet requires id/content columns: {parquet_path}")
        for batch in parquet.iter_batches(columns=["id", "content"], batch_size=4096):
            for record in batch.to_pylist():
                document_id = str(record["id"])
                expected_text = remaining.pop(document_id, None)
                if expected_text is not None and normalize_document(record["content"]) != expected_text:
                    raise ValueError(
                        f"G1 candidate text differs from official BRIGHT {route!r} "
                        f"document {document_id!r}"
                    )
        if remaining:
            preview = ", ".join(sorted(remaining)[:3])
            raise ValueError(
                f"G1 references document IDs absent from official BRIGHT {route!r}: {preview}"
            )
        unique_count += len(expected_documents)
    return {
        "candidate_count": candidate_count,
        "unique_document_ids": unique_count,
        "route_count": len(expected),
    }


def write_bright_index_router(
    output_dir: str | os.PathLike[str],
    training_path: str | os.PathLike[str],
    *,
    revision: str = BRIGHT_REVISION,
) -> Path:
    """Validate G1 candidates against official BRIGHT documents and write the router."""
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
            "source_dataset": BRIGHT_DATASET,
            "source_config": BRIGHT_CONFIG,
            "source_revision": revision,
        }
        actual = {key: manifest.get(key) for key in identity}
        if actual != identity:
            raise ValueError(
                f"BRIGHT route {route!r} has wrong source identity: {actual!r} != {identity!r}"
            )

        remaining = dict(expected_documents)
        corpus_path = Path(manifest["corpus_path"])
        if not corpus_path.is_absolute():
            corpus_path = manifest_path.parent / corpus_path
        with corpus_path.open(encoding="utf-8") as corpus:
            for line in corpus:
                record = json.loads(line)
                document_id = str(record["document_key"])
                expected_text = remaining.pop(document_id, None)
                if expected_text is not None and normalize_document(record["contents"]) != expected_text:
                    raise ValueError(
                        f"G1 candidate text differs from official BRIGHT {route!r} "
                        f"document {document_id!r}"
                    )
        if remaining:
            preview = ", ".join(sorted(remaining)[:3])
            raise ValueError(
                f"G1 references document IDs absent from official BRIGHT {route!r}: {preview}"
            )
        routes[route] = {
            "manifest": os.path.relpath(manifest_path, output_dir),
            "sha256": sha256_file(manifest_path),
        }

    aliases = {
        source: route
        for source, route in BRIGHT_TRAINING_SOURCE_ROUTES.items()
        if source != route and route in routes
    }
    manifest = {
        "format_version": 1,
        "artifact_type": "frozen_document_index_router",
        "source_dataset": BRIGHT_DATASET,
        "source_config": BRIGHT_CONFIG,
        "source_revision": revision,
        "training_data_path": os.path.relpath(training_path, output_dir),
        "training_data_sha256": sha256_file(training_path),
        "training_candidate_count": candidate_count,
        "training_unique_document_ids": sum(len(values) for values in expected.values()),
        "routes": routes,
        "aliases": aliases,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "index_router_manifest.json"
    temporary = output.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return output
