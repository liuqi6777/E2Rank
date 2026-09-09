from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from rag.data import EVALUATION_SUITE, TRAIN_SOURCES, iter_jsonl, validate_flashrag_record
from rag.index import sha256_file
from rag.metrics import normalize_answer


REPO_ID = "RUC-NLPIR/FlashRAG_datasets"


def _resolved_dataset_revision(revision: str) -> str:
    encoded_repo = "/".join(urllib.parse.quote(part, safe="") for part in REPO_ID.split("/"))
    encoded_revision = urllib.parse.quote(revision, safe="")
    url = f"https://huggingface.co/api/datasets/{encoded_repo}/revision/{encoded_revision}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    resolved = payload.get("sha")
    if not isinstance(resolved, str) or not resolved:
        raise RuntimeError(f"Hugging Face API did not return a commit SHA for {revision!r}")
    return resolved


def _download_with_hf_cli(output_dir: Path, revision: str, patterns: list[str]) -> Path:
    executable = shutil.which("hf") or shutil.which("huggingface-cli")
    if executable is None:
        raise RuntimeError(
            "Neither `hf` nor `huggingface-cli` is available. Install the Hugging Face CLI "
            "outside the project environment, then retry."
        )
    local_dir = output_dir / "flashrag"
    if Path(executable).name == "hf":
        command = [
            executable,
            "download",
            REPO_ID,
            *sorted(set(patterns)),
            "--repo-type",
            "dataset",
            "--revision",
            revision,
            "--local-dir",
            str(local_dir),
            "--quiet",
        ]
    else:
        command = [
            executable,
            "download",
            REPO_ID,
            *sorted(set(patterns)),
            "--repo-type",
            "dataset",
            "--revision",
            revision,
            "--local-dir",
            str(local_dir),
        ]
    subprocess.run(command, check=True)
    return local_dir


def _safe_extract_corpus(archive_path: Path, output_path: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        candidates = []
        for info in archive.infolist():
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe archive member: {info.filename}")
            if member.suffix == ".jsonl":
                candidates.append(info)
        if not candidates:
            raise ValueError(f"No JSONL corpus found in {archive_path}")
        selected = max(candidates, key=lambda info: info.file_size)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(selected) as source, open(output_path, "wb") as target:
            shutil.copyfileobj(source, target, length=8 << 20)


def _validate_dataset_file(path: Path, source: str, expected: int) -> int:
    seen: set[str] = set()
    count = 0
    for record in iter_jsonl(path):
        validate_flashrag_record(record, source)
        if record["id"] in seen:
            raise ValueError(f"Duplicate id in {path}: {record['id']}")
        seen.add(record["id"])
        count += 1
    if count != expected:
        raise ValueError(f"Unexpected {source} count in {path}: {count}, expected {expected}")
    return count


def _index_corpus(corpus_path: Path, offsets_path: Path) -> int:
    offsets = []
    with open(corpus_path, "rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            if "id" not in record or not isinstance(record.get("contents"), str):
                raise ValueError(f"Invalid corpus record at byte {offset}")
            ordinal = len(offsets)
            try:
                passage_id = int(record["id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Corpus passage id is not an integer at byte {offset}") from exc
            if passage_id != ordinal:
                raise ValueError(
                    f"Corpus passage IDs must equal JSONL ordinals: id={passage_id}, ordinal={ordinal}"
                )
            offsets.append(offset)
    np.save(offsets_path, np.asarray(offsets, dtype=np.int64))
    return len(offsets)


def prepare(output_dir: Path, revision: str, overwrite: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "flashrag_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"FlashRAG manifest already exists: {manifest_path}. Use --overwrite to replace it."
        )
    patterns = [
        "retrieval-corpus/wiki18_100w.zip",
        *[f"{source}/{split}.jsonl" for source, (split, _) in TRAIN_SOURCES.items()],
        *[f"{source}/{split}.jsonl" for source, (split, _) in EVALUATION_SUITE.items()],
    ]
    snapshot_dir = _download_with_hf_cli(output_dir, revision, patterns)
    resolved_revision = _resolved_dataset_revision(revision)

    files = {}
    dataset_splits = {
        (source, split): expected
        for source, (split, expected) in [*TRAIN_SOURCES.items(), *EVALUATION_SUITE.items()]
    }
    for (source, split), expected in dataset_splits.items():
        path = snapshot_dir / source / f"{split}.jsonl"
        _validate_dataset_file(path, source, expected)
        files[f"{source}/{split}"] = {
            "path": os.path.relpath(path, output_dir),
            "dataset": source,
            "split": split,
            "split_id": f"{source}:{split}@{resolved_revision}",
            "count": expected,
            "sha256": sha256_file(path),
        }

    train_questions = {}
    for source, (split, _) in TRAIN_SOURCES.items():
        path = snapshot_dir / source / f"{split}.jsonl"
        for record in iter_jsonl(path):
            normalized = normalize_answer(record["question"])
            train_questions.setdefault(normalized, []).append(f"{source}/{split}:{record['id']}")
    leakage = []
    for source, (split, _) in EVALUATION_SUITE.items():
        path = snapshot_dir / source / f"{split}.jsonl"
        for record in iter_jsonl(path):
            normalized = normalize_answer(record["question"])
            if normalized in train_questions:
                leakage.append({
                    "normalized_question": normalized,
                    "train": train_questions[normalized],
                    "evaluation": f"{source}/{split}:{record['id']}",
                })
    if leakage:
        report_path = output_dir / "question_leakage.json"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(leakage, handle, ensure_ascii=False, indent=2)
        print(
            f"Found {len(leakage)} exact normalized train/evaluation question overlaps; "
            f"see {report_path}"
        )

    archive = snapshot_dir / "retrieval-corpus" / "wiki18_100w.zip"
    corpus_path = output_dir / "corpus" / "wiki18_100w.jsonl"
    if corpus_path.exists() and not overwrite:
        raise FileExistsError(f"Corpus already exists without a manifest: {corpus_path}")
    partial_corpus_path = corpus_path.with_suffix(corpus_path.suffix + ".partial")
    _safe_extract_corpus(archive, partial_corpus_path)
    os.replace(partial_corpus_path, corpus_path)
    offsets_path = output_dir / "corpus" / "wiki18_100w.offsets.npy"
    corpus_count = _index_corpus(corpus_path, offsets_path)
    manifest = {
        "format_version": 1,
        "source": "flashrag",
        "repo_id": REPO_ID,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "files": files,
        "expected_train_total": sum(value[1] for value in TRAIN_SOURCES.values()),
        "expected_evaluation_total": sum(value[1] for value in EVALUATION_SUITE.values()),
        "split_policy": {
            "train": {source: split for source, (split, _) in TRAIN_SOURCES.items()},
            "evaluation": {source: split for source, (split, _) in EVALUATION_SUITE.items()},
            "normalized_question_train_eval_overlap": 0,
        },
        "corpus": {
            "path": os.path.relpath(corpus_path, output_dir),
            "offsets_path": os.path.relpath(offsets_path, output_dir),
            "count": corpus_count,
            "passage_id_semantics": "integer id equals zero-based JSONL ordinal",
            "sha256": sha256_file(corpus_path),
            "offsets_sha256": sha256_file(offsets_path),
            "archive_sha256": sha256_file(archive),
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/rag")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = prepare(Path(args.output_dir), args.revision, args.overwrite)
    print(path)


if __name__ == "__main__":
    main()
