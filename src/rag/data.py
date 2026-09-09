from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import Dataset

from embedding_protocol import append_configured_token, format_embedding_text


TRAIN_SOURCES = {"nq": ("train", 79168), "hotpotqa": ("train", 90447)}
EVALUATION_SUITE = {
    "nq": ("test", 3610),
    "triviaqa": ("test", 11313),
    "popqa": ("test", 14267),
    "hotpotqa": ("dev", 7405),
    "2wikimultihopqa": ("dev", 12576),
    "musique": ("dev", 2417),
    "bamboogle": ("test", 125),
}
EXPECTED_TRAIN_TOTAL = 169615
EXPECTED_EVALUATION_TOTAL = 51713
RAG_TASK_DESCRIPTION = "Given a question, retrieve Wikipedia documents that answer the question"


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def validate_flashrag_record(record: dict[str, Any], source: str) -> dict[str, Any]:
    query_id = record.get("id")
    question = record.get("question")
    answers = record.get("golden_answers")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError(f"{source} record requires non-empty string id")
    if not isinstance(question, str) or not question:
        raise ValueError(f"{source}:{query_id} requires non-empty question")
    if not isinstance(answers, list) or not answers or not all(isinstance(answer, str) for answer in answers):
        raise ValueError(f"{source}:{query_id} requires non-empty golden_answers")
    return record


def belongs_to_tuning_split(source: str, query_id: str, fraction: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{source}:{query_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < fraction


class CandidateManifestDataset(Dataset):
    """Lazy JSONL dataset containing frozen top-k candidates for query training."""

    query_prompt_template = "Instruct: {task_description}\nQuery:{query}"

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        tuning_fraction: float = 0.05,
        tuning_seed: int = 20260909,
        query_prompt_template: str | None = None,
        max_samples: int | None = None,
    ):
        if split not in {"train", "tuning", "full"}:
            raise ValueError("split must be train, tuning, or full")
        self.manifest_path = str(Path(manifest_path).resolve())
        self.split = split
        self.tuning_fraction = tuning_fraction
        self.tuning_seed = tuning_seed
        if query_prompt_template is not None:
            self.query_prompt_template = query_prompt_template
        self.offsets: list[int] = []
        self._handle = None
        with open(self.manifest_path, "rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                in_tuning = belongs_to_tuning_split(
                    record["source"], record["query_id"], tuning_fraction, tuning_seed
                )
                if split == "full" or (split == "tuning" and in_tuning) or (split == "train" and not in_tuning):
                    self.offsets.append(offset)
                    if max_samples is not None and len(self.offsets) >= max_samples:
                        break

    def __len__(self) -> int:
        return len(self.offsets)

    def _get_handle(self):
        if self._handle is None:
            self._handle = open(self.manifest_path, "rb")
        return self._handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        handle = self._get_handle()
        handle.seek(self.offsets[index])
        record = json.loads(handle.readline())
        record["formatted_query"] = format_embedding_text(
            self.query_prompt_template,
            record["question"],
            task_description=RAG_TASK_DESCRIPTION,
        )
        return record

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class RAGQueryCollator:
    def __init__(self, tokenizer, query_max_length: int = 128, append_token: str = "pad"):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.append_token = append_token

    def __call__(self, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            raise ValueError("Cannot collate an empty RAG batch")
        depth = len(records[0]["candidate_passage_ids"])
        if any(len(record["candidate_passage_ids"]) != depth for record in records):
            raise ValueError("All candidate lists in a batch must have the same depth")
        queries = append_configured_token(
            [record["formatted_query"] for record in records],
            self.tokenizer,
            self.append_token,
        )
        tokenized = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
            return_tensors="pt",
        )
        answer_mask = torch.tensor(
            [record["answer_positive_mask"] for record in records], dtype=torch.bool
        )
        training_mask = torch.tensor(
            [record.get("training_positive_mask", record["answer_positive_mask"]) for record in records],
            dtype=torch.bool,
        )
        return {
            "query": dict(tokenized),
            "query_ids": [record["query_id"] for record in records],
            "sources": [record["source"] for record in records],
            "questions": [record["question"] for record in records],
            "golden_answers": [record["golden_answers"] for record in records],
            "candidate_passage_ids": torch.tensor(
                [record["candidate_passage_ids"] for record in records], dtype=torch.long
            ),
            "answer_positive_mask": answer_mask,
            "training_positive_mask": training_mask,
            "evidence_passage_groups": [record.get("evidence_passage_groups", []) for record in records],
        }
