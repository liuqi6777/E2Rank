import json
import random
from collections import defaultdict
from typing import Any, Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset


TASK_PROMPTS = {
    "msmarco": "Given a web search query, retrieval the documents that answer the query",
    "nq": "Given a question, retrieval Wikipedia documents that answer the question",
    "hotpotqa": "Given a multi-hop question, retrieval the documents that can help answer the question",
    "trivia": "Retrieval Wikipedia documents that answer the question",
    "t2ranking": "Given a Chinese search query, retrieval the documents that answer the query",
    "dureader": "Given a Chinese search query, retrieval the documents that answer the query",
    "mmarco_chinese": "Given a Chinese web search query, retrieval the documents that answer the query",
    "cMedQAv2": "Given a Chinese medical question, retrieval the documents that answer the question",
    "miracl": "Given a question, retrieval Wikipedia documents that answer the question",
    "allnli": "Given a premise, retrieve a hypothesis that is entailed by the premise",
    "fever": "Given a claim, retrieve documents that support or refute the claim",
    "eli5_question_answer": "Given a question, retrieval the answer that explains it",
    "squad": "Given a question, retrieve a Wikipedia passage that answers the question",
    "quora_duplicates": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "mrtydi": "Given a question, retrieval Wikipedia documents that answer the question",
}

DEFAULT_TASK_PROMPTS = "Given a query, retrieval the documents that are relevant to the query"


class RankingDataset(Dataset):
    query_prompt_template = "Instruct: {task_description}\nQuery:{query}"

    def __init__(
        self,
        data_args: Any,
        batch_size: int | None = None,
    ):
        self.batch_size = batch_size or 32
        self.per_dataset_max_samples = data_args.per_dataset_max_samples
        self.samples: list[dict[str, Any]] = []
        self._load(data_args.data_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]

    def _format_query(self, task_name: str, query: str) -> str:
        retrieval_prompt = TASK_PROMPTS.get(task_name, DEFAULT_TASK_PROMPTS)
        return self.query_prompt_template.format(
            task_description=retrieval_prompt,
            query=query,
        )

    def _load(self, data_path: str) -> None:
        with open(data_path, "r") as f:
            raw_samples = [json.loads(line) for line in f.readlines()]

        sample_indices_by_task = defaultdict(list)
        normalized_samples: list[dict[str, Any]] = []

        for idx, sample in enumerate(raw_samples):
            task_name = sample.get("source", "unknown")
            sample_indices_by_task[task_name].append(idx)
            normalized_samples.append(
                {
                    "query": self._format_query(task_name, sample["query"]),
                    "document": sample["document"],
                    "ranking": sample["ranking"],
                }
            )

        ordered_batches = []
        for task_name, indices in sample_indices_by_task.items():
            random.shuffle(indices)
            limited_indices = (
                indices
                if self.per_dataset_max_samples is None
                else indices[: self.per_dataset_max_samples]
            )
            for start in range(0, len(limited_indices), self.batch_size):
                batch = limited_indices[start : start + self.batch_size]
                if len(batch) == self.batch_size:
                    ordered_batches.append(batch)
                else:
                    print(f"Skip 1 batch for dataset {task_name}.")

        random.shuffle(ordered_batches)
        ordered_indices = [idx for batch in ordered_batches for idx in batch]
        self.samples = [normalized_samples[idx] for idx in ordered_indices]
        print(f"Loaded {len(self.samples)} samples.")


class RankingDataCollator:
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        query_max_length: int = 512,
        doc_max_length: int = 1024,
        **_: Any,
    ):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        if not self.tokenizer.pad_token:
            if getattr(self.tokenizer, "eot_token", None):
                self.tokenizer.pad_token = self.tokenizer.eot_token
            else:
                self.tokenizer.pad_token = self.tokenizer.bos_token
        print(f"use ``{self.tokenizer.pad_token}`` as pad token for llm")

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        queries = [instance["query"] + self.tokenizer.pad_token for instance in instances]
        query_inputs = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
            return_tensors="pt",
        )

        documents = sum([instance["document"] for instance in instances], [])
        documents = [doc + self.tokenizer.pad_token for doc in documents]
        document_inputs = self.tokenizer(
            documents,
            padding=True,
            truncation=True,
            max_length=self.doc_max_length,
            return_tensors="pt",
        )

        ranking = torch.tensor([instance["ranking"] for instance in instances]) - 1
        return {
            "query": query_inputs,
            "document": document_inputs,
            "ranking": ranking,
        }
