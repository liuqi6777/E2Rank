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


def build_relevance_labels(
    ranking: torch.Tensor,
    scheme: str = "graded",
) -> torch.Tensor:
    if ranking is None:
        raise ValueError("ranking is required to build relevance labels")
    if ranking.dim() != 2:
        raise ValueError(f"ranking must be a 2D tensor, got shape {tuple(ranking.shape)}")
    if scheme not in {"graded", "binary"}:
        raise ValueError(f"Unsupported relevance scheme: {scheme}")

    batch_size, slate_length = ranking.shape
    relevance = torch.zeros(batch_size, slate_length, device=ranking.device, dtype=torch.float32)

    rank_scores = torch.zeros(slate_length, device=ranking.device, dtype=torch.float32)
    if slate_length > 0:
        rank_scores[0] = 3.0 if scheme == "graded" else 1.0
    if scheme == "graded":
        if slate_length > 1:
            rank_scores[1:min(5, slate_length)] = 2.0
        if slate_length > 5:
            rank_scores[5:min(10, slate_length)] = 1.0

    relevance.scatter_(
        dim=1,
        index=ranking,
        src=rank_scores.unsqueeze(0).expand(batch_size, -1),
    )
    return relevance


class RankingDataCollator:
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        query_max_length: int = 512,
        doc_max_length: int = 1024,
        relevance_scheme: str = "binary",
        **_: Any,
    ):
        if relevance_scheme not in {"binary", "graded"}:
            raise ValueError(f"Unsupported relevance_scheme: {relevance_scheme}")
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        self.relevance_scheme = relevance_scheme
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

        ranking = torch.tensor([instance["ranking"] for instance in instances], dtype=torch.long) - 1
        original_relevance_labels = build_relevance_labels(
            ranking=ranking,
            scheme=self.relevance_scheme,
        )

        positive_documents: list[str] = []
        negative_documents: list[str] = []
        relevance_labels: list[torch.Tensor] = []
        for sample_idx, instance in enumerate(instances):
            documents_for_sample = instance["document"]
            positive_index = int(ranking[sample_idx, 0].item())
            negative_indices = [
                document_idx
                for document_idx in range(len(documents_for_sample))
                if document_idx != positive_index
            ]

            positive_documents.append(documents_for_sample[positive_index])
            negative_documents.extend(documents_for_sample[document_idx] for document_idx in negative_indices)
            ordered_indices = torch.tensor(
                [positive_index, *negative_indices],
                device=original_relevance_labels.device,
                dtype=torch.long,
            )
            relevance_labels.append(original_relevance_labels[sample_idx].gather(dim=0, index=ordered_indices))

        documents = [doc + self.tokenizer.pad_token for doc in [*positive_documents, *negative_documents]]
        document_inputs = self.tokenizer(
            documents,
            padding=True,
            truncation=True,
            max_length=self.doc_max_length,
            return_tensors="pt",
        )

        batch_size = len(instances)
        return {
            "query": query_inputs,
            "positive_document": {
                key: value[:batch_size]
                for key, value in document_inputs.items()
            },
            "negative_document": {
                key: value[batch_size:]
                for key, value in document_inputs.items()
            },
            "relevance_labels": torch.stack(relevance_labels),
        }
