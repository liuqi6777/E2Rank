import json
import random
from collections import defaultdict
from typing import Any, Dict, Iterator, Sequence

import torch
import transformers
from torch.utils.data import Dataset, Sampler


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
        split: str = "train",
    ):
        if split not in {"train", "dev"}:
            raise ValueError(f"split must be 'train' or 'dev', got {split!r}")
        self.batch_size = batch_size or 32
        self.split = split
        self.per_dataset_max_samples = data_args.per_dataset_max_samples
        self.dev_samples_per_source = getattr(data_args, "dev_samples_per_source", 0)
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
            # Seeded by the caller's set_seed(), so the same seed yields the same split.
            random.shuffle(indices)
            # Carve the dev slice off the FRONT, before the training cap, so changing
            # per_dataset_max_samples can never leak a dev example into training.
            if self.dev_samples_per_source > 0:
                dev_indices = indices[: self.dev_samples_per_source]
                train_indices = indices[self.dev_samples_per_source :]
            else:
                dev_indices, train_indices = [], indices
            if self.split == "dev":
                limited_indices = dev_indices
            else:
                limited_indices = (
                    train_indices
                    if self.per_dataset_max_samples is None
                    else train_indices[: self.per_dataset_max_samples]
                )
            for start in range(0, len(limited_indices), self.batch_size):
                batch = limited_indices[start : start + self.batch_size]
                if len(batch) == self.batch_size:
                    ordered_batches.append(batch)
                else:
                    print(f"Skip 1 {self.split} batch for dataset {task_name}.")

        random.shuffle(ordered_batches)
        ordered_indices = [idx for batch in ordered_batches for idx in batch]
        self.samples = [normalized_samples[idx] for idx in ordered_indices]
        self.num_batches = len(ordered_batches)
        print(
            f"Loaded {len(self.samples)} {self.split} samples in "
            f"{self.num_batches} single-source batches."
        )


class SingleSourceBatchSampler(Sampler[int]):
    """Sample-level sampler that preserves ``RankingDataset``'s per-source batching.

    ``RankingDataset`` lays its samples out as consecutive blocks of ``batch_size``
    drawn from a single source, so that every micro-batch shares one task prompt and
    in-batch negatives stay in-domain. The HF Trainer's default ``RandomSampler``
    shuffles at the *sample* level and silently destroys that layout, mixing every
    source into every batch. This sampler shuffles whole blocks instead, keeping the
    intra-block order intact.

    The block permutation is derived from ``seed + epoch`` only, so every rank walks
    the same global batch order; accelerate then hands whole batches to ranks
    round-robin (``split_batches=False``), and each rank still sees single-source
    micro-batches.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        num_samples = len(self.dataset)
        num_blocks = num_samples // self.batch_size
        if self.shuffle and num_blocks > 1:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            block_order = torch.randperm(num_blocks, generator=generator).tolist()
        else:
            block_order = range(num_blocks)

        for block in block_order:
            yield from range(block * self.batch_size, (block + 1) * self.batch_size)
        # RankingDataset drops partial per-source batches, so this is normally empty.
        yield from range(num_blocks * self.batch_size, num_samples)


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
