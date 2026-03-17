import json
import random
from collections import defaultdict
from typing import Any, Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset


TASK_PROMPTS = {
    "msmarco": (
        "Given a web search query, retrieval the documents that answer the query",
        "Given a web search query and some relevant documents, rerank the documents that answer the query",
    ),
    "nq": (
        "Given a question, retrieval Wikipedia documents that answer the question",
        "Given a question and some relevant documents, rerank the documents that answer the question",
    ),
    "hotpotqa": (
        "Given a multi-hop question, retrieval the documents that can help answer the question",
        "Given a multi-hop question and some relevant documents, rerank the documents that answer the question",
    ),
    "trivia": (
        "Retrieval Wikipedia documents that answer the question",
        "Given a question and some relevant Wikipedia documents, rerank the documents that answer the question",
    ),
    "t2ranking": (
        "Given a Chinese search query, retrieval the documents that answer the query",
        "Given a Chinese search query and some relevant documents, rerank the documents that answer the query",
    ),
    "dureader": (
        "Given a Chinese search query, retrieval the documents that answer the query",
        "Given a Chinese search query and some relevant documents, rerank the documents that answer the query",
    ),
    "mmarco_chinese": (
        "Given a Chinese web search query, retrieval the documents that answer the query",
        "Given a Chinese web search query and some relevant documents, rerank the documents that answer the query",
    ),
    "cMedQAv2": (
        "Given a Chinese medical question, retrieval the documents that answer the question",
        "Given a Chinese medical question and some relevant documents, rerank the documents that answer the question",
    ),
    "colliee": (
        "Given a legal case, retrieval the relevant legal articles that can help analyze the case",
        "Given a legal case and some relevant legal articles, rerank the legal articles that can help analyze the case",
    ),
    "law_gpt": (
        "Given a Chinese legal case, retrieval the relevant legal articles that can help analyze the case",
        "Given a Chinese legal case and some relevant legal articles, rerank the legal articles that can help analyze the case",
    ),
    "miracl": (
        "Given a question, retrieval Wikipedia documents that answer the question",
        "Given a question and some relevant Wikipedia documents, rerank the documents that answer the question",
    ),
}

DEFAULT_TASK_PROMPTS = (
    "Given a query, retrieval the documents that are relevant to the query",
    "Given a query and some relevant documents, rerank the documents that are the most relevant to the query",
)


class RankingDataset(Dataset):
    listwise_prompt_template = """{instruction}:
Documents:
{documents}
Query: {query}"""
    query_prompt_template = "Instruct: {task_description}\nQuery:{query}"

    def __init__(
        self,
        data_args: Any,
        batch_size: int = 32,
        per_dataset_max_samples: int = 10000,
    ):
        self.batch_size = batch_size
        self.per_dataset_max_samples = per_dataset_max_samples
        self.use_listwise = data_args.use_listwise
        self.samples: list[dict[str, Any]] = []
        self._load(data_args.data_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]

    def _format_query(self, task_name: str, query: str) -> str:
        retrieval_prompt = TASK_PROMPTS.get(task_name, DEFAULT_TASK_PROMPTS)[0]
        return self.query_prompt_template.format(
            task_description=retrieval_prompt,
            query=query,
        )

    def _format_listwise_prompt(self, task_name: str, query: str, docs: list[str]) -> str:
        rerank_prompt = TASK_PROMPTS.get(task_name, DEFAULT_TASK_PROMPTS)[1]
        return self.listwise_prompt_template.format(
            instruction=rerank_prompt,
            documents="\n".join([f"[{i}] {doc}" for i, doc in enumerate(docs, start=1)]),
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
                    "pseudo_query": self._format_listwise_prompt(task_name, sample["query"], sample["document"])
                    if self.use_listwise
                    else None,
                    "ranking": sample["ranking"],
                }
            )

        ordered_batches = []
        for task_name, indices in sample_indices_by_task.items():
            random.shuffle(indices)
            limited_indices = indices[: self.per_dataset_max_samples]
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

        if instances[0]["pseudo_query"] is not None:
            listwise_prompts = [instance["pseudo_query"] + self.tokenizer.pad_token for instance in instances]
            listwise_inputs = self.tokenizer.apply_chat_template(
                [[{"role": "user", "content": prompt}] for prompt in listwise_prompts],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                padding=True,
                truncation=True,
                max_length=32768,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            listwise_inputs = None

        ranking = torch.tensor([instance["ranking"] for instance in instances]) - 1
        return {
            "query": query_inputs,
            "document": document_inputs,
            "pseudo_query": listwise_inputs,
            "ranking": ranking,
        }
