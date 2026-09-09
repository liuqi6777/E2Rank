from __future__ import annotations

from collections.abc import Sequence

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from embedding_protocol import append_configured_token, format_embedding_text, pool_embeddings
from rag.data import RAG_TASK_DESCRIPTION


class FrozenQueryEncoder:
    """Inference-only query encoder using the same protocol as RAG training."""

    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: str | None = None,
        revision: str | None = None,
        device: str = "cuda",
        max_length: int = 128,
        pooling_method: str = "last",
        padding_side: str = "left",
        append_token: str = "pad",
        query_prompt_template: str = "Instruct: {task_description}\nQuery:{query}",
    ):
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.pooling_method = pooling_method
        self.append_token = append_token
        self.query_prompt_template = query_prompt_template
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            revision=revision,
            padding_side=padding_side,
            trust_remote_code=True,
        )
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model = AutoModel.from_pretrained(
            model_name_or_path,
            revision=revision,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        self.model = model.to(self.device).eval()

    def encode(self, questions: Sequence[str]) -> torch.Tensor:
        texts = [
            format_embedding_text(
                self.query_prompt_template,
                question,
                task_description=RAG_TASK_DESCRIPTION,
            )
            for question in questions
        ]
        texts = append_configured_token(texts, self.tokenizer, self.append_token)
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            return pool_embeddings(
                self.model(**inputs).last_hidden_state,
                inputs["attention_mask"],
                pooling_method=self.pooling_method,
                normalize=True,
            )
