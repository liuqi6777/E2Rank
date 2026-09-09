from __future__ import annotations

import hashlib
import json
import sqlite3
import urllib.request
from pathlib import Path
from typing import Sequence

from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "Answer the question based on the given documents. "
    "Only give me the answer and do not output any other words."
)


class FrozenGeneratorClient:
    """Deterministic, cached client for a vLLM OpenAI-compatible endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        cache_path: str,
        revision: str | None = None,
        max_input_length: int = 2048,
        max_new_tokens: int = 32,
        timeout_seconds: int = 600,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        if not endpoint:
            raise ValueError("A generator endpoint is required for answer_f1 reward")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.revision = revision
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=True)
        self.resolved_revision = getattr(self.tokenizer, "_commit_hash", None) or revision
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(cache_file, timeout=120)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS generations (cache_key TEXT PRIMARY KEY, output TEXT NOT NULL)"
        )
        self.connection.commit()
        self.requests = 0
        self.cache_hits = 0
        self.endpoint_calls = 0
        self.manifest_hash = hashlib.sha256(json.dumps({
            "model": model,
            "revision": self.resolved_revision,
            "system_prompt": self.system_prompt,
            "prompt_format_version": 1,
            "context_truncation": "drop_lowest_ranked_then_prefix",
            "max_input_length": max_input_length,
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
        }, sort_keys=True).encode("utf-8")).hexdigest()

    def cache_key(
        self,
        query_id: str,
        passage_ids: Sequence[int],
        documents: Sequence[dict] | None = None,
    ) -> str:
        context_hash = hashlib.sha256(
            json.dumps(
                [document.get("contents", "") for document in (documents or [])],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        payload = f"{self.manifest_hash}:{query_id}:{','.join(map(str, passage_ids))}:{context_hash}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _render(self, question: str, documents: Sequence[dict]) -> str:
        doc_blocks = []
        for index, document in enumerate(documents, start=1):
            contents = str(document.get("contents", ""))
            lines = contents.splitlines()
            title = document.get("title") or (lines[0].strip('"') if lines else "")
            text = "\n".join(lines[1:]) if len(lines) > 1 else contents
            doc_blocks.append(f"Doc {index} (Title:{title}) {text}")
        joined_documents = "\n".join(doc_blocks)
        user_content = f"The following are given documents:\n{joined_documents}\nQuestion:{question}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(tokens) <= self.max_input_length:
            return prompt
        # Corpus passages are ordered by retrieval score. Drop the lowest-ranked
        # contexts first; if one document remains, retain its prefix and the question.
        if len(documents) > 1:
            return self._render(question, documents[:-1])
        encoded_question = self.tokenizer(
            f"\nQuestion:{question}", add_special_tokens=False
        )["input_ids"]
        kept = tokens[: max(self.max_input_length - len(encoded_question), 1)] + encoded_question
        return self.tokenizer.decode(kept[: self.max_input_length], skip_special_tokens=False)

    def generate_batch(
        self,
        requests: Sequence[tuple[str, str, Sequence[int], Sequence[dict]]],
    ) -> list[str]:
        outputs: list[str | None] = [None] * len(requests)
        self.requests += len(requests)
        missing_indices_by_key: dict[str, list[int]] = {}
        missing_prompts = []
        missing_keys = []
        for index, (query_id, question, passage_ids, documents) in enumerate(requests):
            key = self.cache_key(query_id, passage_ids, documents)
            row = self.connection.execute(
                "SELECT output FROM generations WHERE cache_key = ?", (key,)
            ).fetchone()
            if row is not None:
                outputs[index] = row[0]
                self.cache_hits += 1
            else:
                if key in missing_indices_by_key:
                    missing_indices_by_key[key].append(index)
                else:
                    missing_indices_by_key[key] = [index]
                    missing_keys.append(key)
                    missing_prompts.append(self._render(question, documents))

        if missing_prompts:
            self.endpoint_calls += 1
            body = json.dumps({
                "model": self.model,
                "prompt": missing_prompts,
                "temperature": 0,
                "max_tokens": self.max_new_tokens,
            }).encode("utf-8")
            request = urllib.request.Request(
                f"{self.endpoint}/v1/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.load(response)
            choices = sorted(payload["choices"], key=lambda choice: int(choice.get("index", 0)))
            if len(choices) != len(missing_prompts):
                raise RuntimeError(
                    f"Generator returned {len(choices)} choices for {len(missing_prompts)} prompts"
                )
            generated = [choice["text"].strip() for choice in choices]
            self.connection.executemany(
                "INSERT OR REPLACE INTO generations(cache_key, output) VALUES (?, ?)",
                zip(missing_keys, generated),
            )
            self.connection.commit()
            for key, output in zip(missing_keys, generated):
                for index in missing_indices_by_key[key]:
                    outputs[index] = output
        return [str(output) for output in outputs]

    def statistics(self) -> dict[str, float | int]:
        return {
            "generator_manifest_hash": self.manifest_hash,
            "generation_requests": self.requests,
            "generation_cache_hits": self.cache_hits,
            "generation_cache_hit_rate": self.cache_hits / max(self.requests, 1),
            "generator_endpoint_calls": self.endpoint_calls,
        }

    def close(self) -> None:
        self.connection.close()
