from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "Answer the question based on the given documents. "
    "Only give me the answer and do not output any other words."
)

# vLLM rejects a /v1/completions call carrying more prompts than
# VLLM_MAX_COMPLETION_PROMPTS, which defaults to 1024. RL asks for one generation
# per rollout, so a single step easily exceeds that; batches are split to match.
MAX_PROMPTS_PER_REQUEST = 1024

# Parameters per "cache_key IN (...)" lookup. SQLITE_MAX_VARIABLE_NUMBER is 32766
# on modern builds but only 999 on older ones, and the round trip is what costs,
# not the width, so this stays well under both.
MAX_KEYS_PER_SELECT = 500

# Rows held in memory. An evaluation's cache is around a hundred thousand of them,
# but training appends one per rollout and runs for as long as the campaign does,
# so a table past this size is left on disk and reached through batched lookups.
MAX_PRELOADED_ROWS = 4_000_000


class FrozenGeneratorClient:
    """Cached client for a vLLM OpenAI-compatible endpoint.

    Greedy decoding does not make the endpoint reproducible: continuous batching
    varies the reduction order, and on this corpus about 2-3% of answers change
    when the same prompts are grouped differently, moving aggregate EM by ~0.2.
    A non-empty ``cache_path`` pins the first answer per (question, passage-set)
    in sqlite, which is what makes an evaluation replayable. An empty
    ``cache_path`` disables the cache entirely: training rollouts essentially
    never repeat a pair (measured ~0% hits), and on network storage the per-row
    INSERT costs ~80 ms through FUSE -- tens of hours over a 3M-rollout epoch.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        cache_path: str = "",
        revision: str | None = None,
        max_input_length: int = 2048,
        max_new_tokens: int = 32,
        timeout_seconds: int = 600,
        system_prompt: str = SYSTEM_PROMPT,
        max_prompts_per_request: int = MAX_PROMPTS_PER_REQUEST,
    ):
        if not endpoint:
            raise ValueError("A generator endpoint is required for answer_f1 reward")
        if max_prompts_per_request <= 0:
            raise ValueError("max_prompts_per_request must be positive")
        # A comma-separated list runs one co-located vLLM instance per GPU
        # group; batches are dispatched across them concurrently (below).
        self.endpoints = [part.rstrip("/") for part in str(endpoint).split(",") if part.strip()]
        if not self.endpoints:
            raise ValueError("A generator endpoint is required for answer_f1 reward")
        self.endpoint = self.endpoints[0]
        self.model = model
        self.revision = revision
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds
        self.system_prompt = system_prompt
        self.max_prompts_per_request = int(max_prompts_per_request)
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=True)
        self.resolved_revision = getattr(self.tokenizer, "_commit_hash", None) or revision
        self._lock = threading.Lock()
        self._statistics_lock = threading.Lock()
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
        if not cache_path:
            self._cache: dict[str, str] | None = None
            self.connection = None
            return
        self._cache = {}
        # A caller may drive generation from a worker thread to overlap it with
        # GPU work, so the cache is shared rather than pinned to the creating
        # thread. Every statement below runs under _lock; the counters have
        # their own.
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(cache_file, timeout=120, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS generations (cache_key TEXT PRIMARY KEY, output TEXT NOT NULL)"
        )
        self.connection.commit()
        # The whole table is read once and answered from memory afterwards. These
        # caches live on network storage, where a single-row lookup is a filesystem
        # round trip rather than a b-tree descent: measured on JuiceFS with a 18MB
        # 102k-row cache, one SELECT per key costs 38ms against 6us for the same
        # file on local disk, so a seven-dataset evaluation spent about half an hour
        # of its generation phase waiting on point queries while the endpoint idled.
        # Reading every row instead takes 0.54s and answers all 51,713 lookups in 9ms.
        # Past MAX_PRELOADED_ROWS the reads go to the file, batched; _stored_outputs
        # is what makes both routes return the same answers.
        rows = self.connection.execute("SELECT count(*) FROM generations").fetchone()[0]
        if rows <= MAX_PRELOADED_ROWS:
            self._cache = dict(self.connection.execute("SELECT cache_key, output FROM generations"))

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

    def _format(self, question: str, documents: Sequence[dict]) -> str:
        """Lay out one prompt; whether it fits the budget is decided in batch."""
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
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _token_lengths(self, prompts: Sequence[str]) -> list[int]:
        """Measure a whole group of prompts in one call.

        The fast tokenizer threads a batch inside Rust, so asking it once per prompt
        costs about three times as much as asking once for all of them. Laying the
        prompt out is cheap by comparison; measuring it is what dominated rendering.
        """
        encoded = self.tokenizer(list(prompts), add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in encoded]

    def _truncate(self, question: str, prompt: str) -> str:
        """Keep a prefix of the last remaining document, plus the question."""
        tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        encoded_question = self.tokenizer(
            f"\nQuestion:{question}", add_special_tokens=False
        )["input_ids"]
        kept = tokens[: max(self.max_input_length - len(encoded_question), 1)] + encoded_question
        return self.tokenizer.decode(kept[: self.max_input_length], skip_special_tokens=False)

    def _render_batch(
        self, inputs: Sequence[tuple[str, Sequence[dict]]]
    ) -> list[str]:
        """Render prompts that respect the input budget, measuring them in batches.

        Corpus passages are ordered by retrieval score, so a prompt over the budget
        drops its lowest-ranked context and is measured again; one already down to a
        single document keeps a prefix of it instead. Each round only re-measures the
        prompts still over budget, so the usual case where none are costs one call.
        """
        prompts: list[str | None] = [None] * len(inputs)
        kept = [list(documents) for _, documents in inputs]
        pending = list(range(len(inputs)))
        while pending:
            candidates = [self._format(inputs[index][0], kept[index]) for index in pending]
            lengths = self._token_lengths(candidates)
            overlong = []
            for position, index in enumerate(pending):
                if lengths[position] <= self.max_input_length:
                    prompts[index] = candidates[position]
                elif len(kept[index]) > 1:
                    kept[index].pop()
                    overlong.append(index)
                else:
                    prompts[index] = self._truncate(inputs[index][0], candidates[position])
            pending = overlong
        return [str(prompt) for prompt in prompts]

    def _stored_outputs(self, keys: Sequence[str]) -> dict[str, str]:
        """Re-read keys the in-memory snapshot does not hold.

        A key absent at startup may have been written by another process since;
        asking the file again keeps that behaviour, at one round trip per batch
        rather than one per key.
        """
        if self.connection is None:
            return {}
        found: dict[str, str] = {}
        with self._lock:
            for start in range(0, len(keys), MAX_KEYS_PER_SELECT):
                window = keys[start : start + MAX_KEYS_PER_SELECT]
                placeholders = ",".join("?" * len(window))
                found.update(self.connection.execute(
                    f"SELECT cache_key, output FROM generations WHERE cache_key IN ({placeholders})",
                    window,
                ))
            self._cache.update(found)
        return found

    def generate_batch(
        self,
        requests: Sequence[tuple[str, str, Sequence[int], Sequence[dict]]],
    ) -> list[str]:
        if self._cache is None:
            # Cache-free mode: no lookup, no key hashing, no storage.
            prompts = self._render_batch(
                [(question, documents) for _, question, _, documents in requests]
            )
            generated = self._complete_all(prompts)
            with self._statistics_lock:
                self.requests += len(requests)
            return generated
        outputs: list[str | None] = [None] * len(requests)
        # Hashing a request's context is pure CPU and touches no shared state, so it
        # stays outside the lock; with the lookups served from memory it would
        # otherwise be nearly the whole critical section.
        keys = [
            self.cache_key(query_id, passage_ids, documents)
            for query_id, _, passage_ids, documents in requests
        ]
        missing_indices_by_key: dict[str, list[int]] = {}
        missing_keys: list[str] = []
        missing_inputs: list[tuple[str, Sequence[dict]]] = []
        with self._lock:
            for index, key in enumerate(keys):
                output = self._cache.get(key)
                if output is not None:
                    outputs[index] = output
                elif key in missing_indices_by_key:
                    missing_indices_by_key[key].append(index)
                else:
                    missing_indices_by_key[key] = [index]
                    missing_keys.append(key)
                    missing_inputs.append((requests[index][1], requests[index][3]))

        if missing_keys:
            stored = self._stored_outputs(missing_keys)
            if stored:
                for key in stored:
                    for index in missing_indices_by_key.pop(key):
                        outputs[index] = stored[key]
                surviving = [
                    (key, request_input)
                    for key, request_input in zip(missing_keys, missing_inputs)
                    if key not in stored
                ]
                missing_keys = [key for key, _ in surviving]
                missing_inputs = [request_input for _, request_input in surviving]

        with self._statistics_lock:
            self.requests += len(requests)
            self.cache_hits += sum(output is not None for output in outputs)

        if missing_inputs:
            # Rendering tokenizes every prompt, so it stays outside the cache lock.
            missing_prompts = self._render_batch(missing_inputs)
            generated = self._complete_all(missing_prompts)
            with self._lock:
                self.connection.executemany(
                    "INSERT OR REPLACE INTO generations(cache_key, output) VALUES (?, ?)",
                    zip(missing_keys, generated),
                )
                self.connection.commit()
                self._cache.update(zip(missing_keys, generated))
            for key, output in zip(missing_keys, generated):
                for index in missing_indices_by_key[key]:
                    outputs[index] = output
        return [str(output) for output in outputs]

    def _complete(self, prompts: Sequence[str], endpoint: str | None = None) -> list[str]:
        """Send one batch of prompts the server is willing to accept at once."""
        # Counted under its own lock: _lock is held across a commit, and waiting on
        # that here would delay the request the endpoint is idle waiting for.
        with self._statistics_lock:
            self.endpoint_calls += 1
        body = json.dumps({
            "model": self.model,
            "prompt": list(prompts),
            "temperature": 0,
            "max_tokens": self.max_new_tokens,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{endpoint or self.endpoint}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.load(response)
        choices = sorted(payload["choices"], key=lambda choice: int(choice.get("index", 0)))
        if len(choices) != len(prompts):
            raise RuntimeError(
                f"Generator returned {len(choices)} choices for {len(prompts)} prompts"
            )
        return [choice["text"].strip() for choice in choices]

    def _complete_all(self, prompts: Sequence[str]) -> list[str]:
        """Send every server-sized batch, splitting across the endpoints.

        One 1,024-prompt request already saturates a vLLM instance, so the
        batches are dealt round-robin to the endpoints and sent concurrently --
        sequential round-robin would leave each server idle while it waits for
        the other's turn. Results are reassembled in prompt order.
        """
        batches = [
            prompts[start : start + self.max_prompts_per_request]
            for start in range(0, len(prompts), self.max_prompts_per_request)
        ]
        if len(self.endpoints) == 1 or len(batches) == 1:
            generated = []
            for batch in batches:
                generated.extend(self._complete(batch))
            return generated
        with ThreadPoolExecutor(max_workers=len(batches)) as pool:
            futures = [
                pool.submit(self._complete, batch, self.endpoints[index % len(self.endpoints)])
                for index, batch in enumerate(batches)
            ]
            # future.result() blocks in submission order while every batch runs.
            return [output for future in futures for output in future.result()]

    def statistics(self) -> dict[str, float | int]:
        with self._statistics_lock:
            return {
                "generator_manifest_hash": self.manifest_hash,
                "generation_requests": self.requests,
                "generation_cache_hits": self.cache_hits,
                "generation_cache_hit_rate": self.cache_hits / max(self.requests, 1),
                "generator_endpoint_calls": self.endpoint_calls,
            }

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
