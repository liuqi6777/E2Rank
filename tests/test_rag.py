import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from types import SimpleNamespace

from rag.candidates import (
    build_candidate_record,
    force_evidence_into_candidates,
    map_hotpot_evidence,
)
from rag.data import (
    RAGQueryCollator,
    EVALUATION_SUITE,
    EXPECTED_EVALUATION_TOTAL,
    EXPECTED_TRAIN_TOTAL,
    TRAIN_SOURCES,
    belongs_to_tuning_split,
)
from rag.generator import FrozenGeneratorClient
from rag.index import FrozenDistributedIndex, sha256_file
from rag.metrics import exact_match, extract_evidence_groups, max_token_f1, passage_contains_answer
from rag.models import RAGSupervisedModel, multi_positive_infonce_loss, positive_negative_ranknet_loss
from rag.policy import QueryPolicyHead
from rag.protocol import validate_query_index_protocol
from rag.rewards import answer_mrr_from_mask, source_aware_mrr


class RAGProtocolTest(unittest.TestCase):
    def test_published_suite_counts(self):
        self.assertEqual(sum(value[1] for value in TRAIN_SOURCES.values()), EXPECTED_TRAIN_TOTAL)
        self.assertEqual(sum(value[1] for value in EVALUATION_SUITE.values()), EXPECTED_EVALUATION_TOTAL)
        self.assertEqual(EXPECTED_TRAIN_TOTAL, 169615)
        self.assertEqual(EXPECTED_EVALUATION_TOTAL, 51713)

    def test_hash_split_is_deterministic_and_source_aware(self):
        first = belongs_to_tuning_split("nq", "train_7", 0.05, 42)
        self.assertEqual(first, belongs_to_tuning_split("nq", "train_7", 0.05, 42))
        values = [belongs_to_tuning_split("nq", f"id-{i}", 0.05, 42) for i in range(1000)]
        self.assertGreater(sum(values), 20)
        self.assertLess(sum(values), 80)

    def test_query_collator_appends_one_token_per_query(self):
        class Tokenizer:
            pad_token = "<pad>"

            def __call__(self, texts, **kwargs):
                self.texts = texts
                return {"input_ids": torch.ones((len(texts), 2), dtype=torch.long), "attention_mask": torch.ones((len(texts), 2), dtype=torch.long)}

        tokenizer = Tokenizer()
        collator = RAGQueryCollator(tokenizer, append_token="pad")
        records = [{
            "query_id": "q", "source": "nq", "question": "question",
            "formatted_query": "formatted", "golden_answers": ["answer"],
            "candidate_passage_ids": [1, 2], "answer_positive_mask": [True, False],
            "evidence_passage_groups": [],
        }]
        collator(records)
        self.assertEqual(tokenizer.texts, ["formatted<pad>"])

    def test_query_and_document_embedding_spaces_must_match(self):
        manifest = {
            "model_name_or_path": "Qwen/E0", "resolved_model_revision": "abc",
            "pooling_method": "last", "padding_side": "left", "append_token": "pad",
        }
        validate_query_index_protocol(
            manifest, model_name_or_path="Qwen/E0", resolved_model_revision="abc",
            pooling_method="last", padding_side="left", append_token="pad",
        )
        with self.assertRaises(ValueError):
            validate_query_index_protocol(
                manifest, model_name_or_path="intfloat/e5", resolved_model_revision="abc",
                pooling_method="last", padding_side="left", append_token="pad",
            )

    def test_answer_alias_metrics(self):
        self.assertTrue(passage_contains_answer("The population was 2,718 people.", ["2718", "2,718"]))
        # FlashRAG retrieval recall uses normalized substring matching.
        self.assertTrue(passage_contains_answer("The number was 27180.", ["2718"]))
        self.assertEqual(exact_match("Arthur's Magazine", ["Arthur's Magazine", "Arthur Magazine"]), 1.0)
        self.assertAlmostEqual(max_token_f1("Małgorzata Braunek", ["Braunek", "Małgorzata Braunek"]), 1.0)

    def test_hotpot_and_musique_evidence_extraction(self):
        hotpot = {
            "metadata": {
                "supporting_facts": {"title": ["A", "B"], "sent_id": [1, 0]},
                "context": {"title": ["A", "B"], "sentences": [["x", "gold a"], ["gold b"]]},
            }
        }
        self.assertEqual([g["sentence"] for g in extract_evidence_groups(hotpot)], ["gold a", "gold b"])
        musique = {
            "metadata": {"question_decomposition": [{
                "support_paragraph": {"title": "Green", "paragraph_text": "support", "is_supporting": True}
            }]}
        }
        self.assertEqual(extract_evidence_groups(musique)[0]["sentence"], "support")

    def test_hotpot_mapping_threshold_and_forcing(self):
        record = {
            "metadata": {
                "supporting_facts": {"title": ["Alpha", "Beta"], "sent_id": [0, 0]},
                "context": {"title": ["Alpha", "Beta"], "sentences": [["one exact sentence"], ["two fact"]]},
            }
        }
        catalog = {
            "alpha": [(9, "Alpha\none exact sentence"), (2, "Alpha\none exact sentence")],
            "beta": [(7, "Beta\ntwo fact")],
        }
        self.assertEqual(map_hotpot_evidence(record, catalog), [[2], [7]])
        self.assertEqual(force_evidence_into_candidates([0, 1, 3, 4], [[2], [7]]), [0, 1, 2, 7])
        self.assertIsNone(map_hotpot_evidence(record, {"alpha": catalog["alpha"]}))

    def test_candidate_masks_keep_nq_and_evidence_semantics_separate(self):
        nq = {"id": "q", "question": "q", "golden_answers": ["answer"]}
        self.assertIsNone(build_candidate_record("nq", nq, [1], ["irrelevant"]))
        kept = build_candidate_record("nq", nq, [1, 2], ["has answer", "other"])
        self.assertEqual(kept["training_positive_mask"], [True, False])
        hotpot = build_candidate_record(
            "hotpotqa", nq, [1, 2], ["answer", "other"], [[2]]
        )
        self.assertEqual(hotpot["answer_positive_mask"], [True, False])
        self.assertEqual(hotpot["training_positive_mask"], [False, True])

    def test_rewards_cutoff_aliases_and_multihop_difference(self):
        answer_mask = torch.tensor([
            [[False, True, False], [True, False, False]],
            [[True, False, False], [True, False, False]],
        ])
        answer = answer_mrr_from_mask(answer_mask, k=2)
        evidence_hits = torch.zeros((2, 2, 2, 3), dtype=torch.bool)
        evidence_hits[1, :, 0, 0] = True
        evidence_hits[1, 0, 1, 1] = True
        source = source_aware_mrr(
            answer_mask,
            ["nq", "hotpotqa"],
            evidence_hits,
            evidence_group_counts=[0, 2],
            k=2,
        )
        torch.testing.assert_close(source[0], answer[0])
        self.assertAlmostEqual(source[1, 0].item(), 0.75)
        self.assertAlmostEqual(source[1, 1].item(), 0.5)
        self.assertEqual(answer_mrr_from_mask(torch.tensor([[False, False, True]]), k=2).item(), 0.0)

    def test_supervised_losses_and_query_gradient(self):
        scores = torch.tensor([[0.1, 0.4, -0.2, 0.3]], requires_grad=True)
        positives = torch.tensor([[False, True, False, True]])
        info = multi_positive_infonce_loss(scores, positives, 0.03).mean()
        rank = positive_negative_ranknet_loss(scores, positives, 0.03).mean()
        (info + rank).backward()
        self.assertIsNotNone(scores.grad)
        self.assertEqual(info.ndim, 0)
        self.assertEqual(rank.ndim, 0)

    def test_supervised_wrapper_detaches_frozen_documents(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.eye(3))
                self.config = SimpleNamespace(hidden_size=3)

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(last_hidden_state=input_ids.float() @ self.weight)

        class Index:
            def __init__(self):
                self.documents = torch.tensor(
                    [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], requires_grad=True
                )

            def lookup_embeddings(self, ordinals):
                return self.documents

        index = Index()
        model = RAGSupervisedModel(Encoder(), index, "infonce", pooling_method="last")
        output = model(
            query={
                "input_ids": torch.tensor([[[1.0, 0.0, 0.0]]]),
                "attention_mask": torch.tensor([[1]]),
            },
            candidate_passage_ids=torch.tensor([[0, 1]]),
            training_positive_mask=torch.tensor([[True, False]]),
        )
        output.loss.backward()
        self.assertIsNotNone(model.model.weight.grad)
        self.assertIsNone(index.documents.grad)

    def test_policy_advantages_and_gradient(self):
        policy = QueryPolicyHead(group_size=3, kappa=10)
        means = torch.nn.functional.normalize(torch.randn(2, 5), dim=-1).requires_grad_()
        actions = policy.sample(means)
        output = policy.loss(means, actions, torch.tensor([[0.0, 0.5, 1.0], [1.0, 1.0, 1.0]]))
        output.loss.backward()
        self.assertIsNotNone(means.grad)
        self.assertAlmostEqual(output.degenerate_fraction.item(), 0.5)


class FrozenIndexTest(unittest.TestCase):
    def test_sharded_flat_ip_matches_bruteforce_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vectors = np.asarray([
                [1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5], [-1.0, 0.0]
            ], dtype=np.float16)
            shards = []
            for shard_id, (start, end) in enumerate(((0, 2), (2, 4))):
                path = root / f"vectors-{shard_id}.npy"
                np.save(path, vectors[start:end])
                shards.append({"path": path.name, "start": start, "count": end - start, "sha256": sha256_file(path)})
            corpus = root / "corpus.jsonl"
            offsets = []
            with open(corpus, "wb") as handle:
                for ordinal in range(4):
                    offsets.append(handle.tell())
                    handle.write((json.dumps({"id": str(ordinal), "contents": f"doc {ordinal}"}) + "\n").encode())
            np.save(root / "offsets.npy", np.asarray(offsets, dtype=np.int64))
            manifest = {
                "format_version": 1, "dimension": 2, "count": 4, "shards": shards,
                "corpus_path": corpus.name, "corpus_offsets_path": "offsets.npy",
                "corpus_sha256": sha256_file(corpus),
            }
            manifest_path = root / "index.json"
            manifest_path.write_text(json.dumps(manifest))
            before = [sha256_file(root / shard["path"]) for shard in shards]
            index = FrozenDistributedIndex(str(manifest_path), backend="torch", device="cpu")
            queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            scores, ids = index.search(queries, 3)
            brute = torch.from_numpy(vectors.astype(np.float32))
            expected_scores, expected_ids = (queries @ brute.T).topk(3, dim=-1)
            torch.testing.assert_close(scores, expected_scores)
            torch.testing.assert_close(ids, expected_ids)
            self.assertEqual(index.lookup_records([2])[0]["contents"], "doc 2")
            self.assertEqual(before, [sha256_file(root / shard["path"]) for shard in shards])

    def test_faiss_flat_ip_matches_bruteforce_when_available(self):
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss is installed on the GPU experiment host, not this dev host")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vectors = np.eye(3, dtype=np.float16)
            np.save(root / "vectors.npy", vectors)
            corpus = root / "corpus.jsonl"
            offsets = []
            with open(corpus, "wb") as handle:
                for ordinal in range(3):
                    offsets.append(handle.tell())
                    handle.write((json.dumps({"id": str(ordinal), "contents": str(ordinal)}) + "\n").encode())
            np.save(root / "offsets.npy", np.asarray(offsets, dtype=np.int64))
            manifest = {
                "format_version": 1, "dimension": 3, "count": 3,
                "shards": [{"path": "vectors.npy", "start": 0, "count": 3, "sha256": sha256_file(root / "vectors.npy")}],
                "corpus_path": "corpus.jsonl", "corpus_offsets_path": "offsets.npy",
            }
            (root / "index.json").write_text(json.dumps(manifest))
            index = FrozenDistributedIndex(str(root / "index.json"), backend="faiss", device="cpu")
            scores, ids = index.search(torch.tensor([[0.0, 1.0, 0.0]]), 3)
            expected_scores, expected_ids = torch.tensor([[0.0, 1.0, 0.0]]).topk(3, dim=-1)
            torch.testing.assert_close(scores, expected_scores)
            torch.testing.assert_close(ids, expected_ids)


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return json.dumps(messages, ensure_ascii=False)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}

    def decode(self, tokens, skip_special_tokens=False):
        return " ".join(map(str, tokens))


class GeneratorCacheTest(unittest.TestCase):
    def _client(self):
        client = object.__new__(FrozenGeneratorClient)
        client.endpoint = "http://generator"
        client.model = "frozen-generator"
        client.revision = "rev"
        client.resolved_revision = "rev"
        client.max_input_length = 2048
        client.max_new_tokens = 32
        client.timeout_seconds = 10
        client.system_prompt = "Answer only."
        client.tokenizer = _FakeTokenizer()
        client.connection = sqlite3.connect(":memory:")
        client.connection.execute("CREATE TABLE generations (cache_key TEXT PRIMARY KEY, output TEXT NOT NULL)")
        client.manifest_hash = "manifest"
        client.requests = 0
        client.cache_hits = 0
        client.endpoint_calls = 0
        return client

    def test_greedy_cached_generation_and_batch_deduplication(self):
        client = self._client()
        calls = []

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        def fake_open(request, timeout):
            body = json.loads(request.data)
            calls.append(body)
            return Response(json.dumps({"choices": [{"index": 0, "text": " answer "}]}).encode())

        item = ("q1", "question", [1, 2], [{"contents": "Title\ntext"}])
        with patch("urllib.request.urlopen", fake_open):
            self.assertEqual(client.generate_batch([item, item]), ["answer", "answer"])
            self.assertEqual(client.generate_batch([item]), ["answer"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["temperature"], 0)
        self.assertEqual(calls[0]["max_tokens"], 32)
        self.assertEqual(client.statistics()["generation_cache_hits"], 1)


if __name__ == "__main__":
    unittest.main()
