"""Behavioral protocol checks with real local tokenizers and a tiny Qwen backbone."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen3Config, Qwen3Model, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from config import ModelArguments
from embedding_data import EmbeddingDataCollator
from embedding_protocol import (
    TOKENIZATION_VERSION, load_embedding_protocol, pool_embeddings,
    tokenization_metadata, tokenize_embedding_texts,
)
from eval_mteb.qwen3_embedding_model import TransformersTextEmbedder
from eval_mteb.fixed_corpus_model import PerSubsetCorpusIndex, corpus_encoder_identity
from fixed_corpus.index import validate_frozen_protocol
from fixed_corpus.encode import encode_corpus_shards
from rag.data import RAGQueryCollator
from rag.encoder import FrozenQueryEncoder
from train_baseline import BaselineTrainer


def tokenizer(kind="embedding", padding_side="left"):
    vocabulary = {"[UNK]": 0, "[PAD]": 1, "word": 2, "other": 3, "[CLS]": 4, "[SEP]": 5}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    if kind == "embedding":
        backend.post_processor = TemplateProcessing(single="$A [PAD]", special_tokens=[("[PAD]", 1)])
    elif kind == "encoder":
        backend.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 4), ("[SEP]", 5)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                                  cls_token="[CLS]", sep_token="[SEP]", eos_token="[SEP]",
                                  padding_side=padding_side, model_input_names=["input_ids", "attention_mask"])


@pytest.mark.parametrize("kind", ["base", "embedding"])
@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("max_length", [1, 2, 4])
def test_one_attended_terminal_survives_truncation(kind, padding_side, max_length):
    tok = tokenizer(kind, padding_side)
    batch = tokenize_embedding_texts(["word", "word other word other word", ""], tok, "pad", max_length=max_length)
    expected = [[2][:max_length-1]+[1], [2, 3, 2, 3, 2][:max_length-1]+[1], [1]]
    for ids, mask, wanted in zip(batch["input_ids"], batch["attention_mask"], expected):
        assert ids[mask.bool()].tolist() == wanted
        assert (ids[~mask.bool()] == tok.pad_token_id).all()
    hidden = torch.arange(batch["input_ids"].numel()*3).reshape(*batch["input_ids"].shape, 3).float()
    pooled = pool_embeddings(hidden, batch["attention_mask"], pooling_method="last", normalize=False)
    last_positions = (batch["attention_mask"] * torch.arange(1, hidden.shape[1]+1)).argmax(-1)
    torch.testing.assert_close(pooled, hidden[torch.arange(3), last_positions])


def test_explicit_terminal_is_id_based_and_normalizes_existing_boundary():
    tok = tokenizer()
    batch = tokenize_embedding_texts(["word [PAD] [PAD]", "word [PAD] other"], tok, "pad", max_length=8)
    assert batch["input_ids"][0][batch["attention_mask"][0].bool()].tolist() == [2, 1]
    assert batch["input_ids"][1].tolist() == [2, 1, 3, 1]
    eos = tokenize_embedding_texts(["word"], tok, "eos", max_length=8)
    assert eos["input_ids"].tolist() == [[2, 5]]


def test_none_preserves_native_encoder_special_tokens():
    tok = tokenizer("encoder", "right")
    texts = ["word", "word other word other word"]
    actual = tokenize_embedding_texts(texts, tok, "none", max_length=4)
    native = tok(texts, padding=True, truncation=True, max_length=4, return_tensors="pt")
    for key in native:
        torch.testing.assert_close(actual[key], native[key])
    assert actual["input_ids"].tolist() == [[4, 2, 5, 1], [4, 2, 3, 5]]


def tiny_model():
    return Qwen3Model(Qwen3Config(vocab_size=6, hidden_size=16, intermediate_size=32,
                                num_hidden_layers=1, num_attention_heads=2,
                                num_key_value_heads=2, head_dim=8, max_position_embeddings=64)).eval()


def test_training_mteb_rag_and_native_embedding_agree():
    tok, model = tokenizer(), tiny_model()
    texts = ["word", "word other word other word"]
    records = [dict(query=text, document=["word", "word other word other word"],
                    ranking=[1, 2], pos_index=1) for text in texts]
    training = EmbeddingDataCollator(tok, query_max_length=4, doc_max_length=4)(records)
    with patch("eval_mteb.qwen3_embedding_model.AutoModel.from_pretrained", return_value=model), \
         patch("eval_mteb.qwen3_embedding_model.AutoTokenizer.from_pretrained", return_value=tok):
        evaluator = TransformersTextEmbedder("local-tiny", do_norm=True)
    evaluation = evaluator.tokenize(texts, max_length=4)
    native = tok(texts, padding=True, truncation=True, max_length=4, return_tensors="pt")
    rag = RAGQueryCollator(tok, query_max_length=4)([
        dict(formatted_query=t, candidate_passage_ids=[0, 1], answer_positive_mask=[True, False],
             query_id=str(i), source="toy", question=t, golden_answers=["word"])
        for i, t in enumerate(texts)
    ])
    for key in native:
        torch.testing.assert_close(training["query"][key], native[key])
        torch.testing.assert_close(evaluation[key], native[key])
        torch.testing.assert_close(rag["query"][key], native[key])
    documents = torch.cat([training["positive_document"]["input_ids"], training["negative_document"]["input_ids"]])
    assert documents.tolist() == [[1, 1, 2, 1], [1, 1, 2, 1], [2, 3, 2, 1], [2, 3, 2, 1]]
    with torch.inference_mode():
        expected = pool_embeddings(model(**native).last_hidden_state, native["attention_mask"], pooling_method="last")
        torch.testing.assert_close(evaluator(**evaluation), expected)
    with patch("rag.encoder.AutoModel.from_pretrained", return_value=model), \
         patch("rag.encoder.AutoTokenizer.from_pretrained", return_value=tok):
        rag_encoder = FrozenQueryEncoder("local-tiny", device="cpu", max_length=4, query_prompt_template="{query}")
    torch.testing.assert_close(rag_encoder.encode(texts), expected)


def test_corpus_shards_match_native_vectors_and_record_protocol(tmp_path):
    tok, model = tokenizer(), tiny_model()
    texts, offsets = ["word", "word other word other word", ""], []
    corpus_path = tmp_path / "corpus.jsonl"
    with corpus_path.open("wb") as stream:
        for text in texts:
            offsets.append(stream.tell())
            stream.write((json.dumps({"contents": text})+"\n").encode())
    offsets_path = tmp_path / "offsets.npy"
    np.save(offsets_path, np.array(offsets, dtype=np.int64))
    with patch("fixed_corpus.encode.AutoModel.from_pretrained", return_value=model), \
         patch("fixed_corpus.encode.AutoTokenizer.from_pretrained", return_value=tok):
        dimension, revision, shards, metadata = encode_corpus_shards(
            corpus_path=corpus_path, offsets_path=offsets_path, output_dir=tmp_path,
            model_name_or_path="local-tiny", revision=None, num_shards=1, batch_size=3,
            max_length=4, pooling_method="last", padding_side="left", append_token="pad",
            document_prompt_template="{document}", rank=0, world_size=1, device=torch.device("cpu"),
        )
    assert dimension == 16 and len(shards) == 1
    assert metadata == {**tokenization_metadata(tok, "pad"), "pooling_compute_dtype": "float32"}
    native = tok(texts, padding=True, truncation=True, max_length=4, return_tensors="pt")
    with torch.inference_mode():
        expected = pool_embeddings(model(**native).last_hidden_state, native["attention_mask"], pooling_method="last")
    np.testing.assert_array_equal(np.load(tmp_path / shards[0]["path"]), expected.numpy().astype(np.float16))


def test_checkpoint_save_roundtrip_without_eval_callback(tmp_path):
    tok = tokenizer()
    wrapper = torch.nn.Module()
    wrapper.model = tiny_model()
    trainer = BaselineTrainer(
        model=wrapper, model_args=ModelArguments(model_name_or_path="local-tiny"),
        processing_class=tok, args=TrainingArguments(output_dir=str(tmp_path), report_to=[], use_cpu=True),
    )
    checkpoint = tmp_path / "checkpoint-1"
    trainer.save_model(str(checkpoint))
    saved = load_embedding_protocol(checkpoint)
    restored = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    for key, expected in tokenization_metadata(restored, "pad").items():
        assert saved[key] == expected
    assert saved["pooling_compute_dtype"] == "float32"
    assert saved["add_special_tokens"] is False
    assert saved["terminal_token_id"] == 1
    assert saved["terminal_after_truncation"] is True
    batch = tokenize_embedding_texts(["word other word other"], restored, saved["append_token"], max_length=3)
    assert batch["input_ids"].tolist() == [[2, 3, 1]]
    no_precision = {key: value for key, value in saved.items() if key != "pooling_compute_dtype"}
    (checkpoint / "embedding_protocol.json").write_text(json.dumps(no_precision))
    with pytest.raises(ValueError, match="pooling precision"):
        load_embedding_protocol(checkpoint)
    saved.pop("tokenization_version")
    (checkpoint / "embedding_protocol.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="tokenization protocol"):
        load_embedding_protocol(checkpoint)


def test_old_frozen_indexes_are_rejected(tmp_path):
    expected = dict(model_name_or_path="tiny", resolved_model_revision=None,
                    pooling_method="last", padding_side="left", append_token="pad")
    with pytest.raises(ValueError, match="tokenization_version"):
        validate_frozen_protocol({}, **expected)
    with pytest.raises(ValueError, match="pooling_compute_dtype"):
        validate_frozen_protocol({"tokenization_version": TOKENIZATION_VERSION}, **expected)
    validate_frozen_protocol({"tokenization_version": TOKENIZATION_VERSION, "pooling_compute_dtype": "float32"}, **expected)

    embedder = SimpleNamespace(tokenizer=tokenizer(), append_token="pad", base_model=tiny_model())
    model = SimpleNamespace(model=embedder, mteb_model_meta=SimpleNamespace(revision="fixed-revision"))
    current_identity = corpus_encoder_identity("tiny", model)
    old_identity = {k: v for k, v in current_identity.items() if k != "tokenization_version"}
    old = PerSubsetCorpusIndex(tmp_path, task_name="toy", encoder_identity=old_identity)
    old.begin("subset")
    old.encode(["word"], lambda: np.ones((1, 16), dtype=np.float32))
    old.finish()
    current = PerSubsetCorpusIndex(tmp_path, task_name="toy", encoder_identity=current_identity)
    with pytest.raises(ValueError, match="protocol mismatch"):
        current.begin("subset")


def test_frozen_runner_preflight_enforces_pooling_precision(tmp_path):
    from scripts.experiments.iclr2027 import _check_single_frozen_document_index, digest

    artifacts = {"corpus": "{}\n", "offsets": "0", "document_key_to_ordinal": '{"doc": 0}', "shard": "vector"}
    for name, contents in artifacts.items():
        (tmp_path / name).write_text(contents)
    config = dict(model_name_or_path="tiny", pooling_method="last", padding_side="left", append_token="pad",
                  document_prompt_template="{document}", query_prompt_template="{query}", d_max_len=8, embedding_max_length=8)
    manifest = dict(format_version=1, dimension=2, count=1, tokenization_version=TOKENIZATION_VERSION,
                    pooling_compute_dtype="float32", document_max_length=8,
                    **{key: value for key, value in config.items() if key != "d_max_len"})
    for name in ("corpus", "corpus_offsets", "document_key_to_ordinal"):
        filename = "offsets" if name == "corpus_offsets" else name
        manifest[f"{name}_path"] = filename
        manifest[f"{name}_sha256"] = digest(tmp_path / filename)
    manifest["shards"] = [{"path": "shard", "sha256": digest(tmp_path / "shard")}]
    path = tmp_path / "index_manifest.json"
    assert _check_single_frozen_document_index(path, manifest, config) == []
    manifest.pop("pooling_compute_dtype")
    assert len(_check_single_frozen_document_index(path, manifest, config)) == 1
