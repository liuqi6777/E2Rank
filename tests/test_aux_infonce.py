"""CPU gradient checks; no model downloads, training data, or accelerator needed."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RLArguments
from contrastive import auxiliary_infonce_loss, aux_infonce_contract, compute_infonce_loss
from grpo import GRPO, GRPOModel
from grpo_trainer import GRPOTrainer, restore_exploration_state
from fixed_corpus.models import FixedCorpusGRPOModel, DynamicRetrievalGRPOModel


def test_multi_positive_loss_and_padding_gradients():
    scores = torch.tensor([[1.3, 0.9, -0.2, 99.0]], requires_grad=True)
    positives = torch.tensor([[True, True, False, False]])
    valid = torch.tensor([[True, True, True, False]])
    loss = compute_infonce_loss(scores, positives, temperature=0.5, candidate_mask=valid).sum()
    expected = (F.softplus(torch.tensor(-3.0)) + F.softplus(torch.tensor(-2.2))) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert (scores.grad[0, :2] < 0).all()
    assert scores.grad[0, 2] > 0
    assert scores.grad[0, 3] == 0


def test_in_batch_mask_and_detached_cross_document():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    documents = torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]]], requires_grad=True)
    positives = torch.ones(2, 1, dtype=torch.bool)
    cross = torch.tensor([[False, True], [False, False]])
    loss = auxiliary_infonce_loss(
        queries, documents, positives, temperature=1.0,
        use_in_batch_negatives=True, in_batch_positive_mask=cross,
    )
    torch.testing.assert_close(loss, F.softplus(torch.tensor(1.0)) / 2)
    loss.backward()
    assert queries.grad[0].norm() > 0
    assert documents.grad[0].norm() > 0
    assert queries.grad[1].norm() == 0
    assert documents.grad[1].norm() == 0  # Used by query 0 only as a detached negative.

    zero = auxiliary_infonce_loss(
        queries, documents, positives, use_in_batch_negatives=True,
        in_batch_positive_mask=torch.zeros_like(cross),
    )
    gradients = torch.autograd.grad(zero, (queries, documents))
    assert zero == 0
    assert all(torch.isfinite(g).all() and g.norm() == 0 for g in gradients)


def test_auxiliary_scores_stay_fp32_under_autocast():
    torch.manual_seed(8)
    queries = torch.randn(2, 7, dtype=torch.bfloat16, requires_grad=True)
    documents = torch.randn(2, 3, 7, dtype=torch.bfloat16, requires_grad=True)
    positives = torch.tensor([[True, True, False], [True, False, False]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = auxiliary_infonce_loss(queries, documents, positives, temperature=0.03)
    expected = compute_infonce_loss(
        torch.einsum("bd,bmd->bm", queries.float(), documents.float()), positives, 0.03,
    ).mean()
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    loss.backward()
    assert torch.isfinite(queries.grad).all() and torch.isfinite(documents.grad).all()


def test_auxiliary_requires_binary_positive_identities():
    with pytest.raises(ValueError):
        auxiliary_infonce_loss(torch.ones(1, 2), torch.ones(1, 2, 2), None)


@pytest.mark.parametrize("groups", [
    (("query",),), (("positive",),), (("negative",),),
    (("query",), ("positive",)), (("query",), ("positive", "negative")),
    (("query",), ("positive",), ("negative",)),
])
def test_zero_reward_advantage_still_gets_exact_auxiliary_gradient(groups):
    torch.manual_seed(5)
    query, positive, negative = (
        torch.randn(2, 5, requires_grad=True),
        torch.randn(2, 1, 5, requires_grad=True),
        torch.randn(2, 3, 5, requires_grad=True),
    )
    positives = torch.tensor([[True, False, True, False], [True, False, False, False]])
    valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    coefficient, temperature = 0.4, 0.2
    head = GRPO(
        action_components=groups, group_size=3, kappa=12, reward_type="mrr",
        aux_infonce_coef=coefficient, aux_infonce_temperature=temperature,
        rollout_seed=17,
    )
    values = dict(
        rollout_query_embeddings=query.detach(),
        rollout_positive_document_embeddings=positive.detach(),
        rollout_negative_document_embeddings=negative.detach(),
        policy_query_embeddings=query,
        policy_positive_document_embeddings=positive,
        policy_negative_document_embeddings=negative,
        # All candidates are relevant to this ranking reward: MRR is identically 1.
        # Binary auxiliary identities remain separate from these reward labels.
        relevance_labels=torch.ones(2, 4), positive_mask=positives, candidate_mask=valid,
    )
    loss, rewards, advantages, _, _ = head(**values)
    assert rewards["reward_mean"] == 1
    assert advantages["advantages_std"] == 0
    assert rewards["train/loss_rl"] == 0
    actual = torch.autograd.grad(loss, (query, positive, negative), allow_unused=True)

    active = {role for group in groups for role in group}
    q = F.normalize(query, dim=-1) if "query" in active else F.normalize(query.detach(), dim=-1)
    p = F.normalize(positive, dim=-1) if "positive" in active else F.normalize(positive.detach(), dim=-1)
    n = F.normalize(negative, dim=-1) if "negative" in active else F.normalize(negative.detach(), dim=-1)
    expected_loss = coefficient * auxiliary_infonce_loss(
        q, torch.cat((p, n), dim=1), positives, valid, temperature=temperature,
    )
    expected = torch.autograd.grad(expected_loss, (query, positive, negative), allow_unused=True)
    torch.testing.assert_close(loss, expected_loss)
    for name, got, want in zip(("query", "positive", "negative"), actual, expected):
        if name not in active:
            assert got is None and want is None
        else:
            assert got.norm() > 0
            torch.testing.assert_close(got, want)
    assert actual[2] is None or actual[2][0, 2].norm() == 0  # Padding never gets a gradient.


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.table = nn.Embedding(16, 5)
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.table(input_ids))


def tokens(ids):
    ids = torch.tensor(ids).reshape(-1, 1)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class TinyIndex:
    def __init__(self):
        self.embeddings = F.normalize(torch.randn(6, 5), dim=-1).requires_grad_()
        self.lookup_calls = 0

    def lookup_embeddings(self, ordinals, route_ids=None):
        self.lookup_calls += 1
        return self.embeddings[ordinals]

    def search(self, queries, k, route_ids=None):
        scores = queries @ self.embeddings.detach().T
        scores, ids = scores.topk(min(k, self.embeddings.size(0)), dim=-1)
        return scores, ids


@pytest.mark.parametrize("mode", ["joint", "frozen", "dynamic"])
def test_wrappers_reuse_forward_preserve_reward_and_freeze_index(mode):
    torch.manual_seed(1)
    model = TinyEncoder()
    index = TinyIndex()
    config = dict(
        action_components="query;positive,negative" if mode == "joint" else "query",
        group_size=3, kappa=12, reward_type="ndcg", dynamic_retrieval=mode == "dynamic",
        aux_infonce_temperature=0.2, aux_infonce_use_in_batch_negatives=True,
    )
    args = RLArguments(**config, aux_infonce_coef=0.3)
    cls = {"joint": GRPOModel, "frozen": FixedCorpusGRPOModel, "dynamic": DynamicRetrievalGRPOModel}[mode]
    wrapper = cls(model, rl_args=args, **({} if mode == "joint" else {"index": index}))
    batch = dict(
        query=tokens([0, 1]), relevance_labels=torch.tensor([[3., 1., 0.], [3., 0., 0.]]),
        positive_mask=torch.tensor([[True, True, False], [True, False, False]]),
        candidate_mask=torch.ones(2, 3, dtype=torch.bool),
        in_batch_positive_mask=~torch.eye(2, dtype=torch.bool),
    )
    if mode == "joint":
        batch.update(positive_document=tokens([2, 5]), negative_document=tokens([3, 4, 6, 7]))
    else:
        batch["candidate_ordinals"] = torch.arange(6).reshape(2, 3)
    torch.manual_seed(42)
    output = wrapper(**batch)
    assert model.calls == (2 if mode == "joint" else 1)
    auxiliary = output.reward_terms["train/loss_infonce_weighted"]
    assert auxiliary > 0
    torch.testing.assert_close(output.loss.detach(), output.reward_terms["train/loss_rl"] + auxiliary)
    output.loss.backward()
    assert model.table.weight.grad.norm() > 0
    assert index.embeddings.grad is None

    # Disabling the loss preserves exactly the same rollout reward and RL objective.
    wrapper.grpo.aux_infonce_coef = 0
    batch.pop("positive_mask")  # Disabled configs don't acquire a new input requirement.
    torch.manual_seed(42)
    disabled = wrapper(**batch)
    for name in ("reward_mean", "reward_std", "advantages_mean", "advantages_std"):
        torch.testing.assert_close(getattr(output, name), getattr(disabled, name), rtol=0, atol=0)
    torch.testing.assert_close(disabled.loss, output.reward_terms["train/loss_rl"], rtol=0, atol=0)


@pytest.mark.parametrize("cls", [FixedCorpusGRPOModel, DynamicRetrievalGRPOModel])
def test_frozen_auxiliary_excludes_other_index_routes(cls):
    torch.manual_seed(8)
    model, index = TinyEncoder(), TinyIndex()
    args = RLArguments(
        action_components="query", group_size=3, kappa=12, aux_infonce_coef=0.1,
        aux_infonce_temperature=0.2, aux_infonce_use_in_batch_negatives=True,
        dynamic_retrieval=cls is DynamicRetrievalGRPOModel,
    )
    wrapper = cls(model, index, args)
    ordinals = torch.arange(6).reshape(2, 3)
    positive_mask = torch.tensor([[True, False, False], [True, False, False]])
    output = wrapper(
        query=tokens([0, 1]), candidate_ordinals=ordinals,
        relevance_labels=positive_mask.float(), positive_mask=positive_mask,
        in_batch_positive_mask=torch.ones(2, 2, dtype=torch.bool),
        index_route_ids=torch.tensor([0, 1]),
    )
    # Both cross-query representatives are outside the query's searchable corpus.
    queries = F.normalize(model.table.weight[:2], dim=-1)
    documents = F.normalize(index.embeddings[ordinals].detach(), dim=-1)
    expected = auxiliary_infonce_loss(queries, documents, positive_mask, temperature=0.2)
    torch.testing.assert_close(output.reward_terms["train/loss_infonce"], expected)


@pytest.mark.parametrize("kwargs", [
    {"aux_infonce_coef": -1}, {"aux_infonce_coef": float("nan")},
    {"aux_infonce_coef": float("inf")}, {"aux_infonce_temperature": 0},
    {"aux_infonce_temperature": float("nan")},
])
def test_invalid_auxiliary_configuration(kwargs):
    with pytest.raises(ValueError):
        RLArguments(**kwargs)


def test_auxiliary_resume_contract(tmp_path):
    head = GRPO(group_size=3, aux_infonce_coef=0.2)
    model = SimpleNamespace(grpo=head)
    payload = {"exploration": head.exploration.state_dict(), "estimator": {}}
    path = tmp_path / "exploration_state.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        restore_exploration_state(model, tmp_path)  # Old RL-only checkpoint.
    payload["aux_infonce"] = aux_infonce_contract(head)
    path.write_text(json.dumps(payload))
    restore_exploration_state(model, tmp_path)
    head.aux_infonce_temperature = 0.1
    with pytest.raises(ValueError):
        restore_exploration_state(model, tmp_path)
    head.aux_infonce_coef = 0
    with pytest.raises(ValueError):
        restore_exploration_state(model, tmp_path)
    payload.pop("aux_infonce")
    path.write_text(json.dumps(payload))
    restore_exploration_state(model, tmp_path)


@pytest.mark.parametrize("estimator", ["score_function", "conditional_projection"])
def test_trainer_updates_logs_and_saves_auxiliary_contract(tmp_path, estimator):
    from transformers import BertConfig, BertModel, TrainingArguments

    torch.manual_seed(7)
    backbone = BertModel(BertConfig(
        vocab_size=16, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, hidden_dropout_prob=0, attention_probs_dropout_prob=0,
    ))
    wrapper = GRPOModel(backbone, RLArguments(
        action_components="query;positive,negative", group_size=2, kappa=12,
        aux_infonce_coef=0.3, aux_infonce_temperature=0.2,
        gradient_estimator=estimator,
    ))
    batch = dict(
        query=tokens([0, 1]), positive_document=tokens([2, 5]),
        negative_document=tokens([3, 4, 6, 7]),
        relevance_labels=torch.tensor([[1., 0., 0.], [1., 0., 0.]]),
        positive_mask=torch.tensor([[True, False, False], [True, False, False]]),
        candidate_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    before = backbone.embeddings.word_embeddings.weight.detach().clone()
    trainer = GRPOTrainer(
        model=wrapper,
        args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, max_steps=1,
            per_device_train_batch_size=2, learning_rate=1e-3, logging_steps=1,
            save_steps=1, report_to=[], disable_tqdm=True, remove_unused_columns=False,
            gradient_checkpointing=True,
        ),
        train_dataset=[0, 1], data_collator=lambda _: batch,
    )
    trainer.train()
    assert (backbone.embeddings.word_embeddings.weight.detach() - before).norm() > 0
    step = next(row for row in trainer.state.log_history if "train/loss_infonce" in row)
    assert step["train/loss_infonce"] > 0
    assert step["train/loss_infonce_weighted"] == pytest.approx(0.3 * step["train/loss_infonce"], abs=1e-4)
    checkpoint = tmp_path / "checkpoint-1"
    payload = json.loads((checkpoint / "exploration_state.json").read_text())
    assert payload["aux_infonce"]["coefficient"] == 0.3
    assert payload["estimator"]["gradient_estimator"] == estimator
    restore_exploration_state(wrapper, checkpoint)
