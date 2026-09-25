"""Exercise the diagnostic on a small, real Qwen3 Transformer architecture."""

from pathlib import Path
import sys

import pytest
import torch
from transformers import Qwen3Config, Qwen3Model

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import diagnose_rollout_gradients as probe
from config import RLArguments
from grpo import GRPOModel


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shared_documents,cross_device", [(False, False), (False, True), (True, False), (True, True)])
def test_paired_qwen3_full_gradients_and_online_moments(dtype, shared_documents, cross_device, monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(18)
        backbone = Qwen3Model(Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=32, attention_dropout=0., use_cache=False,
        ))
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = GRPOModel(backbone, RLArguments(action_components="query;positive,negative",
                          group_size=6, kappa=18, reward_type="mrr_in_batch", rollout_seed=19,
                          cross_query_document_gradients=shared_documents,
                          reward_cross_device_negatives=cross_device,
                          ndcg_in_batch_include_negatives=cross_device))
        model.train()
        def tokens(rows):
            ids = torch.randint(1, 32, (rows, 5))
            return dict(input_ids=ids, attention_mask=torch.ones_like(ids))
        batches = [dict(query=tokens(2), positive_document=tokens(2), negative_document=tokens(4),
                        relevance_labels=torch.tensor([[1., 0., 0.], [1., 0., 0.]])) for _ in range(2)]
        if cross_device:
            for batch in batches:
                batch['cross_batch_metadata'] = [dict(keys=[f'{b}-{m}' for m in range(3)], ids=[None]*3,
                    source='s', known_ids=[], known_positive_keys=[]) for b in range(2)]
        initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        collected = {"score_function": [], "conditional_projection": []}
        original = probe.timed_gradient_draw
        def capture(*args, **kwargs):
            row = original(*args, **kwargs)
            gradients = [p.grad.detach().flatten().double() if p.grad is not None else torch.zeros(p.numel(), dtype=torch.double)
                         for p in model.parameters() if p.requires_grad]
            collected[model.grpo.gradient_estimator].append(torch.cat(gradients))
            return row
        monkeypatch.setattr(probe, "timed_gradient_draw", capture)
        result = probe.probe_paired_gradients(model, batches, [17, 18, 19, 20], torch.device("cpu"), dtype)
        for variant, rows in collected.items():
            samples = torch.stack(rows)
            stats = result["summary"][variant]
            assert stats["noise_variance"] == pytest.approx(samples.var(0, unbiased=True).sum().item(), rel=1e-6)
            assert stats["mean_gradient_norm"] == pytest.approx(samples.mean(0).norm().item(), rel=1e-6)
            assert stats["mean_forward_backward_seconds"] > 0
        differences = torch.stack(collected["conditional_projection"]) - torch.stack(collected["score_function"])
        assert result["summary"]["paired_mean_difference_norm"] == pytest.approx(differences.mean(0).norm().item(), rel=1e-6)
        assert result["summary"]["paired_difference_noise_variance"] == pytest.approx(differences.var(0, unbiased=True).sum().item(), rel=1e-6)
        for row in result["draws"]:
            assert row["score_function"]["action_sha256"] == row["conditional_projection"]["action_sha256"]
            assert row["score_function"]["reward_table_sha256"] == row["conditional_projection"]["reward_table_sha256"]
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)
            assert parameter.grad is None
        assert model.grpo.gradient_estimator == "score_function"
    finally:
        torch.set_num_threads(previous_threads)


def test_corrected_signal_does_not_turn_monte_carlo_noise_into_signal():
    parameter = torch.nn.Parameter(torch.zeros(2))
    moments = probe.GradientMoments([("p", parameter)])
    for gradient in (torch.tensor([1., 0.]), torch.tensor([-1., 0.])):
        parameter.grad = gradient
        moments.update()
    summary = moments.summary()
    assert summary["mean_gradient_norm"] == 0
    assert summary["noise_variance"] == 2
    assert summary["signal_squared_unbiased"] == -1
    assert summary["noise_to_signal_ratio_corrected"] is None


def test_combined_reward_probe_uses_training_estimators_and_preserves_pairing():
    from test_shortlists import options, metadata

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(23)
        backbone = Qwen3Model(Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=32, attention_dropout=0., use_cache=False,
        ))
        model = GRPOModel(backbone, RLArguments(**options(
            gradient_estimator='score_function', reward_shortlist_count=1,
            reward_shortlist_size=0, reward_shortlist_hard_count=0,
            reward_shortlist_pairwise_coef=.5)))
        model.eval()
        def tokens(rows):
            ids = torch.randint(1, 32, (rows, 5))
            return dict(input_ids=ids, attention_mask=torch.ones_like(ids))
        batch = dict(query=tokens(2), positive_document=tokens(2), negative_document=tokens(4),
                     relevance_labels=torch.tensor([[3., 1., 0.], [3., 0., 0.]]),
                     candidate_mask=torch.ones(2, 3, dtype=torch.bool),
                     positive_mask=torch.tensor([[True, False, True], [True, False, False]]),
                     cross_batch_metadata=metadata()[:2])
        initial = {name: p.detach().clone() for name, p in model.named_parameters()}
        result = probe.probe_paired_shortlist_gradients(
            model, [batch], [17, 18], torch.device('cpu'), torch.float32)
        assert result['pairwise_reward_agreement_max_gap'] == 0
        for pair in result['draws']:
            cp, sf = pair['conditional_projection'], pair['score_function_rloo']
            for key in ('action_sha256', 'shortlist_sha256', 'graded_reward_sha256',
                        'pairwise_input_sha256'):
                assert cp[key] == sf[key]
            assert cp['gradient_norm'] > 0 and sf['gradient_norm'] > 0
        assert model.grpo.gradient_estimator == 'score_function'
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, initial[name], rtol=0, atol=0)
            assert p.grad is None
    finally:
        torch.set_num_threads(previous_threads)


def test_pairing_mismatch_fails_and_restores_estimator(monkeypatch):
    model = GRPOModel(Qwen3Model(Qwen3Config(vocab_size=16, hidden_size=8, intermediate_size=16,
                     num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=4)),
                     RLArguments(action_components="query;positive,negative", group_size=2))
    def different_actions(*args):
        return dict(action_sha256=model.grpo.gradient_estimator, reward_table_sha256="same",
                    forward_backward_seconds=1.)
    monkeypatch.setattr(probe, "timed_gradient_draw", different_actions)
    with pytest.raises(ValueError):
        probe.probe_paired_gradients(model, [{}], [1, 2], torch.device("cpu"), torch.float32)
    assert model.grpo.gradient_estimator == "score_function"
