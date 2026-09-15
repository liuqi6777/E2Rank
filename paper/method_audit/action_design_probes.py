"""Isolated estimator experiments; no training, checkpoint writes or downloads.

Run: .venv/bin/python paper/method_audit/action_design_probes.py
All comparisons are paired on the same draws. Synthetic examples do not predict
BRIGHT performance or establish variance reduction for a Transformer encoder.
"""

import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from grpo import GRPO, sample_vmf
from policy_math import group_advantages, kappa_for_alignment, mean_alignment
from rewards import compute_reward_from_scores


def tangent(vector, mean):
    return vector - (vector * mean).sum(-1, keepdim=True) * mean


def geometry(dim, own=8, frozen_count=3):
    torch.manual_seed(902 + dim)
    query = F.normalize(torch.randn(dim), dim=0)
    tangents = F.normalize(tangent(torch.randn(own + frozen_count, dim), query), dim=-1)
    # A tight slate with several positives and hard negatives; values are invented.
    cosines = torch.linspace(.63, .57, own + frozen_count)
    docs = cosines[:, None] * query + (1-cosines.square()).sqrt()[:, None] * tangents
    labels = torch.zeros(own + frozen_count)
    labels[[1, 4, 7]] = 1
    return torch.cat([query[None], docs[:own]]), docs[own:], labels


def joint_gradients(means, frozen, query, docs, reward, kappa):
    """[repeat, G, G] product LOO, before/after per-cell conditioning.

    Gradients are with respect to raw pre-normalization vectors evaluated at
    unit means. Frozen vectors are constants in this conditional objective.
    Projectors are numeric constants; no derivative through their construction.
    """
    repeats, group, dim = query.shape
    own = docs.shape[2]
    aq = group / (group-1) * (reward - reward.mean(1, keepdim=True))
    ad = group / (group-1) * (reward - reward.mean(2, keepdim=True))
    scale = kappa / group**2
    weighted_query = torch.einsum("rij,rid->rjd", aq, query)
    raw_q = scale * weighted_query.sum(1)
    raw_d = scale * torch.einsum("rij,rjmd->rmd", ad, docs)

    # Per document bundle, condition on all query inner products that affect R.
    columns = torch.cat([
        means[0].expand(repeats, group, 1, dim), docs,
        frozen.expand(repeats, group, -1, -1),
    ], dim=2).transpose(-2, -1)
    basis, triangular = torch.linalg.qr(columns, mode="reduced")
    if triangular.diagonal(dim1=-2, dim2=-1).abs().min() < 1e-5:
        raise RuntimeError("Toy basis lost rank; production needs rank-revealing SVD/QR.")
    projected_q = scale * (basis @ (basis.transpose(-2, -1) @ weighted_query[..., None])).squeeze(-1).sum(1)

    # Each document action is observed only along this sampled query direction.
    query_dot_mean = torch.einsum("rid,md->rim", query, means[1:])
    u = query[:, :, None, :] - query_dot_mean[..., None] * means[None, None, 1:]
    scalar = torch.einsum("rimd,rjmd->rijm", u, docs)
    weights = torch.einsum("rij,rijm->rim", ad, scalar) / u.square().sum(-1).clamp_min(1e-12)
    projected_d = scale * torch.einsum("rim,rimd->rmd", weights, u)
    raw = tangent(torch.cat([raw_q[:, None], raw_d], dim=1), means)
    projected = tangent(torch.cat([projected_q[:, None], projected_d], dim=1), means)
    return raw, projected


def shared_inner(left, right, gram):
    """Frobenius product after the shared encoder Jacobian, without d*d matrices.

    Encoder h_i(W)=normalize(W*x_i), x_i=means[i], W=I.
    dJ/dW = sum_i g_i x_i^T, with g_i already tangent-projected.
    This includes cross-covariance of query and document contributions.
    """
    return torch.einsum("...id,...jd,ij->...", left, right, gram)


def summary(values, gram, expected=None):
    average = values.mean(0)
    variance = float(shared_inner(values-average, values-average, gram).sum() / (len(values)-1))
    result = dict(shared_encoder_noise_variance=variance,
                  embedding_noise_variance=float((values-average).square().sum() / (len(values)-1)),
                  shared_encoder_mean_norm=float(shared_inner(average, average, gram).clamp_min(0).sqrt()))
    if expected is not None:
        expected = expected.to(values.dtype)
        norm = shared_inner(expected, expected, gram).sqrt()
        error = shared_inner(average-expected, average-expected, gram).clamp_min(0).sqrt()
        result.update(relative_error_to_analytic=float(error/norm),
                      expected_mc_rms_relative_error=math.sqrt(variance/len(values))/float(norm),
                      analytic_gradient_norm=float(norm))
    return result


def joint_projection_probe(dim=1024, repeats=256, group=32, linear=False):
    means, frozen, labels = geometry(dim)
    own = len(means)-1
    gram = means @ means.T
    kappa = kappa_for_alignment(dim, .9)
    alignment = mean_alignment(dim, kappa)
    coefficients = torch.tensor([1., -.4, .3, -.7, 1., -.2, -.6, .8])
    expected = None
    if linear:
        expected = alignment**2 * torch.cat([
            (coefficients[:, None] * means[1:]).sum(0, keepdim=True),
            coefficients[:, None] * means[0],
        ])
        expected = tangent(expected, means)
    collected = [[], []]
    rewards = []
    torch.manual_seed(1037 + dim + int(linear))
    start = time.monotonic()
    for offset in range(0, repeats, 8):
        batch = min(8, repeats-offset)
        draws = sample_vmf(means, kappa, batch*group).reshape(own+1, batch, group, dim)
        query = draws[0]
        docs = draws[1:].permute(1, 2, 0, 3)
        own_scores = torch.einsum("rid,rjmd->rijm", query, docs)
        if linear:
            reward = (own_scores * coefficients).sum(-1)
        else:
            fixed_scores = alignment * torch.einsum("rid,fd->rif", query, frozen)
            scores = torch.cat([own_scores, fixed_scores[:, :, None].expand(-1, -1, group, -1)], dim=-1)
            reward = compute_reward_from_scores(scores, labels.expand(batch, -1), reward_type="ndcg", k=5)
        raw, projected = joint_gradients(means, frozen, query, docs, reward, kappa)
        for destination, value in zip(collected, (raw, projected)):
            destination.append(value.double())
        rewards.append(reward.mean().item())
    raw, projected = [torch.cat(parts) for parts in collected]
    gram = gram.double()
    raw_stats = summary(raw, gram, expected)
    projected_stats = summary(projected, gram, expected)
    # A fixed direction in parameter space, defined by an analytic bilinear reward,
    # is independent of all the Monte Carlo samples used for the rank reward too.
    direction = tangent(torch.cat([(coefficients[:, None]*means[1:]).sum(0, keepdim=True),
                                    coefficients[:, None]*means[0]]), means).double()
    direction /= shared_inner(direction, direction, gram).sqrt()
    paired = shared_inner(projected-raw, direction, gram)
    se = paired.std(unbiased=True)/math.sqrt(repeats)
    return dict(dimension=dim, own_documents=own, frozen_documents=len(frozen),
                group=group, repeats=repeats, reward="bilinear" if linear else "binary_ndcg_at_5",
                mean_sampled_reward=sum(rewards)/len(rewards), raw=raw_stats, projected=projected_stats,
                shared_encoder_variance_ratio=projected_stats["shared_encoder_noise_variance"]/raw_stats["shared_encoder_noise_variance"],
                paired_projected_minus_raw_on_fixed_direction=dict(mean=float(paired.mean()), standard_error=float(se)),
                elapsed_seconds=time.monotonic()-start,
                limitation="Synthetic shared normalized linear encoder at W=I; not a Transformer or a training experiment.")


def raw_formula_check():
    means, frozen, _ = geometry(16)
    live = means.clone().requires_grad_()
    group, kappa = 8, kappa_for_alignment(16, .9)
    draws = sample_vmf(means, kappa, group)
    query, docs = draws[0][None], draws[1:].permute(1, 0, 2)[None]
    reward = torch.randn(1, group, group)
    raw, _ = joint_gradients(means, frozen, query, docs, reward, kappa)
    q_adv, _ = group_advantages(reward.mean(2), baseline="leave_one_out", normalization="none")
    d_adv, _ = group_advantages(reward.mean(1), baseline="leave_one_out", normalization="none")
    directions = F.normalize(live, dim=-1)
    log_prob = GRPO._vmf_log_prob(directions, draws, torch.tensor(kappa))
    estimate = torch.autograd.grad((q_adv*log_prob[0]).mean()+(d_adv*log_prob[1:].sum(0)).mean(), live)[0]
    error = float((raw[0]-estimate).abs().max())
    assert error < 2e-5, error
    return dict(max_absolute_error_against_actual_log_prob_autograd=error)


def normal_pdf(z):
    return torch.exp(-.5*z.square()) / math.sqrt(2*math.pi)


def integrated_ndcg_gradient(scores, noise, gains, tau, k):
    """Exact conditional derivative: integrate the current coordinate out.

    For each m, only noisy scores of OTHER documents are used. A jump occurs
    whenever m passes one opponent. This is score smoothing / partial integration,
    not a new policy-gradient law or an equivalent vMF objective.
    """
    count = len(scores)
    discounts = 1/torch.log2(torch.arange(2, count+2, dtype=scores.dtype))
    discounts[k:] = 0
    idcg = (gains.sort(descending=True).values * discounts).sum()
    jumps_discount = discounts[:-1]-discounts[1:]
    noisy = scores + tau*noise
    gradients = []
    for m in range(count):
        others = torch.arange(count) != m
        boundaries, order = noisy[..., others].sort(descending=True)
        opponent_gains = gains[others][order]
        jumps = (gains[m]-opponent_gains)*jumps_discount/idcg
        gradients.append((jumps*normal_pdf((boundaries-scores[m])/tau)/tau).sum(-1))
    return torch.stack(gradients, dim=-1)


def score_smoothing_probe():
    torch.manual_seed(774)
    group, repeats, tau = 32, 8192, .04
    scores = torch.tensor([.60, .61], dtype=torch.float64)
    gains = torch.tensor([1., 0.], dtype=torch.float64)
    noise = torch.randn(repeats, group, 2, dtype=torch.float64)
    noisy = scores+tau*noise
    lose_reward = 1/math.log2(3)
    reward = torch.where(noisy[..., 0] > noisy[..., 1], 1., lose_reward)
    advantages, _ = group_advantages(reward, baseline="leave_one_out", normalization="none")
    raw = (advantages[..., None]*noise/tau).mean(1)
    integrated = integrated_ndcg_gradient(scores, noise, gains, tau, k=2).mean(1)
    exact_component = (1-lose_reward)*normal_pdf((scores[0]-scores[1])/(math.sqrt(2)*tau))/(math.sqrt(2)*tau)
    expected = torch.stack([exact_component, -exact_component])
    output = dict(group=group, repeats=repeats, tau=tau, scores=scores.tolist(),
                  analytic_gradient=expected.tolist())
    for name, values in (("score_function_loo", raw), ("conditional_integration", integrated)):
        avg = values.mean(0)
        se = values.std(0, unbiased=True)/math.sqrt(repeats)
        assert ((avg-expected).abs() < 5*se).all()
        output[name] = dict(mean_gradient=avg.tolist(), standard_errors=se.tolist(),
                            trace_noise_variance=float(values.var(0, unbiased=True).sum()),
                            relative_error=float((avg-expected).norm()/expected.norm()))
    output["variance_ratio"] = output["conditional_integration"]["trace_noise_variance"]/output["score_function_loo"]["trace_noise_variance"]

    # Multi-document validation: evaluate the exact conditional reward using the
    # repository's actual nDCG at every interval, then differentiate interval
    # probabilities. This independently checks jump signs, gain scale and cutoff.
    test_scores = torch.tensor([.62, .59, .61, .60, .58], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([2., 0., 3., 1., 0.], dtype=torch.float64)
    gains = 2**labels-1
    noises = torch.randn(11, 5, dtype=torch.float64)
    proposed = integrated_ndcg_gradient(test_scores.detach(), noises, gains, tau, k=3)
    exact = torch.zeros_like(proposed)
    normal = torch.distributions.Normal(0., 1.)
    for r in range(11):
        fixed = (test_scores.detach()+tau*noises[r])
        for m in range(5):
            bounds = fixed[torch.arange(5) != m].sort().values
            representatives = torch.cat([bounds[:1]-1, (bounds[:-1]+bounds[1:])/2, bounds[-1:]+1])
            slate = fixed.expand(5, -1).clone()
            slate[:, m] = representatives
            interval_reward = compute_reward_from_scores(slate, labels.expand(5, -1), reward_type="ndcg", relevance_scheme="graded", k=3)
            cdf = normal.cdf((bounds-test_scores[m])/tau)
            mass = torch.cat([cdf[:1], cdf[1:]-cdf[:-1], 1-cdf[-1:]])
            exact[r, m] = torch.autograd.grad((interval_reward*mass).sum(), test_scores)[0][m]
    max_error = float((proposed-exact).abs().max())
    assert max_error < 2e-5, max_error
    output["graded_five_document_cutoff3_check"] = dict(conditional_draws=11, max_absolute_error_vs_interval_integration=max_error)
    output["limitation"] = "Two-document variance result; partial-integration literature already studies this idea. Not retrieval evidence."
    return output


def main():
    torch.set_num_threads(4)
    results = {"raw_formula_check": raw_formula_check()}
    print(json.dumps(results), flush=True)
    results["joint_projection"] = []
    for dim, repeats, linear in ((32, 4096, True), (128, 256, False), (1024, 256, False)):
        result = joint_projection_probe(dim=dim, repeats=repeats, linear=linear)
        results["joint_projection"].append(result)
        print(json.dumps(result), flush=True)
    results["score_smoothing"] = score_smoothing_probe()
    destination = Path(__file__).with_name("action_design_evidence.json")
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(results["score_smoothing"]), flush=True)
    print(f"Wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
