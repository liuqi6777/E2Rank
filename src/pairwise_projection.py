"""Item-local CP for binary pair rewards on a fixed small shortlist.

Each positive/negative pair has its own LOO and projector. Query CP observes only
the score difference; document CP credits only the sampled endpoints. Cross-query
negatives stay detached and use the listwise reward's frozen-score attenuation.
"""
import torch
import torch.nn.functional as F

from score_precision import fp32_scores
from shortlists import shortlist_positive_mask


def _tangents(vectors, means):
    """Stable unit tangents and their lengths; both inputs may broadcast."""
    vectors, means = vectors.double(), means.double()
    radial = (vectors * means).sum(-1)
    tangent = vectors - radial[..., None] * means
    length = tangent.norm(dim=-1)
    unit = (tangent / length.clamp_min(1e-12)[..., None]).masked_fill(
        (length <= 1e-12)[..., None], 0)
    return unit.float(), radial.float(), length.float()


def _document_coefficients(mean, actions, queries, scores, advantage):
    """Per-pair endpoint coefficients [P,D], without [Gq,Gd,P,D] tensors."""
    unit, radial_q, length = _tangents(queries[:, None], mean[None])
    mean = mean.float()
    radial_d = (actions * mean[None]).sum(-1)
    projected_scores = (scores - radial_q[:, None] * radial_d[None]) / length[:, None].clamp_min(1e-12)
    projected_scores = projected_scores.masked_fill((length <= 1e-12)[:, None], 0)
    tangent = torch.einsum('ip,ipd->pd', (advantage * projected_scores).sum(1), unit)
    radial = (advantage * radial_d[None]).sum((0, 1))[:, None] * mean
    return tangent + radial


def pairwise_shortlist_loss(query_mean, document_means, query_actions, document_actions,
                            positive_mask, candidate_mask, frozen_documents, frozen_mask,
                            kappa, *, frozen_scale=1., own_scores=None, cross_scores=None,
                            chunk_size=32):
    """Return a surrogate for mean_q mean_(p,n) E[1(s_p>s_n)] and diagnostics.

    Ties earn 1/2. Every valid annotated positive competes against every valid
    own negative and selected fixed negative. A query with no pair contributes
    zero (it stays in the batch denominator). Pair membership never uses sampled
    scores. Coefficients/actions are detached; only live means receive gradients.
    """
    batch, gq, dim = query_actions.shape
    gd, count = document_actions.shape[1:3]
    if (min(gq, gd) < 2 or query_mean.shape != (batch, dim)
            or document_means.shape != (batch, count, dim)
            or document_actions.shape != (batch, gd, count, dim)
            or candidate_mask.shape != (batch, count) or candidate_mask.dtype != torch.bool
            or frozen_documents.ndim != 3 or frozen_documents.shape[0] != batch
            or frozen_documents.shape[-1] != dim or frozen_mask.shape != frozen_documents.shape[:2]
            or frozen_mask.dtype != torch.bool or isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int) or chunk_size < 1):
        raise ValueError("Pairwise CP requires joint product actions, valid candidate masks and a positive chunk size")
    positives = shortlist_positive_mask(positive_mask, candidate_mask)
    with fp32_scores(query_mean.device):
        with torch.no_grad():
            q, docs = query_actions.detach().float(), document_actions.detach().float()
            hq = F.normalize(query_mean.detach().double(), dim=-1)
            hd = F.normalize(document_means.detach().double(), dim=-1)
            fixed = frozen_documents.detach().float() * frozen_scale
            scores = (torch.einsum('bid,bjmd->bijm', q, docs) if own_scores is None
                      else own_scores.detach().float())
            cross = (torch.einsum('bid,bkd->bik', q, fixed) if cross_scores is None
                     else cross_scores.detach().float())
            if scores.shape != (batch, gq, gd, count) or cross.shape != (batch, gq, fixed.size(1)):
                raise ValueError("Pairwise scores must match the shared shortlist action grid")
            qc, dc = torch.zeros_like(query_mean, dtype=torch.float32), torch.zeros_like(document_means, dtype=torch.float32)
            means, counts, active = q.new_zeros(batch), q.new_zeros(batch), q.new_zeros(batch)
            rank_max = q.new_zeros(())
            for b in range(batch):
                pos = positives[b].nonzero(as_tuple=True)[0]
                neg = (candidate_mask[b] & ~positives[b]).nonzero(as_tuple=True)[0]
                cross_neg = frozen_mask[b].nonzero(as_tuple=True)[0] + count
                neg = torch.cat((neg, cross_neg))
                pairs = pos.numel() * neg.numel()
                counts[b] = pairs
                if not pairs:
                    continue
                a, n = pos.repeat_interleave(neg.numel()), neg.repeat(pos.numel())
                pool = torch.cat((docs[b], fixed[b][None].expand(gd, -1, -1)), dim=1)
                pool_scores = torch.cat((scores[b], cross[b, :, None].expand(-1, gd, -1)), dim=-1)
                radial_q = q[b] @ hq[b].float()
                for start in range(0, pairs, chunk_size):
                    pa, pn = a[start:start+chunk_size], n[start:start+chunk_size]
                    delta = pool[:, pa] - pool[:, pn]
                    difference = pool_scores[..., pa] - pool_scores[..., pn]
                    reward = (difference > 0).float() + .5 * (difference == 0).float()
                    aq = (reward - reward.mean(0, keepdim=True)) * (gq / (gq-1))
                    ad = (reward - reward.mean(1, keepdim=True)) * (gd / (gd-1))
                    means[b] += reward.mean((0, 1)).sum() / pairs
                    flat = reward.flatten(0, 1)
                    active[b] += (flat.max(0).values != flat.min(0).values).float().sum() / pairs

                    # For this bundle/pair, the reward sees only q dot (d_p-d_n).
                    unit, radial_delta, length = _tangents(delta, hq[b])
                    q_projection = (difference - radial_q[:, None, None] * radial_delta[None]) / length[None].clamp_min(1e-12)
                    q_projection = q_projection.masked_fill((length <= 1e-12)[None], 0)
                    qc[b] += torch.einsum('jp,jpd->d', (aq * q_projection).sum(0), unit) / pairs
                    qc[b] += hq[b].float() * (aq * radial_q[:, None, None]).sum() / pairs
                    rank_max = torch.maximum(rank_max, 1 + (length > 1e-12).float().max())

                    positive = _document_coefficients(hd[b, pa], docs[b, :, pa], q[b],
                                                       scores[b, ..., pa], ad)
                    dc[b].index_add_(0, pa, positive / pairs)
                    own = pn < count
                    negative = _document_coefficients(hd[b, pn[own]], docs[b, :, pn[own]], q[b],
                                                       scores[b, ..., pn[own]], ad[..., own])
                    dc[b].index_add_(0, pn[own], negative / pairs)
            scale = torch.as_tensor(kappa, device=q.device).detach().float() / (gq * gd)
            stats = {
                "reward/pairwise/mean": means.mean(),
                "reward/pairwise/pairs_mean": counts.mean(),
                "reward/pairwise/active_pair_fraction": active.mean(),
                "reward/pairwise/no_pairs_frac": (counts == 0).float().mean(),
                "projection/pairwise_query_span_rank_max": rank_max,
            }
        loss = -scale * ((query_mean.float() * qc).sum(-1)
                         + (document_means.float() * dc).sum((1, 2))).mean()
    return loss, stats
