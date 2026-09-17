"""Conditional score-vector projection for joint vMF product rollouts.

Sampling and rewards are unchanged. Only the detached coefficient of the live
mean direction changes. See paper/METHOD_REDESIGN.md for the conditioning proof.
"""

import torch
import torch.nn.functional as F

from score_precision import fp32_scores


def project_span(vectors, columns, *, tolerance_width=None):
    """Project [..., D] onto the numerical span of [..., D, K].

    Reduced SVD handles duplicate vectors, zero padding and K > D. A plain QR
    would add arbitrary directions for dependent columns. Rank uses the standard
    dimension-scaled FP32 tolerance; this numerical cutoff is not a tuning knob.
    """
    basis, singular, _ = torch.linalg.svd(columns, full_matrices=False)
    width = columns.shape[-1] if tolerance_width is None else tolerance_width
    tolerance = max(columns.shape[-2], width) * torch.finfo(columns.dtype).eps * singular[..., :1]
    keep = singular > tolerance
    coordinates = torch.einsum("...dk,...d->...k", basis, vectors) * keep
    return torch.einsum("...dk,...k->...d", basis, coordinates), keep.sum(-1)


def project_with_fixed_documents(vectors, moving_columns, fixed_documents, fixed_mask):
    """Same numerical projector without replicating a large fixed pool over Gd.

    Factor F = U S V^T once per query; replacing F with U S preserves A A^T,
    its singular values and its left singular vectors. Keep the ORIGINAL width
    in the rank tolerance. A conservative singular-value bound allows an exact
    identity projection when F alone guarantees full numerical rank.
    """
    batch, group, dim = vectors.shape
    projected = torch.empty_like(vectors)
    ranks = torch.empty((batch, group), dtype=torch.long, device=vectors.device)
    width = moving_columns.size(-1) + fixed_documents.size(1)
    for b in range(batch):
        fixed = fixed_documents[b].detach().float().masked_fill(~fixed_mask[b, :, None], 0).T
        basis, singular, _ = torch.linalg.svd(fixed, full_matrices=False)
        moving = moving_columns[b]
        upper = (singular[0].square() + moving.square().sum((-2, -1)).max()).sqrt()
        threshold = max(dim, width) * torch.finfo(vectors.dtype).eps * upper
        if singular.numel() == dim and singular[-1] > threshold:
            projected[b] = vectors[b]
            ranks[b] = dim
            continue
        compressed = basis * singular[None, :]
        for draw in range(group):
            columns = torch.cat((moving[draw], compressed), dim=-1)
            projected[b, draw], ranks[b, draw] = project_span(
                vectors[b, draw], columns, tolerance_width=width,
            )
    return projected, ranks


def conditional_projection_loss(
    query_mean, document_means, query_actions, document_actions, rewards,
    kappa, candidate_mask, *, frozen_documents=None, frozen_mask=None, stream_frozen=False,
):
    """Return a surrogate and ranks for [B, Gq, Gd] cellwise LOO rewards.

    query_mean/document_means are live, on-policy unit directions. Actions and
    every quantity used to construct projectors/advantages are detached. Cross
    candidates must include ALL fixed directions used by any reward term.
    No [B, Gq, Gd, M, D] tensor is materialized.
    """
    batch, gq, dim = query_actions.shape
    gd, count = document_actions.shape[1:3]
    if (rewards.shape != (batch, gq, gd) or min(gq, gd) < 2
            or document_actions.shape != (batch, gd, count, dim)
            or query_mean.shape != (batch, dim)
            or document_means.shape != (batch, count, dim)
            or candidate_mask.shape != (batch, count)):
        raise ValueError("Conditional projection requires joint query/document product rollouts")
    with fp32_scores(query_mean.device):
        with torch.no_grad():
            q, docs = query_actions.detach().float(), document_actions.detach().float()
            hq = F.normalize(query_mean.detach().float(), dim=-1)
            hd = F.normalize(document_means.detach().float(), dim=-1)
            reward = rewards.detach().float()
            aq = (reward - reward.mean(1, keepdim=True)) * (gq / (gq - 1))
            ad = (reward - reward.mean(2, keepdim=True)) * (gd / (gd - 1))
            weighted_q = torch.einsum("bij,bid->bjd", aq, q)
            columns = [hq[:, None, None].expand(-1, gd, -1, -1),
                       docs.masked_fill(~candidate_mask[:, None, :, None], 0)]
            if frozen_documents is not None:
                if frozen_mask is None or frozen_mask.shape != frozen_documents.shape[:2]:
                    raise ValueError("Frozen projection directions require a matching mask")
                if not stream_frozen:
                    fixed = frozen_documents.detach().float().masked_fill(~frozen_mask[..., None], 0)
                    columns.append(fixed[:, None].expand(-1, gd, -1, -1))
            moving = torch.cat(columns, dim=2).transpose(-2, -1)
            if stream_frozen and frozen_documents is not None:
                projected_q, ranks = project_with_fixed_documents(weighted_q, moving, frozen_documents, frozen_mask)
            else:
                projected_q, ranks = project_span(weighted_q, moving)
            q_coefficient = projected_q.sum(1)

            # For document m at query draw i, span(h_m, q_i) has at most two
            # dimensions. Form the tangent in FP64 to avoid cancellation for
            # nearly parallel directions; the contractions remain FP32.
            hd64 = F.normalize(hd.double(), dim=-1)
            tangent = q.double()[:, :, None] - (
                q.double()[:, :, None] * hd64[:, None]
            ).sum(-1, keepdim=True) * hd64[:, None]
            tangent_norm = tangent.norm(dim=-1, keepdim=True)
            tangent = (tangent / tangent_norm.clamp_min(1e-12)).float()
            tangent = tangent.masked_fill(tangent_norm <= 1e-12, 0)
            # First sum over document draws, then project for each query draw.
            weighted_docs = torch.einsum("bij,bjmd->bimd", ad, docs)
            tangent_coefficient = (weighted_docs * tangent).sum(-1, keepdim=True)
            projected_d = (tangent_coefficient * tangent).sum(1)
            # Retain the radial part too: this is the full conditional score
            # vector. Normalization of live means removes it in backpropagation.
            projected_d += (weighted_docs * hd[:, None]).sum((1, 3))[..., None] * hd
            projected_d = projected_d.masked_fill(~candidate_mask[..., None], 0)
            scale = torch.as_tensor(kappa, device=q.device).detach().float() / (gq * gd)
        surrogate = -scale * (
            (query_mean.float() * q_coefficient).sum(-1)
            + (document_means.float() * projected_d).sum((1, 2))
        ).mean()
    return surrogate, ranks
