"""Cross-query negative pools for RAG contrastive training.

G1 found that widening the negative pool is the single largest contrastive
lever: moving from device-local representatives to a cross-device pool of every
query's full candidate list was worth roughly a point of nDCG. RAG can take
that lever more cheaply than G1 did. Because the corpus vectors are frozen, a
pooled candidate carries no gradient, so the pool is assembled with a plain
``all_gather`` rather than the autograd-aware gather G1 needs.

The pool must not smuggle in false negatives. Another query's candidate can
easily be one of *this* query's positives or answer-bearing passages, and
demoting it is exactly the failure that made the original G3 runs score below
their own initialization. Every pooled ordinal is therefore checked against the
borrowing query's own judged set and masked out on a hit.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor


def _sample_per_query(
    candidate_ordinals: Tensor,
    candidate_mask: Tensor,
    pool_size: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Uniformly sample ``pool_size`` valid candidates from each query's list."""
    batch, depth = candidate_ordinals.shape
    if pool_size >= depth:
        return candidate_ordinals
    # Gumbel-style trick: random keys on valid entries, -inf elsewhere, then
    # topk. Keeps the whole sample on-device and vectorized across the batch.
    keys = torch.rand(
        (batch, depth), device=candidate_ordinals.device, generator=generator
    ).masked_fill(~candidate_mask.bool(), -1.0)
    chosen = keys.topk(pool_size, dim=-1).indices
    return candidate_ordinals.gather(1, chosen)


def build_cross_query_pool(
    *,
    candidate_ordinals: Tensor,
    candidate_mask: Tensor,
    judged_mask: Tensor,
    pool_size: int,
    include_negatives: bool,
    cross_device: bool,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Assemble extra negatives borrowed from other queries.

    Args:
        candidate_ordinals: ``[batch, depth]`` corpus ordinals for this rank.
        candidate_mask: ``[batch, depth]`` validity of those ordinals.
        judged_mask: ``[batch, depth]`` candidates this rank's queries consider
            positive or answer-bearing. Used to suppress false negatives.
        pool_size: candidates contributed per query when ``include_negatives``.
        include_negatives: contribute each query's sampled candidates rather
            than only its top-ranked representative.
        cross_device: gather the pool across all data-parallel ranks.

    Returns:
        ``(pool_ordinals, pool_mask)`` where ``pool_ordinals`` is ``[width]``
        and ``pool_mask`` is ``[batch, width]``: entries this rank's queries may
        legitimately treat as negatives.
    """
    batch, depth = candidate_ordinals.shape
    if include_negatives:
        contributed = _sample_per_query(candidate_ordinals, candidate_mask, pool_size, generator)
    else:
        contributed = candidate_ordinals[:, :1]

    local_width = contributed.size(1)
    # Owner bookkeeping so a query never borrows from its own row.
    if cross_device and dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        gathered = [torch.empty_like(contributed) for _ in range(world_size)]
        dist.all_gather(gathered, contributed.contiguous())
        # all_gather returns detached copies; splice this rank's own tensor back
        # so the pool is identical on every rank without an extra sync.
        gathered[rank] = contributed
        pool = torch.cat(gathered, dim=0)
        owner_offset = rank * batch
    else:
        pool = contributed
        owner_offset = 0

    pool_ordinals = pool.reshape(-1)
    width = pool_ordinals.numel()
    owners = torch.arange(pool.size(0), device=pool.device).repeat_interleave(local_width)
    local_rows = torch.arange(batch, device=pool.device) + owner_offset
    mask = owners.unsqueeze(0) != local_rows.unsqueeze(1)

    # False-negative guard: drop pooled ordinals that this query itself judges
    # positive or answer-bearing. Only ~57 of 1000 candidates are judged, so
    # compact them first -- comparing against the full depth would allocate a
    # [batch, width, depth] boolean.
    judged = judged_mask.bool() & candidate_mask.bool()
    judged_depth = int(judged.sum(dim=-1).max().item())
    if judged_depth:
        order = judged.int().argsort(dim=-1, descending=True, stable=True)[:, :judged_depth]
        judged_ordinals = candidate_ordinals.gather(1, order).masked_fill(
            ~judged.gather(1, order), -1
        )
        collision = (
            pool_ordinals.view(1, width, 1) == judged_ordinals.view(batch, 1, judged_depth)
        ).any(dim=-1)
        mask &= ~collision
    mask &= pool_ordinals.unsqueeze(0) >= 0
    return pool_ordinals, mask
