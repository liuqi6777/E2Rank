from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F


SUPPORTED_REWARD_TYPES = {
    "ndcg",
    "ndcg_in_batch",
    "rbo",
    "top_weighted_pairwise",
    "contrastive",
    "infonce",
    "mrr",
    "mrr_in_batch",
}

# Rewards split into two families with incomparable scales: ranking metrics are bounded in
# [0, 1] while the contrastive margins live on a temperature-scaled log-score axis whose
# within-group spread is typically an order of magnitude larger. Mixing families under a raw
# weighted sum therefore does NOT mix them in the ratio of their weights (see
# SUPPORTED_REWARD_COMBINE_MODES), which is what these sets are used to warn about.
BOUNDED_REWARD_TYPES = frozenset(
    {"ndcg", "ndcg_in_batch", "mrr", "mrr_in_batch", "rbo", "top_weighted_pairwise"}
)
UNBOUNDED_REWARD_TYPES = frozenset({"contrastive", "infonce"})

# 'sum'            -- R = sum_i w_i R_i, then one advantage over the combined reward. Keeps the
#                     terms' relative effect sizes, so w_i is a weight on the *raw* reward.
# 'normalized_sum' -- A = sum_i w_i A_i with each A_i group-standardized on its own. The loss is
#                     linear in the advantage, so this is a scale-free combination in which w_i
#                     really is the mixing ratio. Preferred whenever families are mixed.
SUPPORTED_REWARD_COMBINE_MODES = ("sum", "normalized_sum")

_TERM_FIELD_ALIASES = {
    "k": "k",
    "ndcg_k": "k",
    "reward_ndcg_k": "k",
    "cutoff": "k",
    "temperature": "temperature",
    "contrastive_temperature": "temperature",
    "tau": "temperature",
    "p": "rbo_p",
    "rbo_p": "rbo_p",
    "reward_rbo_p": "rbo_p",
    "weight": "weight",
    "w": "weight",
    "name": "name",
    "type": "type",
    "reward_type": "type",
    "include_negatives": "ndcg_in_batch_include_negatives",
    "ndcg_in_batch_include_negatives": "ndcg_in_batch_include_negatives",
    "in_batch_negatives": "contrastive_use_in_batch_negatives",
    "contrastive_use_in_batch_negatives": "contrastive_use_in_batch_negatives",
}

_TRUE_STRINGS = {"true", "1", "yes", "y", "on"}
_FALSE_STRINGS = {"false", "0", "no", "n", "off"}


@dataclass(frozen=True)
class RewardTerm:
    """One additive term of the reward.

    ``k``/``temperature``/``rbo_p``/the two in-batch flags may be left unset (``None``), in
    which case :func:`normalize_reward_terms` fills them from the run-level defaults. That
    keeps a single-term config byte-identical to the pre-combination behaviour.
    """

    type: str
    weight: float = 1.0
    k: int | None = None
    temperature: float | None = None
    ndcg_in_batch_include_negatives: bool | None = None
    contrastive_use_in_batch_negatives: bool | None = None
    name: str = ""
    rbo_p: float | None = None


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
    raise ValueError(f"Expected a boolean-like value, got {value!r}")


def _parse_term_string(spec: str) -> dict:
    """Parse the compact CLI form ``type[:weight][,key=value]*``."""
    head, _, tail = spec.partition(",")
    reward_type, _, weight = head.partition(":")
    fields: dict = {"type": reward_type.strip()}
    if weight.strip():
        fields["weight"] = weight.strip()

    for raw_field in tail.split(","):
        raw_field = raw_field.strip()
        if not raw_field:
            continue
        key, separator, value = raw_field.partition("=")
        if not separator:
            raise ValueError(
                f"Malformed reward term field {raw_field!r} in {spec!r}; expected 'key=value'"
            )
        fields[key.strip()] = value.strip()
    return fields


def _build_term(fields: Mapping) -> RewardTerm:
    normalized_fields: dict = {}
    for raw_key, value in fields.items():
        key = _TERM_FIELD_ALIASES.get(str(raw_key).strip().lower())
        if key is None:
            raise ValueError(
                f"Unsupported reward term field: {raw_key!r}. "
                f"Supported fields: {sorted(set(_TERM_FIELD_ALIASES))}"
            )
        if value is None:
            continue
        normalized_fields[key] = value

    if "type" not in normalized_fields:
        raise ValueError(f"Reward term is missing the required 'type' field: {dict(fields)!r}")

    reward_type = str(normalized_fields["type"]).strip().lower()
    if reward_type not in SUPPORTED_REWARD_TYPES:
        raise ValueError(
            f"Unsupported reward type: {reward_type}. Supported types: {sorted(SUPPORTED_REWARD_TYPES)}"
        )

    weight = float(normalized_fields.get("weight", 1.0))
    if not (weight == weight) or weight in {float("inf"), float("-inf")}:
        raise ValueError(f"Reward term weight must be finite, got {weight}")

    k = normalized_fields.get("k")
    if k is not None:
        k = None if str(k).strip().lower() in {"none", "null", ""} else int(k)

    temperature = normalized_fields.get("temperature")
    if temperature is not None:
        temperature = float(temperature)
        if temperature <= 0:
            raise ValueError(f"Reward term temperature must be positive, got {temperature}")

    rbo_p = normalized_fields.get("rbo_p")
    if rbo_p is not None:
        rbo_p = float(rbo_p)
        if not 0.0 <= rbo_p < 1.0:
            raise ValueError(f"RBO persistence p must lie in [0, 1), got {rbo_p}")

    include_negatives = normalized_fields.get("ndcg_in_batch_include_negatives")
    use_in_batch_negatives = normalized_fields.get("contrastive_use_in_batch_negatives")

    return RewardTerm(
        type=reward_type,
        weight=weight,
        k=k,
        temperature=temperature,
        rbo_p=rbo_p,
        ndcg_in_batch_include_negatives=(
            None if include_negatives is None else _coerce_bool(include_negatives)
        ),
        contrastive_use_in_batch_negatives=(
            None if use_in_batch_negatives is None else _coerce_bool(use_in_batch_negatives)
        ),
        name=str(normalized_fields.get("name", "")).strip(),
    )


def normalize_reward_terms(
    reward_terms,
    default_k: int | None = 10,
    default_temperature: float = 0.03,
    default_ndcg_in_batch_include_negatives: bool = False,
    default_contrastive_use_in_batch_negatives: bool = False,
    default_rbo_p: float = 0.9,
) -> tuple[RewardTerm, ...]:
    """Normalize a reward-term spec into fully resolved :class:`RewardTerm` objects.

    Accepts, in order of increasing verbosity:

    * a single type string -- ``"ndcg"``
    * the compact CLI form -- ``"ndcg:1.0,k=16;contrastive:0.5,in_batch_negatives=true"``
    * a YAML list of mappings -- ``[{type: ndcg, weight: 1.0, k: 16}, ...]``

    Unset per-term fields inherit the run-level defaults, so the single-term spec reproduces
    the legacy ``reward_type`` behaviour exactly.
    """
    if reward_terms is None:
        raise ValueError("reward_terms must not be None")
    if not 0.0 <= default_rbo_p < 1.0:
        raise ValueError(f"Default RBO persistence p must lie in [0, 1), got {default_rbo_p}")

    if isinstance(reward_terms, RewardTerm):
        raw_terms: list = [reward_terms]
    elif isinstance(reward_terms, Mapping):
        raw_terms = [reward_terms]
    elif isinstance(reward_terms, str):
        raw_terms = [chunk.strip() for chunk in reward_terms.split(";") if chunk.strip()]
    elif isinstance(reward_terms, Sequence):
        raw_terms = list(reward_terms)
    else:
        raise ValueError(
            "reward_terms must be a type string, a 'type:weight,key=value;...' string, "
            "or a list of mappings"
        )

    if not raw_terms:
        raise ValueError("reward_terms must contain at least one term")

    terms: list[RewardTerm] = []
    for raw_term in raw_terms:
        if isinstance(raw_term, RewardTerm):
            term = raw_term
        elif isinstance(raw_term, Mapping):
            term = _build_term(raw_term)
        elif isinstance(raw_term, str):
            term = _build_term(_parse_term_string(raw_term))
        else:
            raise ValueError(f"Unsupported reward term entry: {raw_term!r}")

        term = replace(
            term,
            k=default_k if term.k is None else term.k,
            temperature=default_temperature if term.temperature is None else term.temperature,
            rbo_p=default_rbo_p if term.rbo_p is None else term.rbo_p,
            ndcg_in_batch_include_negatives=(
                default_ndcg_in_batch_include_negatives
                if term.ndcg_in_batch_include_negatives is None
                else term.ndcg_in_batch_include_negatives
            ),
            contrastive_use_in_batch_negatives=(
                default_contrastive_use_in_batch_negatives
                if term.contrastive_use_in_batch_negatives is None
                else term.contrastive_use_in_batch_negatives
            ),
        )
        terms.append(term)

    if all(term.weight == 0.0 for term in terms):
        raise ValueError("At least one reward term must carry a non-zero weight")

    # Names index the per-term metrics, so they have to be unique and stable across ranks.
    named_terms: list[RewardTerm] = []
    used_names: dict[str, int] = {}
    for term in terms:
        base_name = term.name or term.type
        occurrence = used_names.get(base_name, 0)
        used_names[base_name] = occurrence + 1
        if occurrence and term.name:
            raise ValueError(f"Duplicate reward term name: {term.name!r}")
        name = base_name if not occurrence else f"{base_name}_{occurrence + 1}"
        named_terms.append(replace(term, name=name))
    return tuple(named_terms)


def normalize_reward_combine_mode(mode) -> str:
    if not isinstance(mode, str) or mode.strip().lower() not in SUPPORTED_REWARD_COMBINE_MODES:
        raise ValueError(
            f"Unsupported reward_combine mode: {mode!r}. "
            f"Expected one of {SUPPORTED_REWARD_COMBINE_MODES}."
        )
    return mode.strip().lower()


def reward_terms_mix_scales(reward_terms: Sequence[RewardTerm]) -> bool:
    """True when bounded ranking rewards are mixed with unbounded contrastive margins."""
    types = {term.type for term in reward_terms if term.weight != 0.0}
    return bool(types & BOUNDED_REWARD_TYPES) and bool(types & UNBOUNDED_REWARD_TYPES)


def ranking_reward_pool_size(term: RewardTerm, slate_size: int, batch_size: int) -> int | None:
    """How many candidates a rank-based term ranks over. None for the score-based families."""
    if term.type in {"mrr", "ndcg", "rbo", "top_weighted_pairwise"}:
        return slate_size
    if term.type in {"ndcg_in_batch", "mrr_in_batch"}:
        if term.ndcg_in_batch_include_negatives:
            return batch_size * slate_size
        return slate_size + max(batch_size - 1, 0)
    return None


def warn_on_inert_cutoffs(
    reward_terms: Sequence[RewardTerm],
    slate_size: int,
    batch_size: int,
) -> list[str]:
    """Report rank-based terms whose cutoff can never bind, and how coarse each one is.

    Three cases, distinguished because only one of them is a mistake:

    * ``k > pool`` is a config error. The cutoff can never apply, so ``@k`` names a truncation
      that does not exist, and every caption quoting it is wrong. Easy to reintroduce by
      editing ``slate_size`` alone, since the cutoff lives in a different config slot.
    * ``k == pool`` is the full metric by construction. That is the intended setting for the
      own-slate rows, so it is stated rather than flagged.
    * whatever ``k`` is, a small pool caps the reward's *resolution*: a rank-based reward takes
      at most one value per candidate, so a group of G rollouts over an n-candidate pool cannot
      resolve more than min(G, n) levels. The rest tie, and tied groups contribute no gradient.

    Returns the lines rather than logging them, so callers decide where they go and tests can
    assert on them.
    """
    warnings: list[str] = []
    for term in reward_terms:
        pool = ranking_reward_pool_size(term, slate_size, batch_size)
        if pool is None:
            continue
        if term.k is not None and term.k > pool:
            warnings.append(
                f"reward term '{term.name}': cutoff k={term.k} EXCEEDS its {pool}-candidate "
                f"pool, so no truncation ever happens and '@{term.k}' is a mislabel. Set "
                f"k <= {pool}."
            )
        cutoff = "no cutoff, i.e. the full metric" if term.k is None or term.k == pool \
            else f"cutoff @{min(term.k, pool)}"
        if term.type in {"rbo", "top_weighted_pairwise"}:
            warnings.append(
                f"reward term '{term.name}': pool={pool} ({cutoff}); this permutation-native "
                "reward can realize more distinct values than the candidate count."
            )
        else:
            warnings.append(
                f"reward term '{term.name}': pool={pool} ({cutoff}), so the reward has at most "
                f"{pool} distinct values; watch reward/{term.name}/n_distinct against the group size."
            )
    return warnings


def reward_terms_need_in_batch_positives(reward_terms: Sequence[RewardTerm]) -> bool:
    return any(
        (
            term.type in {"ndcg_in_batch", "mrr_in_batch"}
            and not term.ndcg_in_batch_include_negatives
        )
        or (term.type in {"contrastive", "infonce"} and term.contrastive_use_in_batch_negatives)
        for term in reward_terms
    )


def reward_terms_need_in_batch_candidates(reward_terms: Sequence[RewardTerm]) -> bool:
    return any(
        term.type in {"ndcg_in_batch", "mrr_in_batch"} and term.ndcg_in_batch_include_negatives
        for term in reward_terms
    )


def compute_reward_terms(
    reward_terms: Sequence[RewardTerm],
    scores: torch.Tensor,
    relevance_labels: torch.Tensor,
    relevance_scheme: str | None = None,
    in_batch_positive_scores: torch.Tensor | None = None,
    in_batch_candidate_scores: torch.Tensor | None = None,
    rank_labels: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate every reward term against one shared score table.

    The caller builds the (expensive) in-batch score tables once; each term picks the ones it
    needs via its own flags, so terms never pay for tables they ignore.
    """
    return {
        term.name: compute_reward_from_scores(
            scores=scores,
            candidate_mask=candidate_mask,
            relevance_labels=relevance_labels,
            rank_labels=rank_labels,
            reward_type=term.type,
            k=term.k,
            ndcg_in_batch_include_negatives=bool(term.ndcg_in_batch_include_negatives),
            contrastive_use_in_batch_negatives=bool(term.contrastive_use_in_batch_negatives),
            contrastive_temperature=float(term.temperature),
            rbo_p=float(term.rbo_p),
            relevance_scheme=relevance_scheme,
            in_batch_positive_scores=in_batch_positive_scores,
            in_batch_candidate_scores=in_batch_candidate_scores,
        )
        for term in reward_terms
    }


def resolve_relevant_mask(
    ranked_relevance: torch.Tensor,
    relevance_labels: torch.Tensor,
    relevance_scheme: str | None = None,
) -> torch.Tensor:
    """Binary "is this row relevant" mask, used by MRR here and in ``baselines.metrics``.

    With no explicit scheme the threshold is inferred from the labels: graded labels
    (which reach 3) count only grade >= 2 as relevant, binary labels count anything > 0.
    """
    if relevance_scheme is not None and relevance_scheme not in {"graded", "binary"}:
        raise ValueError(f"Unsupported relevance scheme: {relevance_scheme}")

    use_graded_threshold = (
        relevance_scheme == "graded"
        if relevance_scheme is not None
        else bool((relevance_labels > 1).any().item())
    )
    return ranked_relevance >= 2.0 if use_graded_threshold else ranked_relevance > 0.0


def _temperature_scaled_logsumexp(
    values: torch.Tensor,
    temperature: float,
    empty_value: float = 0.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    temperature_tensor = torch.as_tensor(
        float(temperature),
        device=values.device,
        dtype=values.dtype,
    )
    finite_mask = torch.isfinite(values)
    scaled_values = torch.where(
        finite_mask,
        values / temperature_tensor,
        torch.full_like(values, float("-inf")),
    )
    aggregated_values = temperature_tensor * torch.logsumexp(scaled_values, dim=-1)
    default_values = torch.full_like(aggregated_values, fill_value=empty_value)
    return torch.where(finite_mask.any(dim=-1), aggregated_values, default_values)


def compute_reward_from_scores(
    scores: torch.Tensor,
    relevance_labels: torch.Tensor,
    reward_type: str = "ndcg",
    k: int | None = 10,
    ndcg_in_batch_include_negatives: bool = False,
    contrastive_use_in_batch_negatives: bool = False,
    contrastive_temperature: float = 0.03,
    relevance_scheme: str | None = None,
    in_batch_positive_scores: torch.Tensor | None = None,
    in_batch_candidate_scores: torch.Tensor | None = None,
    rank_labels: torch.Tensor | None = None,
    rbo_p: float = 0.9,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if candidate_mask is not None:
        if candidate_mask.shape != relevance_labels.shape or candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be bool [batch, candidates]")
        if not candidate_mask.any(dim=-1).all():
            raise ValueError("Each query needs valid candidates")
        if not candidate_mask.all():
            # Exact compaction also handles order-based rewards (RBO/pairwise) and k=None.
            # Preserve the batch's relevance convention when evaluating individual rows.
            scheme = relevance_scheme or ("graded" if (relevance_labels > 1).any() else "binary")
            return torch.cat([
                compute_reward_from_scores(
                    scores=scores[i:i+1][..., mask], relevance_labels=relevance_labels[i:i+1, mask],
                    reward_type=reward_type, k=k,
                    ndcg_in_batch_include_negatives=ndcg_in_batch_include_negatives,
                    contrastive_use_in_batch_negatives=contrastive_use_in_batch_negatives,
                    contrastive_temperature=contrastive_temperature, relevance_scheme=scheme,
                    in_batch_positive_scores=None if in_batch_positive_scores is None else in_batch_positive_scores[i:i+1],
                    in_batch_candidate_scores=None if in_batch_candidate_scores is None else in_batch_candidate_scores[i:i+1],
                    rank_labels=None if rank_labels is None else rank_labels[i:i+1, mask], rbo_p=rbo_p,
                ) for i, mask in enumerate(candidate_mask)
            ], dim=0)
    squeeze_rollout_dim = False
    if scores.dim() == 2:
        scores = scores.unsqueeze(1)
        squeeze_rollout_dim = True
        if in_batch_positive_scores is not None:
            in_batch_positive_scores = in_batch_positive_scores.unsqueeze(1)
        if in_batch_candidate_scores is not None:
            in_batch_candidate_scores = in_batch_candidate_scores.unsqueeze(1)

    reward_type = reward_type.lower()
    if reward_type not in SUPPORTED_REWARD_TYPES:
        raise ValueError(
            f"Unsupported reward type: {reward_type}. Supported types: {sorted(SUPPORTED_REWARD_TYPES)}"
        )
    if scores.dim() < 3:
        raise ValueError(
            "scores must be [batch, slate] or [batch, *rollout, slate], "
            f"got shape {tuple(scores.shape)}"
        )
    if relevance_labels.dim() != 2:
        raise ValueError(
            f"relevance_labels must be [batch, slate], got shape {tuple(relevance_labels.shape)}"
        )
    if scores.shape[0] != relevance_labels.shape[0] or scores.shape[-1] != relevance_labels.shape[1]:
        raise ValueError(
            "relevance_labels shape must match scores [batch, slate], "
            f"got scores={tuple(scores.shape)} labels={tuple(relevance_labels.shape)}"
        )
    if rank_labels is not None and rank_labels.shape != relevance_labels.shape:
        raise ValueError(
            "rank_labels shape must match relevance_labels [batch, slate], "
            f"got ranks={tuple(rank_labels.shape)} labels={tuple(relevance_labels.shape)}"
        )
    if not 0.0 <= rbo_p < 1.0:
        raise ValueError(f"RBO persistence p must lie in [0, 1), got {rbo_p}")

    batch_size = scores.size(0)
    rollout_shape = scores.shape[1:-1]
    slate_length = scores.size(-1)

    def finish(reward: torch.Tensor) -> torch.Tensor:
        """Fold the flattened rollout axis back into the caller's rollout shape."""
        reward = reward.reshape(batch_size, *rollout_shape)
        return reward.squeeze(1) if squeeze_rollout_dim else reward

    rollout_count = 1
    for rollout_dim in rollout_shape:
        rollout_count *= rollout_dim

    scores = scores.reshape(batch_size, rollout_count, slate_length)
    if in_batch_positive_scores is not None:
        in_batch_positive_scores = in_batch_positive_scores.reshape(batch_size, rollout_count, -1)
    if in_batch_candidate_scores is not None:
        in_batch_candidate_scores = in_batch_candidate_scores.reshape(batch_size, rollout_count, -1)

    def zero_reward() -> torch.Tensor:
        return torch.zeros(batch_size, *rollout_shape, device=scores.device, dtype=scores.dtype)

    expanded_labels = relevance_labels.unsqueeze(1).expand(batch_size, rollout_count, slate_length)
    if reward_type in {"top_weighted_pairwise", "rbo"}:
        if rank_labels is None:
            raise ValueError(f"rank_labels are required for the {reward_type} reward")
        if not torch.isfinite(rank_labels).all():
            raise ValueError("rank_labels must be finite")

        cutoff = slate_length if k is None else min(k, slate_length)
        if cutoff <= 0:
            return finish(zero_reward())

        teacher_order = rank_labels.argsort(dim=-1, descending=True, stable=True)
        positions = torch.arange(1, slate_length + 1, device=scores.device, dtype=torch.long)
        metric_dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
        if reward_type == "rbo":
            predicted_order = scores.argsort(dim=-1, descending=True, stable=True)
            teacher_positions = torch.empty_like(teacher_order)
            teacher_positions.scatter_(
                dim=-1,
                index=teacher_order,
                src=positions.view(1, -1).expand_as(teacher_order),
            )
            predicted_positions = torch.empty_like(predicted_order)
            predicted_positions.scatter_(
                dim=-1,
                index=predicted_order,
                src=positions.view(1, 1, -1).expand_as(predicted_order),
            )

            # A document first enters the prefix intersection at the deeper of its teacher and
            # predicted positions. Histogram those entry depths and cumulatively sum them instead
            # of materializing two [batch, rollout, slate, slate] prefix-membership tensors.
            entry_depths = torch.maximum(predicted_positions, teacher_positions.unsqueeze(1))
            enters_by_cutoff = entry_depths <= cutoff
            overlap_histogram = torch.zeros(
                batch_size,
                rollout_count,
                cutoff,
                device=scores.device,
                dtype=metric_dtype,
            )
            overlap_histogram.scatter_add_(
                dim=-1,
                index=entry_depths.clamp_max(cutoff) - 1,
                src=enters_by_cutoff.to(metric_dtype),
            )
            prefix_overlap = overlap_histogram.cumsum(dim=-1)
            depths = torch.arange(1, cutoff + 1, device=scores.device, dtype=metric_dtype)
            agreement = prefix_overlap / depths
            depth_weights = torch.as_tensor(rbo_p, device=scores.device, dtype=metric_dtype).pow(
                torch.arange(cutoff, device=scores.device, dtype=metric_dtype)
            )
            return finish((agreement * depth_weights).sum(dim=-1) / depth_weights.sum())

        # Compare every teacher-preferred pair whose better item lies inside the teacher's
        # top-k. A pair receives the logarithmic discount of that better item's teacher rank.
        # This uses only ordinal supervision: unlike pseudo-gain nDCG, it invents no relevance
        # grades or gaps between adjacent teacher positions.
        better_positions, worse_positions = torch.triu_indices(
            slate_length,
            slate_length,
            offset=1,
            device=scores.device,
        )
        inside_cutoff = better_positions < cutoff
        better_positions = better_positions[inside_cutoff]
        worse_positions = worse_positions[inside_cutoff]
        better_indices = teacher_order[:, better_positions]
        worse_indices = teacher_order[:, worse_positions]
        pair_count = better_indices.size(-1)
        better_scores = scores.gather(
            dim=-1,
            index=better_indices.unsqueeze(1).expand(batch_size, rollout_count, pair_count),
        )
        worse_scores = scores.gather(
            dim=-1,
            index=worse_indices.unsqueeze(1).expand(batch_size, rollout_count, pair_count),
        )
        better_ranks = rank_labels.gather(dim=-1, index=better_indices)
        worse_ranks = rank_labels.gather(dim=-1, index=worse_indices)
        strict_preference = better_ranks > worse_ranks
        pair_weights = 1.0 / torch.log2(better_positions.to(metric_dtype) + 2.0)
        pair_weights = pair_weights.unsqueeze(0) * strict_preference

        score_differences = better_scores - worse_scores
        concordance = (score_differences > 0).to(metric_dtype)
        concordance = concordance + 0.5 * (score_differences == 0).to(metric_dtype)
        normalizer = pair_weights.sum(dim=-1, keepdim=True)
        weighted_concordance = (pair_weights.unsqueeze(1) * concordance).sum(dim=-1)
        return finish(
            torch.where(
                normalizer > 0,
                weighted_concordance / normalizer,
                torch.zeros_like(weighted_concordance),
            )
        )

    if reward_type in {"ndcg", "ndcg_in_batch"}:
        ranking_scores = scores
        ranking_labels = expanded_labels
        if reward_type == "ndcg_in_batch":
            extra_scores = (
                in_batch_candidate_scores
                if ndcg_in_batch_include_negatives
                else in_batch_positive_scores
            )
            if extra_scores is not None:
                ranking_scores = torch.cat((ranking_scores, extra_scores), dim=-1)
                ranking_labels = torch.cat((ranking_labels, torch.zeros_like(extra_scores)), dim=-1)

        cutoff = ranking_scores.size(-1) if k is None else min(k, ranking_scores.size(-1))
        if cutoff <= 0:
            return finish(zero_reward())

        topk_indices = ranking_scores.topk(k=cutoff, dim=-1).indices
        topk_relevance = ranking_labels.gather(dim=-1, index=topk_indices)
        discounts = 1.0 / torch.log2(torch.arange(2, cutoff + 2, device=scores.device, dtype=scores.dtype))
        dcg = (((2.0 ** topk_relevance) - 1.0) * discounts).sum(dim=-1)

        ideal_relevance = ranking_labels.topk(k=cutoff, dim=-1).values
        idcg = (((2.0 ** ideal_relevance) - 1.0) * discounts).sum(dim=-1)
        return finish(torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg)))

    if reward_type in {"mrr", "mrr_in_batch"}:
        ranking_scores = scores
        ranking_labels = expanded_labels
        if reward_type == "mrr_in_batch":
            extra_scores = (
                in_batch_candidate_scores
                if ndcg_in_batch_include_negatives
                else in_batch_positive_scores
            )
            if extra_scores is not None:
                ranking_scores = torch.cat((ranking_scores, extra_scores), dim=-1)
                ranking_labels = torch.cat((ranking_labels, torch.zeros_like(extra_scores)), dim=-1)

        ranking_length = ranking_scores.size(-1)
        cutoff = ranking_length if k is None else min(k, ranking_length)
        if cutoff <= 0:
            return finish(zero_reward())
        ranked_indices = ranking_scores.topk(k=cutoff, dim=-1).indices
        ranked_relevance = ranking_labels.gather(dim=-1, index=ranked_indices)
        relevant_mask = resolve_relevant_mask(
            ranked_relevance=ranked_relevance,
            relevance_labels=relevance_labels,
            relevance_scheme=relevance_scheme,
        )
        reciprocal_ranks = relevant_mask.to(scores.dtype) / torch.arange(
            1,
            cutoff + 1,
            device=scores.device,
            dtype=scores.dtype,
        )
        return finish(reciprocal_ranks.max(dim=-1).values)

    positive_indices = relevance_labels.argmax(dim=-1)
    arange_b = torch.arange(batch_size, device=scores.device)
    positive_scores = scores[arange_b, :, positive_indices]
    negative_scores = scores.masked_fill(
        F.one_hot(positive_indices, num_classes=slate_length).bool().unsqueeze(1),
        float("-inf"),
    )
    if contrastive_use_in_batch_negatives and in_batch_positive_scores is not None:
        negative_scores = torch.cat((negative_scores, in_batch_positive_scores), dim=-1)

    if reward_type == "contrastive":
        # No positive in the partition: the reward is the margin of the positive over the
        # soft-max of the negatives alone.
        partition_scores = negative_scores
    elif reward_type == "infonce":
        partition_scores = torch.cat((positive_scores.unsqueeze(-1), negative_scores), dim=-1)
    else:
        raise AssertionError(f"Unhandled reward type: {reward_type}")
    return finish(
        positive_scores
        - _temperature_scaled_logsumexp(partition_scores, temperature=contrastive_temperature)
    )
