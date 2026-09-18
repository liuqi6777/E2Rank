import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HFTrainingArguments

from embedding_protocol import validate_embedding_protocol
from contrastive import validate_aux_infonce
from shortlists import validate_shortlist_sampling, validate_shortlist_objectives
from rewards import (
    SUPPORTED_REWARD_TYPES,
    normalize_reward_combine_mode,
    normalize_reward_terms,
    reward_terms_mix_scales,
)


logger = logging.getLogger(__name__)


SUPPORTED_ACTION_COMPONENTS = {"query", "positive", "negative"}

SUPPORTED_ADVANTAGE_NORM_MODES = ("per_component", "shared", "none")

SUPPORTED_ADVANTAGE_BASELINES = ("group", "leave_one_out", "ema")

SUPPORTED_DOCUMENT_ADVANTAGE_BASELINES = ("shared", "counterfactual")

SUPPORTED_SAMPLING_LAWS = ("vmf", "gaussian")

SUPPORTED_ROLLOUTS = ("product", "diagonal")

SUPPORTED_BASELINE_LOSSES = ("infonce", "ranknet", "lambdaloss")

SUPPORTED_GRADIENT_ESTIMATORS = ("score_function", "conditional_projection")


def validate_reward_shortlists(count, size, hard_count, hard_pool_size, *,
                              reward_cross_device_negatives, cross_query_document_gradients,
                              rollout_seed, pool_source="cross_device_all", **policy):
    if pool_source not in {"cross_device_all", "local_all", "cross_device_representatives"}:
        raise ValueError("Unknown reward_shortlist_pool_source")
    if not count and pool_source != "cross_device_all":
        raise ValueError("A custom shortlist pool source requires reward_shortlist_count > 0")
    validate_shortlist_sampling(count, size, hard_count, hard_pool_size)
    if not count:
        return
    if reward_cross_device_negatives != (pool_source != "local_all"):
        raise ValueError("reward_cross_device_negatives must match reward_shortlist_pool_source")
    if cross_query_document_gradients or rollout_seed is None:
        raise ValueError("Reward shortlists require fixed documents and an explicit rollout_seed")
    # Both SF and CP must estimate the same fixed-kappa, unnormalized LOO target.
    validate_gradient_estimator("conditional_projection", **policy)


def validate_cross_query_document_gradients(enabled, *, reward_terms, action_components,
                                           sampling_law, sigma_learnable, rollout,
                                           advantage_baseline, advantage_norm, reward_combine,
                                           document_advantage_baseline, document_log_prob_reduction,
                                           in_batch_use_sampled_documents, dynamic_retrieval=False,
                                           reward_cross_device_negatives=False, rollout_seed=None):
    if not enabled:
        return
    if (dynamic_retrieval or len(action_components) != 2 or ("query",) not in action_components
            or not any(set(group) == {"positive", "negative"} for group in action_components)
            or sampling_law != "vmf" or sigma_learnable or rollout != "product"
            or advantage_baseline != "leave_one_out" or advantage_norm != "none"
            or reward_combine != "sum" or document_advantage_baseline != "shared"
            or document_log_prob_reduction != "sum" or in_batch_use_sampled_documents):
        raise ValueError("Cross-query document gradients require static joint vMF product rollouts, "
                         "fixed kappa, LOO without normalization, shared document baseline, sum reductions, "
                         "and the legacy in_batch_use_sampled_documents flag disabled")
    if not reward_terms or any(term.type not in {"ndcg_in_batch", "mrr_in_batch"}
                               or term.k is None or term.k <= 0 for term in reward_terms):
        raise ValueError("Cross-query document gradients require in-batch nDCG/MRR with a positive cutoff")
    if reward_cross_device_negatives and rollout_seed is None:
        raise ValueError("Cross-device document policies require an explicit rollout_seed for independent rank-local actions")


def validate_reward_cross_device_negatives(enabled, *, reward_terms, action_components,
                                         rollout, in_batch_use_sampled_documents,
                                         document_advantage_baseline, dynamic_retrieval=False):
    if not enabled:
        return
    if (dynamic_retrieval or rollout != "product" or in_batch_use_sampled_documents
            or document_advantage_baseline != "shared"
            or tuple(map(tuple, action_components)) != (("query",), ("positive", "negative"))):
        raise ValueError("Cross-device reward negatives require static joint product rollouts with fixed cross documents")
    if not reward_terms or any(
        term.type not in {"ndcg_in_batch", "mrr_in_batch"}
        or not term.ndcg_in_batch_include_negatives or term.k is None or term.k <= 0
        for term in reward_terms
    ):
        raise ValueError("Cross-device reward negatives require in-batch nDCG/MRR with all candidates and a positive cutoff")


def validate_gradient_estimator(
    mode, *, action_components, sampling_law, sigma_learnable, rollout,
    advantage_baseline, advantage_norm, reward_combine,
    in_batch_use_sampled_documents, document_advantage_baseline,
    document_log_prob_reduction, dynamic_retrieval=False,
):
    if mode not in SUPPORTED_GRADIENT_ESTIMATORS:
        raise ValueError(f"Unsupported gradient_estimator: {mode!r}")
    if mode == "score_function":
        return
    requirements = {
        "joint query and full-document actions": (
            len(action_components) == 2 and ("query",) in action_components
            and any(set(group) == {"positive", "negative"} for group in action_components)
        ),
        "sampling_law='vmf'": sampling_law == "vmf",
        "sigma_learnable=false": not sigma_learnable,
        "rollout='product'": rollout == "product",
        "advantage_baseline='leave_one_out'": advantage_baseline == "leave_one_out",
        "advantage_norm='none'": advantage_norm == "none",
        "reward_combine='sum'": reward_combine == "sum",
        "in_batch_use_sampled_documents=false": not in_batch_use_sampled_documents,
        "document_advantage_baseline='shared'": document_advantage_baseline == "shared",
        "document_log_prob_reduction='sum'": document_log_prob_reduction == "sum",
        "static candidates": not dynamic_retrieval,
    }
    missing = [name for name, valid in requirements.items() if not valid]
    if missing:
        raise ValueError("gradient_estimator='conditional_projection' requires " + ", ".join(missing))


def validate_document_advantage_baseline(
    mode, *, action_components, sampling_law, sigma_learnable, rollout,
    advantage_baseline, advantage_norm, reward_combine,
    in_batch_use_sampled_documents, dynamic_retrieval=False,
):
    """Limit the first factorized baseline to the estimator it was derived for."""
    if mode not in SUPPORTED_DOCUMENT_ADVANTAGE_BASELINES:
        raise ValueError(f"Unsupported document_advantage_baseline: {mode!r}")
    if mode == "shared":
        return
    requirements = {
        "a joint positive/negative document action group": any(
            set(group) == {"positive", "negative"} for group in action_components
        ),
        "sampling_law='vmf'": sampling_law == "vmf",
        # Per-document differences do not sum to zero; the dropped vMF normalizer
        # would then contribute a missing gradient if kappa were learnable.
        "sigma_learnable=false": not sigma_learnable,
        "rollout='product'": rollout == "product",
        "advantage_baseline='leave_one_out'": advantage_baseline == "leave_one_out",
        "advantage_norm='none'": advantage_norm == "none",
        "reward_combine='sum'": reward_combine == "sum",
        "in_batch_use_sampled_documents=false": not in_batch_use_sampled_documents,
        "static candidates": not dynamic_retrieval,
    }
    missing = [name for name, valid in requirements.items() if not valid]
    if missing:
        raise ValueError("document_advantage_baseline='counterfactual' requires " + ", ".join(missing))


def normalize_advantage_norm_mode(mode) -> str:
    """Map legacy bool values (and their YAML/CLI string forms) onto the mode names."""
    if isinstance(mode, bool):
        return "per_component" if mode else "none"
    if isinstance(mode, str):
        lowered = mode.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return "per_component"
        if lowered in {"false", "0", "no"}:
            return "none"
        if lowered in SUPPORTED_ADVANTAGE_NORM_MODES:
            return lowered
    raise ValueError(
        f"Unsupported advantage_norm mode: {mode!r}. "
        f"Expected one of {SUPPORTED_ADVANTAGE_NORM_MODES} (or a legacy bool)."
    )


def normalize_action_components(action_components) -> tuple[tuple[str, ...], ...]:
    if isinstance(action_components, str):
        groups = []
        for raw_group in action_components.split(";"):
            raw_group = raw_group.strip()
            if not raw_group:
                continue
            groups.append([component.strip() for component in raw_group.split(",") if component.strip()])
    elif isinstance(action_components, Sequence):
        groups = action_components
    else:
        raise ValueError("action_components must be a semicolon string or a list of component groups")

    normalized_groups: list[tuple[str, ...]] = []
    seen_components: set[str] = set()
    for group in groups:
        if isinstance(group, str):
            components = tuple(component.strip() for component in group.split(",") if component.strip())
        elif isinstance(group, Sequence):
            components = tuple(str(component).strip() for component in group if str(component).strip())
        else:
            raise ValueError("Each action component group must be a string or a list of strings")

        if not components:
            raise ValueError("action component groups must not be empty")
        invalid_components = sorted(set(components) - SUPPORTED_ACTION_COMPONENTS)
        if invalid_components:
            raise ValueError(f"Unsupported action component(s): {', '.join(invalid_components)}")
        if "query" in components and len(components) > 1:
            raise ValueError("query must be its own action component group")
        if len(set(components)) != len(components):
            raise ValueError(f"Duplicate component in action group: {components}")
        duplicate_components = sorted(seen_components.intersection(components))
        if duplicate_components:
            raise ValueError(f"Duplicate action component(s): {', '.join(duplicate_components)}")

        seen_components.update(components)
        normalized_groups.append(components)

    if not normalized_groups:
        raise ValueError("At least one action component group must be configured")

    return tuple(normalized_groups)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to a pretrained model or a Hugging Face model ID"}
    )
    model_revision: Optional[str] = field(
        default=None,
        metadata={"help": "Pinned revision shared by the backbone, config and tokenizer"},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional config path if it differs from model_name_or_path"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional tokenizer path if it differs from model_name_or_path"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory used to cache downloaded model files"},
    )
    pooling_method: str = field(
        default="last",
        metadata={"help": "Embedding pooling rule: last, mean, or cls"},
    )
    padding_side: str = field(
        default="left",
        metadata={"help": "Tokenizer padding side used by this embedding checkpoint"},
    )
    append_token: str = field(
        default="pad",
        metadata={
            "help": (
                "Terminal token appended after truncating raw content IDs: none, eos, or pad. "
                "Explicit terminals disable automatic special tokens; none uses the native tokenizer."
            )
        },
    )
    query_prompt_template: str = field(
        default="Instruct: {task_description}\nQuery:{query}",
        metadata={
            "help": (
                "Query formatting template. Supports {text}/{query} and {task_description}."
            )
        },
    )
    document_prompt_template: str = field(
        default="{document}",
        metadata={"help": "Document formatting template. Supports {text}/{document}."},
    )
    embedding_max_length: int = field(
        default=8192,
        metadata={
            "help": (
                "Maximum sequence length supported by the embedding checkpoint. Training "
                "query/document limits are clamped to this value."
            )
        },
    )
    document_encoder_mode: str = field(
        default="joint",
        metadata={"help": "Document encoder mode: joint or frozen_index"},
    )
    frozen_document_index_manifest: Optional[str] = field(
        default=None,
        metadata={"help": "Immutable index manifest required by frozen_index mode"},
    )
    frozen_document_verify_hashes: bool = field(
        default=True,
        metadata={"help": "Verify every frozen corpus artifact before training"},
    )
    frozen_document_index_backend: str = field(
        default="lookup",
        metadata={"help": "Frozen index backend: lookup, torch, or faiss"},
    )
    frozen_document_search_batch_size: int = field(
        default=1024,
        metadata={"help": "Query batch size used by full-corpus index search"},
    )

    def __post_init__(self) -> None:
        self.pooling_method = self.pooling_method.strip().lower()
        self.padding_side = self.padding_side.strip().lower()
        self.append_token = self.append_token.strip().lower()
        self.document_encoder_mode = self.document_encoder_mode.strip().lower()
        self.frozen_document_index_backend = self.frozen_document_index_backend.strip().lower()
        if self.document_encoder_mode not in {"joint", "frozen_index"}:
            raise ValueError("document_encoder_mode must be joint or frozen_index")
        if self.frozen_document_index_backend not in {"lookup", "torch", "faiss"}:
            raise ValueError("frozen_document_index_backend must be lookup, torch, or faiss")
        if self.frozen_document_search_batch_size <= 0:
            raise ValueError("frozen_document_search_batch_size must be positive")
        if self.document_encoder_mode == "frozen_index" and not self.frozen_document_index_manifest:
            raise ValueError(
                "frozen_document_index_manifest is required when document_encoder_mode=frozen_index"
            )
        if self.embedding_max_length <= 0:
            raise ValueError(
                f"embedding_max_length must be positive, got {self.embedding_max_length}"
            )
        validate_embedding_protocol(
            pooling_method=self.pooling_method,
            padding_side=self.padding_side,
            append_token=self.append_token,
            query_prompt_template=self.query_prompt_template,
            document_prompt_template=self.document_prompt_template,
        )


@dataclass
class DataArguments:
    data_path: str = field(
        metadata={"help": "Path to the Stage II embedding dataset"}
    )
    per_dataset_max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to keep from each source dataset. Use null to keep all samples."},
    )
    q_max_len: int = field(
        default=512,
        metadata={"help": "Maximum token length for query inputs"},
    )
    d_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum token length for document inputs"},
    )
    relevance_scheme: str = field(
        default="binary",
        metadata={"help": "Relevance labels: binary or graded"},
    )
    dev_samples_per_source: int = field(
        default=0,
        metadata={
            "help": (
                "Hold out this many examples per source as a development split. 0 disables it. "
                "The split is taken after the seeded per-source shuffle and before the training "
                "cap, so it is disjoint from training data and stable across runs with the same "
                "seed. Needed to tune smoothing/LR without touching the evaluation benchmark."
            )
        },
    )
    slate_size: int = field(
        default=8,
        metadata={
            "help": (
                "Maximum candidate documents per sample (1 positive + up to "
                "slate_size-1 negatives). Shorter samples are padded and masked per batch."
            )
        },
    )
    file_glob: str = field(
        default="*_len-0-500.jsonl",
        metadata={
            "help": (
                "Filename glob(s) used to pick which length bucket(s) to read from each "
                "source subdirectory when data_path is a directory. Accepts a "
                "comma-separated list to mix buckets, e.g. "
                "'*_len-0-500.jsonl,*_len-500-1000.jsonl'; matches are unioned and "
                "de-duplicated."
            )
        },
    )
    batch_per_length_bucket: bool = field(
        default=False,
        metadata={
            "help": (
                "When file_glob matches several length buckets, batch each bucket "
                "separately so every micro-batch holds documents of one length range "
                "(reduces padding waste). The per-source training cap "
                "(per_dataset_max_samples) still applies per source, not per bucket. "
                "Off by default, keeping single-bucket runs unchanged."
            )
        },
    )
    include_sources: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated source subdirectory names to include (e.g. "
                "'MSMARCO,NQ,HotpotQA'). None includes every subdirectory under data_path."
            )
        },
    )
    index_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Directory to cache the per-file byte-offset index used for lazy loading. "
                "None writes a sidecar '<file>.e2rank_idx.json' next to each data file."
            )
        },
    )

    def __post_init__(self) -> None:
        if self.relevance_scheme not in {"binary", "graded"}:
            raise ValueError(f"Unsupported relevance_scheme: {self.relevance_scheme}")
        if self.dev_samples_per_source < 0:
            raise ValueError(
                f"dev_samples_per_source must be non-negative, got {self.dev_samples_per_source}"
            )
        if self.slate_size < 2:
            raise ValueError(f"slate_size must be >= 2, got {self.slate_size}")


@dataclass
class TrainingArguments(HFTrainingArguments):
    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "Allow training into a non-empty output directory"},
    )


@dataclass
class BaselineArguments:
    """Objective settings for supervised post-training controls."""

    baseline_loss: str = field(
        default="infonce",
        metadata={"help": "Supervised objective: infonce, ranknet, or lambdaloss"},
    )
    baseline_temperature: float = field(
        default=0.03,
        metadata={
            "help": (
                "Temperature applied to cosine scores before the supervised loss. "
                "Used by InfoNCE and RankNet."
            )
        },
    )
    baseline_ndcg_k: int = field(
        default=10,
        metadata={"help": "nDCG cutoff used by the LambdaRank LambdaLoss variant"},
    )
    lambdaloss_sigma: float = field(
        default=1.0,
        metadata={"help": "Pairwise logistic scale used by LambdaLoss"},
    )
    baseline_use_in_batch_negatives: bool = field(
        default=False,
        metadata={
            "help": (
                "Append the other samples' positive documents to each query's "
                "candidate pool. Their embeddings are detached in the cross-query "
                "scores, matching the frozen in-batch candidates used by RL."
            )
        },
    )
    baseline_in_batch_include_negatives: bool = field(
        default=False,
        metadata={"help": "Reuse all other queries' candidates, rather than only representative positives"},
    )
    baseline_cross_device_negatives: bool = field(
        default=False,
        metadata={"help": "Gather cross-query candidates across the data-parallel process group"},
    )
    baseline_detach_in_batch_documents: bool = field(
        default=True,
        metadata={"help": "Stop document gradients from cross-query negative scores"},
    )

    @property
    def extended_negative_pool(self) -> bool:
        return (self.baseline_in_batch_include_negatives
                or self.baseline_cross_device_negatives
                or not self.baseline_detach_in_batch_documents)

    def __post_init__(self) -> None:
        self.baseline_loss = self.baseline_loss.strip().lower()
        if self.baseline_loss not in SUPPORTED_BASELINE_LOSSES:
            raise ValueError(
                f"Unsupported baseline_loss: {self.baseline_loss!r}. "
                f"Expected one of {SUPPORTED_BASELINE_LOSSES}."
            )
        if self.baseline_temperature <= 0:
            raise ValueError(
                "baseline_temperature must be positive, "
                f"got {self.baseline_temperature}"
            )
        if self.extended_negative_pool and (
            self.baseline_loss != "infonce" or not self.baseline_use_in_batch_negatives
        ):
            raise ValueError("Extended baseline negatives require InfoNCE and baseline_use_in_batch_negatives=true")
        if self.baseline_ndcg_k <= 0:
            raise ValueError(
                f"baseline_ndcg_k must be positive, got {self.baseline_ndcg_k}"
            )
        if self.lambdaloss_sigma <= 0:
            raise ValueError(
                f"lambdaloss_sigma must be positive, got {self.lambdaloss_sigma}"
            )


@dataclass
class MTEBEvalArguments:
    mteb_eval_tasks: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated MTEB task names to run after each checkpoint save"},
    )
    mteb_eval_benchmark: Optional[str] = field(
        default=None,
        metadata={"help": "MTEB benchmark name to run after each checkpoint save"},
    )
    mteb_eval_langs: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated MTEB languages used when selecting tasks"},
    )
    mteb_eval_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory for in-training MTEB results. Defaults to <output_dir>/mteb_eval"},
    )
    mteb_eval_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size used by MTEB encoding during in-training eval"},
    )
    mteb_eval_precision: str = field(
        default="fp16",
        metadata={"help": "MTEB model precision: amp_fp16, amp_bf16, fp16, bf16, or fp32"},
    )
    mteb_eval_model_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "JSON object passed as model_kwargs to the MTEB wrapper"},
    )
    mteb_eval_encode_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "JSON object passed as encode_kwargs to MTEB.run()"},
    )
    mteb_eval_run_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "JSON object passed as extra kwargs to MTEB.run()"},
    )


@dataclass
class LoraArguments:
    lora_enabled: bool = False
    lora_path: Optional[str] = None
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    )
    lora_bias: str = "none"


@dataclass
class RLArguments:
    gradient_estimator: str = field(
        default="score_function",
        metadata={"help": "score_function or conditional_projection (joint static vMF product rollouts)"},
    )
    rollout_seed: Optional[int] = field(
        default=None,
        metadata={"help": "Independent action-sampling seed; None preserves the legacy global RNG"},
    )
    dynamic_retrieval: bool = field(
        default=False,
        metadata={"help": "Retrieve candidates from the frozen full corpus for every query action"},
    )
    dynamic_retrieval_k: int = field(
        default=20,
        metadata={"help": "Number of full-corpus results retrieved per sampled query action"},
    )
    action_components: str = field(
        default="query",
        metadata={
            "help": (
                "GRPO action component groups. YAML may use nested lists, e.g. "
                "[[query], [positive, negative]]. CLI may use 'query;positive,negative'."
            )
        },
    )
    group_size: int = field(
        default=8,
        metadata={"help": "Number of sampled actions per input"},
    )
    sigma: float = field(
        default=0.05,
        metadata={
            "help": (
                "Exploration scale; the vMF policy concentration is kappa = 1/sigma^2. "
                "Ignored when kappa is set explicitly."
            )
        },
    )
    kappa: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "vMF concentration override (unless target_alignment is set). sigma = 1/sqrt(kappa). "
                "E.g. kappa=755 matches the sampling concentration (mean cosine 0.53 at d=1024) "
                "of the legacy projected-Gaussian sampler with sigma=0.05."
            )
        },
    )
    target_alignment: Optional[float] = field(default=None, metadata={"help": "Target vMF mean cosine; overrides kappa using the actual embedding dimension"})
    final_alignment: Optional[float] = field(default=None, metadata={"help": "Final mean cosine for linear exploration shrinkage"})
    exploration_schedule: str = field(default="fixed", metadata={"help": "fixed or linear, indexed by optimizer steps"})
    document_log_prob_reduction: str = field(default="sum", metadata={"help": "sum is the joint policy density; mean applies legacy 1/n role weighting"})
    document_advantage_baseline: str = field(
        default="shared",
        metadata={"help": "shared uses the component baseline; counterfactual replaces one document action with its unit mean direction"},
    )
    sigma_learnable: bool = field(
        default=False,
        metadata={"help": "Learn a global sigma scalar for GRPO"},
    )
    sigma_min: float = field(
        default=1e-3,
        metadata={"help": "Lower clamp for learnable sigma (prevents exploration collapse)"},
    )
    sigma_max: float = field(
        default=0.5,
        metadata={"help": "Upper clamp for learnable sigma"},
    )
    reward_type: str = field(
        default="ndcg",
        metadata={
            "help": (
                "Reward type: ndcg, ndcg_in_batch, top_weighted_pairwise, rbo, "
                "contrastive, infonce, mrr, or mrr_in_batch. Used when "
                "reward_terms is empty, and as the per-term default elsewhere."
            )
        },
    )
    reward_terms: str = field(
        default="",
        metadata={
            "help": (
                "Additive reward terms. Empty falls back to the single reward_type. YAML may "
                "use a list of mappings, e.g. [{type: ndcg_in_batch, weight: 1.0, k: 16}, "
                "{type: contrastive, weight: 0.5, in_batch_negatives: true}]. CLI may use "
                "'ndcg_in_batch:1.0,k=16;contrastive:0.5,in_batch_negatives=true'. Unset "
                "per-term fields inherit reward_ndcg_k / reward_rbo_p / "
                "contrastive_temperature / the "
                "in-batch flags below."
            )
        },
    )
    reward_combine: str = field(
        default="sum",
        metadata={
            "help": (
                "How multiple reward terms are combined. 'sum' adds the raw rewards and takes "
                "one advantage, preserving each term's true effect size (so the effective "
                "mixing ratio is weight x within-group std, not the weight alone). "
                "'normalized_sum' standardizes each term's advantage separately before the "
                "weighted sum, making the weights scale-free -- use it whenever bounded "
                "ranking rewards are mixed with contrastive margins."
            )
        },
    )
    reward_ndcg_k: int = field(
        default=10,
        metadata={"help": "Default cutoff for ranking reward terms"},
    )
    reward_rbo_p: float = field(
        default=0.9,
        metadata={
            "help": (
                "Default RBO persistence in [0, 1); larger values spread more reward "
                "weight across deeper prefixes"
            )
        },
    )
    ndcg_in_batch_include_negatives: bool = field(
        default=False,
        metadata={"help": "Append all candidates from other samples, not only positives, for ndcg_in_batch"},
    )
    reward_cross_device_negatives: bool = field(
        default=False,
        metadata={"help": "Use the Strong CL cross-device document pool for static joint ranking RL"},
    )
    reward_shortlist_pool_source: str = field(
        default="cross_device_all",
        metadata={"help": "Shortlist source: cross_device_all, local_all, or cross_device_representatives"},
    )
    reward_shortlist_count: int = field(
        default=0, metadata={"help": "Number of separately rewarded/projected cross-pool shortlists; 0 disables"},
    )
    reward_shortlist_size: int = field(
        default=15, metadata={"help": "Cross-query negatives per shortlist; own candidates are always retained"},
    )
    reward_shortlist_hard_count: int = field(
        default=8, metadata={"help": "Negatives per shortlist drawn from the highest-scoring stratum; 0 is uniform"},
    )
    reward_shortlist_hard_pool_size: int = field(
        default=64, metadata={"help": "Size of the high-score stratum, ranked using detached means"},
    )
    reward_shortlist_binary_weight: float = field(
        default=0.0,
        metadata={"help": "Mix original-positive binary nDCG into shortlist nDCG: (1-alpha)*base + alpha*binary"},
    )
    cross_query_document_gradients: bool = field(
        default=False,
        metadata={"help": "Share sampled document actions across queries and accumulate all reward gradients; supports local and cross-device pools"},
    )
    contrastive_use_in_batch_negatives: bool = field(
        default=False,
        metadata={"help": "Use positives from other samples as extra negatives for contrastive reward"},
    )
    contrastive_temperature: float = field(
        default=0.03,
        metadata={"help": "Temperature used by the contrastive/infonce reward"},
    )
    aux_infonce_coef: float = field(
        default=0.0,
        metadata={"help": "Weight of direct multi-positive InfoNCE on unperturbed embeddings; 0 disables it"},
    )
    aux_infonce_temperature: float = field(
        default=0.03,
        metadata={"help": "Temperature of the direct InfoNCE auxiliary loss, independent of reward temperature"},
    )
    aux_infonce_use_in_batch_negatives: bool = field(
        default=False,
        metadata={"help": "Append masked, detached cross-query representatives to the auxiliary InfoNCE candidates"},
    )
    aux_infonce_strong_negatives: bool = field(
        default=False,
        metadata={"help": "Use all cross-device candidates and full document gradients within the RL action scope"},
    )
    advantage_norm: str = field(
        default="none",
        metadata={
            "help": (
                "Advantage normalization: 'per_component' divides each component's group by its own "
                "std (legacy true), 'shared' divides all components by the per-sample std of the raw "
                "reward tensor (preserves relative effect sizes between components), 'none' only "
                "centers (legacy false)."
            )
        },
    )
    sampling_law: str = field(
        default="vmf",
        metadata={
            "help": (
                "Law the actions are drawn from. 'vmf' samples exactly (Wood's rejection "
                "sampler). 'gaussian' is the projected-Gaussian shortcut e = normalize(h + "
                "sigma*eps), which is scored under the vMF log-density it was NOT drawn from; "
                "it exists as an ablation of sampling fidelity, not as a supported mode."
            )
        },
    )
    rollout: str = field(
        default="product",
        metadata={
            "help": (
                "'product' evaluates the full cross product of per-component samples (one "
                "reward-tensor axis per sampled component). 'diagonal' evaluates only the paired "
                "entries r^(g,g,...,g), so every component shares one group index and one "
                "advantage vector."
            )
        },
    )
    frozen_doc_rescale: bool = field(
        default=True,
        metadata={
            "help": (
                "Rescale frozen-document score tables by the policy's mean resultant length so "
                "frozen and sampled candidates share a score scale. Disabling it reproduces the "
                "reward-collapse failure mode and exists only as an ablation."
            )
        },
    )
    advantage_baseline: str = field(
        default="leave_one_out",
        metadata={
            "help": (
                "Baseline: 'leave_one_out' excludes the current action; 'group' is the group mean; "
                "'ema' is a single global exponential-moving-average scalar, which reduces the "
                "method to REINFORCE with a running baseline and exists as an ablation."
            )
        },
    )
    advantage_baseline_momentum: float = field(
        default=0.99,
        metadata={"help": "Momentum of the running baseline when advantage_baseline='ema'"},
    )
    in_batch_use_sampled_documents: bool = field(
        default=False,
        metadata={
            "help": (
                "Legacy behavior: score in-batch candidates with their sampled (perturbed) embeddings, "
                "which leaks other samples' perturbations into each sample's advantages via the shared "
                "group index. Default False scores them with detached mean embeddings so per-sample "
                "credit assignment stays exact."
                " Use cross_query_document_gradients for shared actions with complete document gradients."
            )
        },
    )
    kl_coef: float = field(
        default=0.0,
        metadata={
            "help": (
                "KL penalty coefficient between the adapter-enabled policy and the base "
                "(LoRA-disabled) reference. 0 disables the KL term. Requires a PEFT/LoRA model."
            )
        },
    )

    def __post_init__(self) -> None:
        from policy_math import validate_exploration
        from rollout_rng import validate_rollout_seed
        validate_aux_infonce(self.aux_infonce_coef, self.aux_infonce_temperature)
        validate_rollout_seed(self.rollout_seed)
        if self.rollout_seed is not None and self.dynamic_retrieval:
            raise ValueError("rollout_seed is currently supported for static-candidate GRPO only")
        validate_exploration(self.target_alignment, self.final_alignment, self.exploration_schedule)
        if self.document_log_prob_reduction not in {"sum", "mean"}:
            raise ValueError("document_log_prob_reduction must be sum or mean")
        if self.target_alignment is not None and (self.sigma_learnable or self.sampling_law != "vmf"):
            raise ValueError("Alignment-based exploration requires vMF with sigma_learnable=false")
        if self.dynamic_retrieval_k <= 0:
            raise ValueError("dynamic_retrieval_k must be positive")
        self.action_components = normalize_action_components(self.action_components)
        self.advantage_norm = normalize_advantage_norm_mode(self.advantage_norm)
        if self.sampling_law not in SUPPORTED_SAMPLING_LAWS:
            raise ValueError(
                f"Unsupported sampling_law: {self.sampling_law!r}. "
                f"Expected one of {SUPPORTED_SAMPLING_LAWS}."
            )
        if self.rollout not in SUPPORTED_ROLLOUTS:
            raise ValueError(
                f"Unsupported rollout: {self.rollout!r}. Expected one of {SUPPORTED_ROLLOUTS}."
            )
        if self.advantage_baseline not in SUPPORTED_ADVANTAGE_BASELINES:
            raise ValueError(
                f"Unsupported advantage_baseline: {self.advantage_baseline!r}. "
                f"Expected one of {SUPPORTED_ADVANTAGE_BASELINES}."
            )
        if not 0.0 <= self.advantage_baseline_momentum < 1.0:
            raise ValueError(
                f"advantage_baseline_momentum must lie in [0, 1), got {self.advantage_baseline_momentum}"
            )
        if self.kappa is not None:
            if self.kappa <= 0:
                raise ValueError(f"kappa must be positive, got {self.kappa}")
            self.sigma = self.kappa ** -0.5
        if self.kl_coef < 0:
            raise ValueError(f"kl_coef must be non-negative, got {self.kl_coef}")

        self.reward_type = str(self.reward_type).strip().lower()
        if self.reward_type not in SUPPORTED_REWARD_TYPES:
            raise ValueError(
                f"Unsupported reward type: {self.reward_type}. "
                f"Supported types: {sorted(SUPPORTED_REWARD_TYPES)}"
            )
        if self.contrastive_temperature <= 0:
            raise ValueError(
                f"contrastive_temperature must be positive, got {self.contrastive_temperature}"
            )
        if not 0.0 <= self.reward_rbo_p < 1.0:
            raise ValueError(
                f"reward_rbo_p must lie in [0, 1), got {self.reward_rbo_p}"
            )
        self.reward_combine = normalize_reward_combine_mode(self.reward_combine)
        validate_document_advantage_baseline(
            self.document_advantage_baseline, action_components=self.action_components,
            sampling_law=self.sampling_law, sigma_learnable=self.sigma_learnable,
            rollout=self.rollout, advantage_baseline=self.advantage_baseline,
            advantage_norm=self.advantage_norm, reward_combine=self.reward_combine,
            in_batch_use_sampled_documents=self.in_batch_use_sampled_documents,
            dynamic_retrieval=self.dynamic_retrieval,
        )
        validate_gradient_estimator(
            self.gradient_estimator, action_components=self.action_components,
            sampling_law=self.sampling_law, sigma_learnable=self.sigma_learnable,
            rollout=self.rollout, advantage_baseline=self.advantage_baseline,
            advantage_norm=self.advantage_norm, reward_combine=self.reward_combine,
            in_batch_use_sampled_documents=self.in_batch_use_sampled_documents,
            document_advantage_baseline=self.document_advantage_baseline,
            document_log_prob_reduction=self.document_log_prob_reduction,
            dynamic_retrieval=self.dynamic_retrieval,
        )
        # An empty spec means "single term from reward_type", which resolves to exactly the
        # arguments the pre-combination code passed, so legacy configs are untouched.
        self.reward_terms = normalize_reward_terms(
            self.reward_terms if self.reward_terms else self.reward_type,
            default_k=self.reward_ndcg_k,
            default_temperature=self.contrastive_temperature,
            default_rbo_p=self.reward_rbo_p,
            default_ndcg_in_batch_include_negatives=self.ndcg_in_batch_include_negatives,
            default_contrastive_use_in_batch_negatives=self.contrastive_use_in_batch_negatives,
        )
        validate_reward_cross_device_negatives(
            self.reward_cross_device_negatives or self.reward_shortlist_count > 0, reward_terms=self.reward_terms,
            action_components=self.action_components, rollout=self.rollout,
            in_batch_use_sampled_documents=self.in_batch_use_sampled_documents,
            document_advantage_baseline=self.document_advantage_baseline,
            dynamic_retrieval=self.dynamic_retrieval,
        )
        validate_cross_query_document_gradients(
            self.cross_query_document_gradients, reward_terms=self.reward_terms,
            action_components=self.action_components, sampling_law=self.sampling_law,
            sigma_learnable=self.sigma_learnable, rollout=self.rollout,
            advantage_baseline=self.advantage_baseline, advantage_norm=self.advantage_norm,
            reward_combine=self.reward_combine, document_advantage_baseline=self.document_advantage_baseline,
            document_log_prob_reduction=self.document_log_prob_reduction,
            in_batch_use_sampled_documents=self.in_batch_use_sampled_documents,
            dynamic_retrieval=self.dynamic_retrieval,
            reward_cross_device_negatives=self.reward_cross_device_negatives,
            rollout_seed=self.rollout_seed,
        )
        validate_reward_shortlists(
            self.reward_shortlist_count, self.reward_shortlist_size,
            self.reward_shortlist_hard_count, self.reward_shortlist_hard_pool_size,
            pool_source=self.reward_shortlist_pool_source,
            reward_cross_device_negatives=self.reward_cross_device_negatives,
            cross_query_document_gradients=self.cross_query_document_gradients,
            rollout_seed=self.rollout_seed,
            **{key: getattr(self, key) for key in (
                "action_components", "sampling_law", "sigma_learnable", "rollout",
                "advantage_baseline", "advantage_norm", "reward_combine",
                "in_batch_use_sampled_documents", "document_advantage_baseline",
                "document_log_prob_reduction", "dynamic_retrieval")},
        )
        validate_shortlist_objectives(
            self.reward_shortlist_count, self.reward_terms, self.reward_shortlist_binary_weight,
        )
        if (
            len(self.reward_terms) > 1
            and self.reward_combine == "sum"
            and reward_terms_mix_scales(self.reward_terms)
        ):
            logger.warning(
                "reward_combine='sum' mixes bounded ranking rewards with unbounded contrastive "
                "margins (%s). Advantages are divided by the group std of the COMBINED reward, "
                "so each term contributes in proportion to weight x its own within-group std, "
                "not to its weight alone -- watch reward/<term>/group_std to see the ratio you "
                "actually got. Use reward_combine='normalized_sum' for weights that mean what "
                "they say.",
                ", ".join(f"{term.name}(w={term.weight})" for term in self.reward_terms),
            )
        if (
            self.advantage_baseline == "ema"
            and self.reward_combine == "normalized_sum"
            and len(self.reward_terms) > 1
        ):
            raise ValueError(
                "advantage_baseline='ema' is incompatible with reward_combine='normalized_sum' "
                "for multiple terms: the running baseline is a single global scalar and cannot "
                "track per-term reward scales. Use reward_combine='sum' or the group baseline."
            )
