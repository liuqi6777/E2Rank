import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HFTrainingArguments

from rewards import (
    SUPPORTED_REWARD_TYPES,
    normalize_reward_combine_mode,
    normalize_reward_terms,
    reward_terms_mix_scales,
)


logger = logging.getLogger(__name__)


SUPPORTED_ACTION_COMPONENTS = {"query", "positive", "negative"}

SUPPORTED_ADVANTAGE_NORM_MODES = ("per_component", "shared", "none")

SUPPORTED_ADVANTAGE_BASELINES = ("group", "ema")

SUPPORTED_SAMPLING_LAWS = ("vmf", "gaussian")

SUPPORTED_ROLLOUTS = ("product", "diagonal")


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
                "Number of candidate documents per sample (1 positive + slate_size-1 "
                "negatives). Samples with fewer than slate_size-1 negatives are dropped."
            )
        },
    )
    file_glob: str = field(
        default="*_len-0-500.jsonl",
        metadata={
            "help": (
                "Filename glob used to pick which length bucket(s) to read from each "
                "source subdirectory when data_path is a directory."
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
    pass


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
                "vMF concentration override. When set, sigma is derived as 1/sqrt(kappa). "
                "E.g. kappa=755 matches the sampling concentration (mean cosine 0.53 at d=1024) "
                "of the legacy projected-Gaussian sampler with sigma=0.05."
            )
        },
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
                "Reward type: ndcg, ndcg_in_batch, contrastive, infonce, or mrr. Used when "
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
                "per-term fields inherit reward_ndcg_k / contrastive_temperature / the "
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
        metadata={"help": "Default ranking cutoff for the nDCG/MRR reward terms"},
    )
    ndcg_in_batch_include_negatives: bool = field(
        default=False,
        metadata={"help": "Append all candidates from other samples, not only positives, for ndcg_in_batch"},
    )
    contrastive_use_in_batch_negatives: bool = field(
        default=False,
        metadata={"help": "Use positives from other samples as extra negatives for contrastive reward"},
    )
    contrastive_temperature: float = field(
        default=0.03,
        metadata={"help": "Temperature used by the contrastive/infonce reward"},
    )
    advantage_norm: str = field(
        default="per_component",
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
        default="group",
        metadata={
            "help": (
                "Baseline subtracted from rewards. 'group' is GRPO's per-input group mean; "
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
        self.reward_combine = normalize_reward_combine_mode(self.reward_combine)
        # An empty spec means "single term from reward_type", which resolves to exactly the
        # arguments the pre-combination code passed, so legacy configs are untouched.
        self.reward_terms = normalize_reward_terms(
            self.reward_terms if self.reward_terms else self.reward_type,
            default_k=self.reward_ndcg_k,
            default_temperature=self.contrastive_temperature,
            default_ndcg_in_batch_include_negatives=self.ndcg_in_batch_include_negatives,
            default_contrastive_use_in_batch_negatives=self.contrastive_use_in_batch_negatives,
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
