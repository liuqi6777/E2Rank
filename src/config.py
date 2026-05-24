from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HFTrainingArguments


SUPPORTED_ACTION_COMPONENTS = {"query", "positive", "negative"}


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
        metadata={"help": "Path to the Stage II ranking dataset"}
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

    def __post_init__(self) -> None:
        if self.relevance_scheme not in {"binary", "graded"}:
            raise ValueError(f"Unsupported relevance_scheme: {self.relevance_scheme}")


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
        metadata={"help": "Initial Gaussian exploration scale"},
    )
    sigma_learnable: bool = field(
        default=False,
        metadata={"help": "Learn a global sigma scalar for GRPO"},
    )
    reward_type: str = field(
        default="ndcg",
        metadata={"help": "Reward type: ndcg, ndcg_in_batch, contrastive, infonce, or mrr"},
    )
    reward_ndcg_k: int = field(
        default=10,
        metadata={"help": "Ranking cutoff used by the nDCG/MRR reward component"},
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
    advantage_norm: bool = field(
        default=True,
        metadata={"help": "Normalize GRPO advantages for each sample group"},
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
        if self.kl_coef < 0:
            raise ValueError(f"kl_coef must be non-negative, got {self.kl_coef}")
