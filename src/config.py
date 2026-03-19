from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HFTrainingArguments


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
    use_listwise: bool = field(
        default=True,
        metadata={"help": "Build listwise prompts for the reranking GRPO branch"},
    )
    q_max_len: int = field(
        default=512,
        metadata={"help": "Maximum token length for query inputs"},
    )
    d_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum token length for document inputs"},
    )


@dataclass
class TrainingArguments(HFTrainingArguments):
    pass


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
    rl_mode: str = field(
        default="dual",
        metadata={"help": "Which GRPO branches to optimize: query_only, listwise_only, or dual"},
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
        metadata={"help": "Learn a global sigma scalar for each GRPO branch"},
    )
    query_reward_type: str = field(
        default="ndcg",
        metadata={"help": "Reward for the query-side branch: ndcg, contrastive, mrr, or mixed"},
    )
    listwise_reward_type: str = field(
        default="ndcg",
        metadata={"help": "Reward for the listwise branch: ndcg, contrastive, mrr, or mixed"},
    )
    query_reward_ndcg_k: int = field(
        default=10,
        metadata={"help": "Ranking cutoff used by the query-side nDCG/MRR reward component"},
    )
    listwise_reward_ndcg_k: int = field(
        default=16,
        metadata={"help": "Ranking cutoff used by the listwise nDCG/MRR reward component"},
    )
    query_mixed_contrastive_weight: float = field(
        default=1.0,
        metadata={"help": "Contrastive reward weight when query_reward_type=mixed"},
    )
    query_mixed_ndcg_weight: float = field(
        default=1.0,
        metadata={"help": "nDCG reward weight when query_reward_type=mixed"},
    )
    listwise_mixed_contrastive_weight: float = field(
        default=1.0,
        metadata={"help": "Contrastive reward weight when listwise_reward_type=mixed"},
    )
    listwise_mixed_ndcg_weight: float = field(
        default=1.0,
        metadata={"help": "nDCG reward weight when listwise_reward_type=mixed"},
    )
    query_contrastive_use_in_batch_negatives: bool = field(
        default=False,
        metadata={"help": "Use positives from other samples as extra negatives for query-side contrastive reward"},
    )
    listwise_contrastive_use_in_batch_negatives: bool = field(
        default=False,
        metadata={"help": "Use positives from other samples as extra negatives for listwise contrastive reward"},
    )
    query_contrastive_negative_aggregation: str = field(
        default="mean",
        metadata={"help": "How to aggregate query-side contrastive negatives: max or mean"},
    )
    listwise_contrastive_negative_aggregation: str = field(
        default="mean",
        metadata={"help": "How to aggregate listwise contrastive negatives: max or mean"},
    )
    listwise_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight applied to the listwise GRPO loss in dual mode"},
    )
    advantage_norm: bool = field(
        default=True,
        metadata={"help": "Normalize GRPO advantages for each sample group"},
    )
    query_relevance_scheme: str = field(
        default="binary",
        metadata={"help": "Relevance labels for the query-side branch: binary or graded"},
    )
    listwise_relevance_scheme: str = field(
        default="graded",
        metadata={"help": "Relevance labels for the listwise branch: binary or graded"},
    )
