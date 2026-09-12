from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


SUPPORTED_RAG_OBJECTIVES = ("infonce", "ranknet", "rl")
SUPPORTED_RAG_REWARDS = ("source_aware_mrr", "answer_mrr", "answer_f1")


@dataclass
class RAGDatasetArguments:
    """FlashRAG inputs and the immutable candidate manifest used for training."""

    rag_dataset_root: str = field(
        metadata={"help": "Directory containing FlashRAG dataset subdirectories"}
    )
    rag_corpus_path: str = field(
        metadata={"help": "FlashRAG wiki18_100w JSONL corpus path"}
    )
    rag_candidate_manifest: Optional[str] = field(
        default=None,
        metadata={"help": "JSONL produced by mine_rag_candidates.py"},
    )
    rag_train_sources: str = field(default="nq,hotpotqa")
    rag_tuning_fraction: float = field(default=0.05)
    rag_tuning_seed: int = field(default=20260909)
    rag_split: str = field(default="train", metadata={"help": "train, tuning, or full"})
    rag_objective: str = field(default="rl")
    rag_temperature: float = field(default=0.03)
    rag_candidate_depth: int = field(default=1000)
    rag_query_max_length: int = field(default=128)
    rag_max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Optional smoke-test cap; null keeps the full selected split"}
    )

    def __post_init__(self) -> None:
        self.rag_objective = self.rag_objective.strip().lower()
        if self.rag_objective not in SUPPORTED_RAG_OBJECTIVES:
            raise ValueError(
                f"Unsupported rag_objective={self.rag_objective!r}; "
                f"expected one of {SUPPORTED_RAG_OBJECTIVES}"
            )
        if self.rag_split not in {"train", "tuning", "full"}:
            raise ValueError("rag_split must be train, tuning, or full")
        if not 0.0 < self.rag_tuning_fraction < 1.0:
            raise ValueError("rag_tuning_fraction must be between 0 and 1")
        if self.rag_candidate_depth <= 0:
            raise ValueError("rag_candidate_depth must be positive")
        if self.rag_temperature <= 0:
            raise ValueError("rag_temperature must be positive")
        if self.rag_max_train_samples is not None and self.rag_max_train_samples <= 0:
            raise ValueError("rag_max_train_samples must be positive or null")

    @property
    def train_sources(self) -> tuple[str, ...]:
        return tuple(source.strip().lower() for source in self.rag_train_sources.split(",") if source.strip())


@dataclass
class RAGIndexArguments:
    """Location and runtime layout of an immutable corpus-vector index."""

    rag_index_manifest: str = field(
        metadata={"help": "JSON manifest written by encode_rag_corpus.py"}
    )
    rag_index_backend: str = field(default="faiss", metadata={"help": "faiss or torch"})
    rag_index_device: str = field(default="cuda")
    rag_index_shard_assignment: str = field(default="round_robin")
    rag_index_search_batch_size: int = field(default=1024)
    rag_index_verify_hashes: bool = field(default=True)

    def __post_init__(self) -> None:
        self.rag_index_backend = self.rag_index_backend.strip().lower()
        if self.rag_index_backend not in {"faiss", "torch"}:
            raise ValueError("rag_index_backend must be faiss or torch")
        if self.rag_index_shard_assignment != "round_robin":
            raise ValueError("Only deterministic round_robin shard assignment is supported")
        if self.rag_index_search_batch_size <= 0:
            raise ValueError("rag_index_search_batch_size must be positive")


@dataclass
class RAGRewardArguments:
    rag_retrieval_reward: str = field(default="source_aware_mrr")
    rag_retrieval_k: int = field(default=20)
    rag_group_size: int = field(default=32)
    rag_kappa: float = field(default=755.0)
    rag_advantage_normalize: Optional[bool] = field(default=None)
    rag_advantage_norm: str = field(default="none")
    rag_advantage_baseline: str = field(default="leave_one_out")
    rag_advantage_baseline_momentum: float = field(default=0.99)
    rag_target_alignment: Optional[float] = field(default=None)
    rag_final_alignment: Optional[float] = field(default=None)
    rag_exploration_schedule: str = field(default="fixed")

    def __post_init__(self) -> None:
        from policy_math import validate_exploration
        validate_exploration(self.rag_target_alignment, self.rag_final_alignment, self.rag_exploration_schedule)
        if self.rag_advantage_norm not in {"none", "shared", "per_component"}:
            raise ValueError("Unsupported RAG advantage normalization")
        if self.rag_advantage_baseline not in {"group", "leave_one_out", "ema"}:
            raise ValueError("Unsupported RAG baseline")
        if not 0 <= self.rag_advantage_baseline_momentum < 1:
            raise ValueError("RAG baseline momentum must lie in [0, 1)")
        self.rag_retrieval_reward = self.rag_retrieval_reward.strip().lower()
        if self.rag_retrieval_reward not in SUPPORTED_RAG_REWARDS:
            raise ValueError(
                f"Unsupported RAG reward {self.rag_retrieval_reward!r}; "
                f"expected one of {SUPPORTED_RAG_REWARDS}"
            )
        if self.rag_retrieval_k <= 0 or self.rag_group_size <= 1 or self.rag_kappa <= 0:
            raise ValueError("retrieval_k and kappa must be positive; group_size must exceed one")


@dataclass
class RAGGeneratorArguments:
    rag_generator_model: str = field(default="Qwen/Qwen2.5-7B-Instruct")
    rag_generator_revision: Optional[str] = field(default=None)
    rag_generator_system_prompt: str = field(
        default=(
            "Answer the question based on the given documents. "
            "Only give me the answer and do not output any other words."
        )
    )
    rag_generator_endpoint: Optional[str] = field(default=None)
    rag_generator_cache: str = field(default="data/rag/generator_cache.sqlite3")
    rag_generator_top_k: int = field(default=5)
    rag_generator_max_input_length: int = field(default=2048)
    rag_generator_max_new_tokens: int = field(default=32)
    rag_generator_timeout_seconds: int = field(default=600)
    rag_generator_gpu_count: int = field(default=1)

    def __post_init__(self) -> None:
        if self.rag_generator_top_k <= 0:
            raise ValueError("rag_generator_top_k must be positive")
        if self.rag_generator_max_input_length <= 0 or self.rag_generator_max_new_tokens <= 0:
            raise ValueError("generator lengths must be positive")
        if self.rag_generator_gpu_count <= 0:
            raise ValueError("rag_generator_gpu_count must be positive")
