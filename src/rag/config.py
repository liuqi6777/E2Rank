from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rag.relevance import RELEVANCE_SCHEMES
from rag.shortlist_rl import GRADIENT_ESTIMATORS


SUPPORTED_RAG_OBJECTIVES = ("infonce", "ranknet", "lambdaloss", "rl", "shortlist_rl")
SUPPORTED_RAG_REWARDS = (
    "mrr",
    "ndcg",
    "answer_f1",
    "answer_em",
    "ndcg_answer_f1",
    "ndcg_answer_em",
)
# Rewards that score generated answers and therefore need a generator client
# on rank 0 (train_rag.py gates the client's construction on this set).
GENERATION_REWARDS = frozenset(
    {"answer_f1", "answer_em", "ndcg_answer_f1", "ndcg_answer_em"}
)


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
    rag_relevance_scheme: str = field(
        default="binary",
        metadata={
            "help": (
                "binary: external qrels only (the historical G3 behaviour). "
                "answer_masked: qrel positives, with answer-bearing non-positives "
                "dropped from the contrastive denominator instead of demoted. "
                "graded: qrel=3, evidence=2, answer-bearing=1 gains for nDCG."
            )
        },
    )
    rag_use_in_batch_candidates: bool = field(
        default=False,
        metadata={"help": "CL-Strong: score every other query's candidates as extra negatives"},
    )
    rag_in_batch_include_negatives: bool = field(
        default=False,
        metadata={"help": "Use each query's full candidate list in the cross-query pool, not just its representative"},
    )
    rag_cross_device_negatives: bool = field(
        default=False,
        metadata={"help": "Gather the cross-query negative pool across all data-parallel ranks"},
    )
    rag_in_batch_pool_size: int = field(
        default=15,
        metadata={"help": "Candidates sampled per other query when rag_in_batch_include_negatives is set"},
    )
    rag_tuning_eval: bool = field(
        default=False,
        metadata={"help": "Re-rank the tuning split at every save point and log answer recall/MRR"},
    )
    rag_tuning_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Cap on tuning queries scored per probe; null uses the whole split"},
    )
    rag_retrieval_probe: bool = field(
        default=False,
        metadata={
            "help": (
                "Run a real full-corpus ANN search over a fixed slice of the "
                "evaluation suite at each save point, reported separately for "
                "in-domain (nq/hotpotqa) and held-out datasets. Unlike "
                "rag_tuning_eval this measures retrieval rather than re-ranking, "
                "which is the axis round 2 regressed on."
            )
        },
    )
    rag_retrieval_probe_per_dataset: int = field(
        default=96,
        metadata={"help": "Probe queries sampled per evaluation dataset"},
    )
    rag_retrieval_probe_k: int = field(
        default=20,
        metadata={"help": "Retrieval depth for the probe; each passage costs a corpus text read"},
    )
    rag_anchor_coef: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight of the E0 anchor penalty, coef * (1 - cos(e_theta(q), e_E0(q))). "
                "0 disables (round-3 recipe). Round 3 measured the best arm at "
                "+0.0214 in-domain / -0.0072 held-out; this bounds the drift that "
                "causes the held-out term. The anchors are precomputed once with "
                "the initial weights and cached on the shared disk."
            )
        },
    )

    def _validate_anchor(self) -> None:
        if self.rag_anchor_coef < 0:
            raise ValueError("rag_anchor_coef must be non-negative")

    def __post_init__(self) -> None:
        self.rag_objective = self.rag_objective.strip().lower()
        self.rag_relevance_scheme = self.rag_relevance_scheme.strip().lower()
        if self.rag_relevance_scheme not in RELEVANCE_SCHEMES:
            raise ValueError(
                f"Unsupported rag_relevance_scheme={self.rag_relevance_scheme!r}; "
                f"expected one of {RELEVANCE_SCHEMES}"
            )
        if self.rag_in_batch_pool_size <= 0:
            raise ValueError("rag_in_batch_pool_size must be positive")
        self._validate_anchor()
        if self.rag_in_batch_include_negatives and not self.rag_use_in_batch_candidates:
            raise ValueError("rag_in_batch_include_negatives requires rag_use_in_batch_candidates")
        # shortlist_rl builds its own cross-query pool, so it reads the flag
        # directly rather than through the contrastive in-batch path.
        if (
            self.rag_cross_device_negatives
            and not self.rag_use_in_batch_candidates
            and self.rag_objective != "shortlist_rl"
        ):
            raise ValueError("rag_cross_device_negatives requires rag_use_in_batch_candidates")
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
    rag_retrieval_reward: str = field(default="mrr")
    rag_retrieval_k: int = field(default=10)
    rag_group_size: int = field(default=32)
    rag_kappa: float = field(default=755.0)
    rag_advantage_normalize: Optional[bool] = field(default=None)
    rag_advantage_norm: str = field(default="none")
    rag_advantage_baseline: str = field(default="leave_one_out")
    rag_advantage_baseline_momentum: float = field(default=0.99)
    rag_target_alignment: Optional[float] = field(default=None)
    rag_final_alignment: Optional[float] = field(default=None)
    rag_exploration_schedule: str = field(default="fixed")
    rag_gradient_estimator: str = field(
        default="score_function",
        metadata={
            "help": (
                "score_function or conditional_projection. CP requires static "
                "candidates (rag_objective=shortlist_rl); under live full-corpus "
                "retrieval the reward depends on documents outside the returned "
                "top-k, which makes the projection biased rather than merely "
                "lower-variance."
            )
        },
    )
    rag_slate_size: int = field(
        default=20,
        metadata={"help": "Own candidates ranked per query under shortlist_rl"},
    )
    rag_shortlist_size: int = field(
        default=15,
        metadata={"help": "Negatives borrowed from other queries per shortlist_rl step"},
    )

    def __post_init__(self) -> None:
        from policy_math import validate_exploration
        self.rag_gradient_estimator = self.rag_gradient_estimator.strip().lower()
        if self.rag_gradient_estimator not in GRADIENT_ESTIMATORS:
            raise ValueError(
                f"Unsupported rag_gradient_estimator={self.rag_gradient_estimator!r}; "
                f"expected one of {GRADIENT_ESTIMATORS}"
            )
        if self.rag_slate_size <= 1:
            raise ValueError("rag_slate_size must exceed one")
        if self.rag_shortlist_size < 0:
            raise ValueError("rag_shortlist_size must be non-negative")
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
    rag_generator_top_k: int = field(default=10)
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
