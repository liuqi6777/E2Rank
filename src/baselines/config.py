from dataclasses import dataclass, field


@dataclass
class BaselineArguments:
    baseline_type: str = field(
        default="ranknet",
        metadata={
            "help": "Baseline ranking loss: ranknet, lambdaloss, neuralndcg, approxndcg, softrank, or infonce"
        },
    )
    relevance_scheme: str = field(
        default="graded",
        metadata={"help": "Relevance labels: binary or graded"},
    )
    baseline_ndcg_k: int = field(
        default=10,
        metadata={"help": "Ranking cutoff used by LambdaLoss and training metrics"},
    )
    ranknet_sigma: float = field(
        default=1.0,
        metadata={"help": "Pairwise logistic scaling used by RankNet"},
    )
    lambdaloss_sigma: float = field(
        default=1.0,
        metadata={"help": "Pairwise logistic scaling used by LambdaLoss"},
    )
    approxndcg_alpha: float = field(
        default=10.0,
        metadata={"help": "Smoothness factor used by ApproxNDCG"},
    )
    neuralndcg_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature used by the NeuralNDCG NeuralSort surrogate"},
    )
    softrank_sigma: float = field(
        default=1.0,
        metadata={"help": "Gaussian smoothing scale used by SoftRank"},
    )
    infonce_temperature: float = field(
        default=0.03,
        metadata={"help": "Temperature used by the InfoNCE baseline"},
    )
