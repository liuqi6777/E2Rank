"""R2 dynamic-only adapter: independent action RNG without changing static runs."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import train
from config import RLArguments
from fixed_corpus.models import DynamicRetrievalGRPOModel
from rollout_rng import RolloutRNG, validate_rollout_seed


class DynamicRLArguments(RLArguments):
    def __post_init__(self):
        if not self.dynamic_retrieval:
            raise ValueError('This entrypoint requires dynamic_retrieval=true')
        seed = self.rollout_seed
        validate_rollout_seed(seed)
        if seed is None:
            raise ValueError('R2 dynamic retrieval requires an independent rollout seed')
        # The shared dataclass forbids dynamic RNG because the shared model does
        # not implement it. Validate all other constraints, then use our adapter.
        self.rollout_seed = None
        try:
            super().__post_init__()
        finally:
            self.rollout_seed = seed


class SeededDynamicModel(DynamicRetrievalGRPOModel):
    def __init__(self, model, index, rl_args, pooling_method='last'):
        super().__init__(model, index, rl_args, pooling_method)
        head = self.policy
        head.rollout_seed = rl_args.rollout_seed
        head.rollout_rng = RolloutRNG(rl_args.rollout_seed)
        head.gradient_estimator = 'score_function'
        original_sample = head.sample

        def sample(means):
            with head.rollout_rng.draw(means.device, head.exploration.step, head.training):
                return original_sample(means)
        head.sample = sample


def main():
    # Only this fresh training process uses these implementations. The existing
    # static suite, source hashes and checkpoint recovery contracts stay intact.
    train.RLArguments = DynamicRLArguments
    train.DynamicRetrievalGRPOModel = SeededDynamicModel
    try:
        train.main()
    finally:
        train.shutdown_distributed()


if __name__ == '__main__':
    main()
