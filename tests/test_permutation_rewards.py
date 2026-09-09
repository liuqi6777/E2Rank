import pathlib
import sys
import unittest

import torch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rewards import compute_reward_from_scores, normalize_reward_terms


class PermutationRewardTests(unittest.TestCase):
    def setUp(self):
        self.relevance = torch.zeros(1, 4)
        self.ranks = torch.tensor([[4.0, 3.0, 2.0, 1.0]])

    def reward(self, scores, reward_type, **kwargs):
        return compute_reward_from_scores(
            torch.as_tensor(scores, dtype=torch.float32),
            self.relevance,
            rank_labels=self.ranks,
            reward_type=reward_type,
            **kwargs,
        )

    def test_perfect_permutation_maximizes_both_rewards(self):
        perfect = [[4.0, 3.0, 2.0, 1.0]]
        for reward_type in ("top_weighted_pairwise", "rbo"):
            reward = self.reward(perfect, reward_type, k=4)
            torch.testing.assert_close(reward, torch.ones_like(reward))

    def test_pairwise_reverse_and_ties_have_expected_credit(self):
        reverse = self.reward([[1.0, 2.0, 3.0, 4.0]], "top_weighted_pairwise", k=4)
        ties = self.reward([[0.0, 0.0, 0.0, 0.0]], "top_weighted_pairwise", k=4)
        torch.testing.assert_close(reverse, torch.zeros_like(reverse))
        torch.testing.assert_close(ties, torch.full_like(ties, 0.5))

    def test_cutoff_ignores_tail_only_exchange(self):
        perfect_and_tail_swap = torch.tensor(
            [[[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 1.0, 2.0]]]
        )
        for reward_type in ("top_weighted_pairwise", "rbo"):
            reward = self.reward(perfect_and_tail_swap, reward_type, k=2)
            torch.testing.assert_close(reward, torch.ones_like(reward))

    def test_rbo_top_one_is_controlled_by_zero_persistence(self):
        top_swap = [[3.0, 4.0, 2.0, 1.0]]
        reward = self.reward(top_swap, "rbo", k=4, rbo_p=0.0)
        torch.testing.assert_close(reward, torch.zeros_like(reward))

    def test_reward_term_parser_accepts_rbo_p_alias(self):
        term = normalize_reward_terms("rbo:1.0,k=10,p=0.85")[0]
        self.assertEqual(term.type, "rbo")
        self.assertEqual(term.k, 10)
        self.assertEqual(term.rbo_p, 0.85)

    def test_permutation_rewards_require_rank_labels(self):
        with self.assertRaisesRegex(ValueError, "rank_labels are required"):
            compute_reward_from_scores(
                torch.tensor([[4.0, 3.0, 2.0, 1.0]]),
                self.relevance,
                reward_type="rbo",
            )


if __name__ == "__main__":
    unittest.main()
