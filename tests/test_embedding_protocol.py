import pathlib
import sys
import unittest

import torch
from transformers import BertConfig, BertModel


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embedding_protocol import pool_embeddings
from config import RLArguments
from grpo import GRPOModel


class PoolEmbeddingsTest(unittest.TestCase):
    def setUp(self):
        self.hidden = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 1.0], [99.0, 99.0]],
                [[3.0, 1.0], [4.0, 2.0], [5.0, 3.0]],
            ],
            requires_grad=True,
        )
        self.mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    def test_pooling_methods_select_the_checkpoint_native_representation(self):
        last = pool_embeddings(
            self.hidden, self.mask, pooling_method="last", normalize=False
        )
        mean = pool_embeddings(
            self.hidden, self.mask, pooling_method="mean", normalize=False
        )
        cls = pool_embeddings(
            self.hidden, self.mask, pooling_method="cls", normalize=False
        )

        torch.testing.assert_close(last, torch.tensor([[2.0, 1.0], [5.0, 3.0]]))
        torch.testing.assert_close(mean, torch.tensor([[1.5, 0.5], [4.0, 2.0]]))
        torch.testing.assert_close(cls, torch.tensor([[1.0, 0.0], [3.0, 1.0]]))

    def test_pooling_remains_differentiable_and_normalized(self):
        embedding = pool_embeddings(
            self.hidden, self.mask, pooling_method="mean", normalize=True
        )
        torch.testing.assert_close(
            embedding.norm(dim=-1),
            torch.ones(embedding.size(0)),
        )
        embedding.sum().backward()
        self.assertIsNotNone(self.hidden.grad)
        self.assertTrue(torch.isfinite(self.hidden.grad).all())


class EncoderBackboneGRPOTest(unittest.TestCase):
    def test_encoder_model_runs_rl_forward_and_backward_with_mean_pooling(self):
        backbone = BertModel(
            BertConfig(
                vocab_size=32,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
            )
        )
        model = GRPOModel(
            backbone,
            RLArguments(
                action_components="query",
                group_size=2,
                sigma=0.1,
                reward_type="ndcg",
            ),
            pooling_method="mean",
        )

        def inputs(rows, mask):
            return {
                "input_ids": torch.randint(0, 32, (rows, 4)),
                "attention_mask": torch.tensor(mask),
            }

        output = model(
            query=inputs(2, [[1, 1, 1, 0], [1, 1, 1, 1]]),
            positive_document=inputs(2, [[1, 1, 1, 0], [1, 1, 1, 1]]),
            negative_document=inputs(
                4,
                [[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0]],
            ),
            relevance_labels=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )

        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in backbone.parameters()))


if __name__ == "__main__":
    unittest.main()
