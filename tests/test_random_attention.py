import unittest

import torch

from vggt.layers.attention import Attention


def make_attention(selector="random", **kwargs):
    config = {
        "causal_online_controller": False,
    }
    config.update(kwargs)
    return Attention(
        dim=16,
        num_heads=4,
        patch_width=2,
        patch_height=2,
        method="random",
        rand_token_select=selector,
        **config,
    ).eval()


class RandomAttentionTest(unittest.TestCase):
    def test_random_method_preserves_output_shape(self):
        for selector in ("random", "uniform"):
            with self.subTest(selector=selector):
                attention = make_attention(selector)
                tokens = torch.randn(2, 27, 16)
                output = attention(
                    tokens, global_merging=0, attn_type="global", block_id=0
                )
                self.assertEqual(output.shape, tokens.shape)
                self.assertTrue(torch.isfinite(output).all())

    def test_random_selection_is_deterministic(self):
        attention = make_attention(
            "random",
            random_seed=7,
            causal_online_controller=True,
            causal_controller_enter_threshold=float("inf"),
        )
        tokens = torch.randn(1, 27, 16)
        first = attention(tokens, global_merging=3, attn_type="global", block_id=3)
        second = attention(tokens, global_merging=3, attn_type="global", block_id=3)
        torch.testing.assert_close(first, second)

    def test_uniform_selection_keeps_first_view_and_spreads_later_tokens(self):
        attention = make_attention("uniform", keep_ratio=0.25)
        indices = attention._sample_per_view(
            num_keys=27,
            keep_ratio=0.25,
            device=torch.device("cpu"),
            batch_shape=(1,),
            block_id=0,
            group_id=0,
        )
        self.assertEqual(indices.shape, (1, 13))
        torch.testing.assert_close(indices[0, :9], torch.arange(9))
        self.assertTrue(((indices[0, 9:11] >= 9) & (indices[0, 9:11] < 18)).all())
        self.assertTrue(((indices[0, 11:] >= 18) & (indices[0, 11:] < 27)).all())

    def test_invalid_selector_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rand_token_select"):
            make_attention("unsupported")


if __name__ == "__main__":
    unittest.main()
