import unittest

from aether.model.depth_voice_head import DepthTransformerVoiceHead, DepthTransformerVoiceHeadConfig


class DepthTransformerVoiceHeadSafetyTests(unittest.TestCase):
    def test_is_lazy_until_load(self) -> None:
        head = DepthTransformerVoiceHead(DepthTransformerVoiceHeadConfig(hidden_state_dim=8))
        self.assertFalse(head.loaded)

    def test_methods_require_explicit_load(self) -> None:
        head = DepthTransformerVoiceHead(DepthTransformerVoiceHeadConfig(hidden_state_dim=8))
        with self.assertRaises(RuntimeError):
            head.parameters()
        with self.assertRaises(RuntimeError):
            head.train_mode()
        with self.assertRaises(RuntimeError):
            head.state_dict()
        with self.assertRaises(RuntimeError):
            head.load_state_dict({})
        with self.assertRaises(RuntimeError):
            head.compute_training_loss(None, None)

    def test_rejects_too_few_codebooks(self) -> None:
        with self.assertRaises(ValueError):
            DepthTransformerVoiceHeadConfig(hidden_state_dim=8, num_codebooks=1)

    def test_rejects_non_positive_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            DepthTransformerVoiceHeadConfig(hidden_state_dim=0)
        with self.assertRaises(ValueError):
            DepthTransformerVoiceHeadConfig(hidden_state_dim=8, vocab_size=1)
        with self.assertRaises(ValueError):
            DepthTransformerVoiceHeadConfig(hidden_state_dim=8, max_audio_tokens=0)


if __name__ == "__main__":
    unittest.main()
