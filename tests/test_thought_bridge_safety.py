import unittest

from aether.model.thought_bridge import ThoughtBridge, ThoughtBridgeConfig


class ThoughtBridgeSafetyTests(unittest.TestCase):
    def test_is_lazy_until_load(self) -> None:
        bridge = ThoughtBridge(ThoughtBridgeConfig(hidden_state_dim=8, embedding_dim=16))
        self.assertFalse(bridge.loaded)

    def test_project_requires_explicit_load(self) -> None:
        bridge = ThoughtBridge(ThoughtBridgeConfig(hidden_state_dim=8, embedding_dim=16))
        with self.assertRaises(RuntimeError):
            bridge.project([0.0] * 8)

    def test_rejects_non_positive_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            ThoughtBridgeConfig(hidden_state_dim=0, embedding_dim=16)
        with self.assertRaises(ValueError):
            ThoughtBridgeConfig(hidden_state_dim=8, embedding_dim=0)
        with self.assertRaises(ValueError):
            ThoughtBridgeConfig(hidden_state_dim=8, embedding_dim=16, num_soft_tokens=0)


if __name__ == "__main__":
    unittest.main()
