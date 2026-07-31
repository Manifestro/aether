import unittest

from aether.model.llm_backbone import LLMBackboneConfig, SharedLLMBackbone


class LLMBackboneSafetyTests(unittest.TestCase):
    def test_backbone_is_lazy_and_downloads_are_disabled_by_default(self) -> None:
        backbone = SharedLLMBackbone(LLMBackboneConfig("Qwen/Qwen3-1.7B"))
        self.assertFalse(backbone.loaded)
        self.assertFalse(backbone.config.allow_download)

    def test_stream_requires_explicit_load(self) -> None:
        backbone = SharedLLMBackbone(LLMBackboneConfig("local/model/path"))
        self.assertFalse(backbone.loaded)

    def test_generate_with_soft_prompt_requires_explicit_load(self) -> None:
        backbone = SharedLLMBackbone(LLMBackboneConfig("local/model/path"))
        with self.assertRaises(RuntimeError):
            backbone.generate_with_soft_prompt([{"role": "user", "content": "hi"}])


if __name__ == "__main__":
    unittest.main()
