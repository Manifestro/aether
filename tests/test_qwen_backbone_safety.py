import unittest

from vox.model.qwen_backbone import QwenBackboneConfig, SharedQwenBackbone


class QwenBackboneSafetyTests(unittest.TestCase):
    def test_backbone_is_lazy_and_downloads_are_disabled_by_default(self) -> None:
        backbone = SharedQwenBackbone(QwenBackboneConfig("Qwen/Qwen3-1.7B"))
        self.assertFalse(backbone.loaded)
        self.assertFalse(backbone.config.allow_download)

    def test_stream_requires_explicit_load(self) -> None:
        backbone = SharedQwenBackbone(QwenBackboneConfig("local/model/path"))
        self.assertFalse(backbone.loaded)


if __name__ == "__main__":
    unittest.main()
