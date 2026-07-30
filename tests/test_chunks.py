import unittest

from vox.domain.chunks import ChunkState, SpeechChunk


class SpeechChunkTests(unittest.TestCase):
    def test_dependency_free_chunk_is_ready_immediately(self) -> None:
        chunk = SpeechChunk("lead-in", "Подтвердить запрос")
        self.assertEqual(chunk.state, ChunkState.READY)

    def test_dependent_chunk_unblocks_only_after_fact(self) -> None:
        chunk = SpeechChunk(
            "answer",
            "Сообщить погоду",
            dependencies=frozenset({"weather"}),
        )
        self.assertFalse(chunk.resolve(frozenset()))
        self.assertTrue(chunk.resolve(frozenset({"weather"})))

    def test_committed_chunk_cannot_be_cancelled(self) -> None:
        chunk = SpeechChunk("lead-in", "Подтвердить запрос")
        chunk.transition_to(ChunkState.GENERATING)
        chunk.transition_to(ChunkState.BUFFERED)
        chunk.transition_to(ChunkState.COMMITTED)
        with self.assertRaises(ValueError):
            chunk.transition_to(ChunkState.CANCELLED)


if __name__ == "__main__":
    unittest.main()

