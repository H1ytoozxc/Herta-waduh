import math
from collections import deque

import numpy as np
from silero_vad import VADIterator, load_silero_vad

from config import AudioInputConfig, VadConfig


class StreamingVADSegmenter:
    def __init__(self, audio_config: AudioInputConfig, vad_config: VadConfig) -> None:
        self.sample_rate = audio_config.sample_rate
        self.block_size = audio_config.block_size
        self.min_utterance_samples = int(self.sample_rate * vad_config.min_utterance_duration_ms / 1000)
        self.max_utterance_samples = int(self.sample_rate * vad_config.max_utterance_seconds)
        self._expected_block_size = 512 if self.sample_rate == 16000 else 256

        if self.sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD supports only 8000 or 16000 Hz audio")
        if self.block_size != self._expected_block_size:
            raise ValueError(
                f"Silero VAD expects block_size={self._expected_block_size} for sample_rate={self.sample_rate}"
            )

        model = load_silero_vad()
        self.iterator = VADIterator(
            model,
            threshold=vad_config.threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=vad_config.min_silence_duration_ms,
            speech_pad_ms=vad_config.speech_pad_ms,
        )
        preroll_chunks = max(1, math.ceil(vad_config.speech_pad_ms / 1000 * self.sample_rate / self.block_size))
        self._pre_buffer: deque[np.ndarray] = deque(maxlen=preroll_chunks)
        self._current_chunks: list[np.ndarray] = []
        self._current_samples = 0
        self._collecting = False

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray | None:
        audio_chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if audio_chunk.size < self.block_size:
            audio_chunk = np.pad(audio_chunk, (0, self.block_size - audio_chunk.size))
        elif audio_chunk.size > self.block_size:
            audio_chunk = audio_chunk[: self.block_size]

        event = self.iterator(audio_chunk)

        if self._collecting:
            self._current_chunks.append(audio_chunk.copy())
            self._current_samples += audio_chunk.size
        elif event and "start" in event:
            buffered_chunks = list(self._pre_buffer)
            self._current_chunks = [*buffered_chunks, audio_chunk.copy()]
            self._current_samples = sum(part.size for part in self._current_chunks)
            self._collecting = True
            self._pre_buffer.clear()
        else:
            self._pre_buffer.append(audio_chunk.copy())

        if self._collecting and self._current_samples >= self.max_utterance_samples:
            return self._finalize_utterance()

        if event and "end" in event and self._collecting:
            return self._finalize_utterance()

        return None

    def reset(self) -> None:
        self.iterator.reset_states()
        self._pre_buffer.clear()
        self._current_chunks.clear()
        self._current_samples = 0
        self._collecting = False

    def _finalize_utterance(self) -> np.ndarray | None:
        utterance = np.concatenate(self._current_chunks) if self._current_chunks else np.array([], dtype=np.float32)
        self.reset()
        if utterance.size < self.min_utterance_samples:
            return None
        return utterance
