import queue
from typing import Any, Final

import numpy as np
import sounddevice as sd

from config import AudioInputConfig, AudioOutputConfig


LIVE_INPUT_SAMPLE_RATE: Final[int] = 16000
LIVE_OUTPUT_SAMPLE_RATE: Final[int] = 24000


class LiveMicrophoneInput:
    def __init__(self, config: AudioInputConfig) -> None:
        self.config = config
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=config.queue_max_chunks)
        self._stream: sd.RawInputStream | None = None

    def _push_chunk(self, chunk: bytes) -> None:
        try:
            self._queue.put_nowait(chunk)
            return
        except queue.Full:
            pass

        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            pass

    def _audio_callback(self, indata: bytes, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        del frames, time_info, status
        self._push_chunk(bytes(indata))

    def start(self) -> None:
        if self._stream is not None:
            return

        self._stream = sd.RawInputStream(
            samplerate=LIVE_INPUT_SAMPLE_RATE,
            blocksize=self.config.block_size,
            device=self.config.device,
            channels=1,
            dtype='int16',
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return

        self._stream.stop()
        self._stream.close()
        self._stream = None
        self.clear_queue()

    def read_chunk(self, timeout: float = 1.0) -> bytes | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def __enter__(self) -> 'LiveMicrophoneInput':
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.stop()


class LiveAudioOutput:
    def __init__(self, config: AudioOutputConfig) -> None:
        self.config = config
        self.channels = max(1, int(config.channels))
        self._stream: sd.OutputStream | None = None

    def start(self) -> None:
        if self._stream is not None:
            return

        self._stream = sd.OutputStream(
            samplerate=LIVE_OUTPUT_SAMPLE_RATE,
            device=self.config.device,
            channels=self.channels,
            dtype='int16',
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return

        self._stream.stop()
        self._stream.close()
        self._stream = None

    def write_pcm16_mono(self, audio_data: bytes) -> None:
        if self._stream is None or not audio_data:
            return

        samples = np.frombuffer(audio_data, dtype='<i2')
        if samples.size == 0:
            return

        if self.channels > 1:
            output = np.repeat(samples[:, None], self.channels, axis=1)
        else:
            output = samples

        self._stream.write(np.ascontiguousarray(output))

    def __enter__(self) -> 'LiveAudioOutput':
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.stop()
