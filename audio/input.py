import queue
from typing import Any

import numpy as np
import sounddevice as sd

from config import AudioInputConfig


class MicrophoneInput:
    def __init__(self, config: AudioInputConfig) -> None:
        self.config = config
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=config.queue_max_chunks)
        self._stream: sd.InputStream | None = None

    def _push_chunk(self, chunk: np.ndarray) -> None:
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

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        del frames, time_info, status
        chunk = np.asarray(indata[:, 0], dtype=np.float32).copy()
        self._push_chunk(chunk)

    def start(self) -> None:
        if self._stream is not None:
            return

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.block_size,
            device=self.config.device,
            channels=self.config.channels,
            dtype=self.config.dtype,
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

    def read_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
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

    def __enter__(self) -> "MicrophoneInput":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.stop()



def list_input_devices() -> list[str]:
    devices = sd.query_devices()
    descriptions: list[str] = []

    for index, device in enumerate(devices):
        if int(device["max_input_channels"]) <= 0:
            continue
        default_samplerate = int(device["default_samplerate"])
        descriptions.append(
            f"[{index}] {device['name']} | inputs={int(device['max_input_channels'])} | default_samplerate={default_samplerate}"
        )

    return descriptions
