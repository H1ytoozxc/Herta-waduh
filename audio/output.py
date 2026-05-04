import math

import numpy as np
import sounddevice as sd

from config import AudioOutputConfig


class SpeakerOutput:
    def __init__(self, config: AudioOutputConfig) -> None:
        self.config = config

    def _build_tone(self) -> np.ndarray:
        sample_count = max(1, int(self.config.sample_rate * self.config.tone_duration_seconds))
        time_axis = np.arange(sample_count, dtype=np.float32) / float(self.config.sample_rate)
        waveform = self.config.tone_volume * np.sin(2.0 * math.pi * self.config.tone_frequency_hz * time_axis)
        if self.config.channels <= 1:
            return waveform.astype(np.float32)
        return np.repeat(waveform[:, None], self.config.channels, axis=1).astype(np.float32)

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        prepared = np.asarray(audio, dtype=np.float32)
        if prepared.ndim == 1:
            if self.config.channels > 1:
                prepared = np.repeat(prepared[:, None], self.config.channels, axis=1)
            return prepared

        if prepared.ndim != 2:
            raise ValueError('Audio must be a 1D or 2D numpy array.')

        if prepared.shape[1] == self.config.channels:
            return prepared

        if prepared.shape[1] == 1 and self.config.channels > 1:
            return np.repeat(prepared, self.config.channels, axis=1)

        if self.config.channels == 1:
            return prepared[:, 0]

        raise ValueError(
            f'Configured output expects {self.config.channels} channels, got {prepared.shape[1]}.'
        )

    def play_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        prepared = self._prepare_audio(audio)
        sd.play(
            prepared,
            samplerate=sample_rate,
            device=self.config.device,
            blocking=True,
        )
        sd.stop()

    def play_test_tone(self) -> None:
        tone = self._build_tone()
        self.play_audio(tone, self.config.sample_rate)


def list_output_devices() -> list[str]:
    devices = sd.query_devices()
    descriptions: list[str] = []

    for index, device in enumerate(devices):
        if int(device['max_output_channels']) <= 0:
            continue
        default_samplerate = int(device['default_samplerate'])
        descriptions.append(
            f"[{index}] {device['name']} | outputs={int(device['max_output_channels'])} | default_samplerate={default_samplerate}"
        )

    return descriptions


def describe_output_device(device: int | str | None) -> str:
    if device is None:
        default_output_index = sd.default.device[1]
        if default_output_index is None or default_output_index < 0:
            return 'unknown'
        selected = sd.query_devices(default_output_index)
        return f"{default_output_index}: {selected['name']}"

    selected = sd.query_devices(device)
    return f"{selected['index']}: {selected['name']}"
