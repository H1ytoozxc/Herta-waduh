import logging

import numpy as np
from faster_whisper import WhisperModel

from config import WhisperSTTConfig


LOGGER = logging.getLogger("the_herta.stt")
CUDA_ERROR_MARKERS = (
    "cublas",
    "cudnn",
    "cuda",
    "cudart",
)


class FasterWhisperSTT:
    def __init__(self, config: WhisperSTTConfig) -> None:
        self.config = config
        self.active_device = config.device
        self.model = self._load_model(self.active_device)

    def _load_model(self, device: str) -> WhisperModel:
        return WhisperModel(
            self.config.model_size,
            device=device,
            compute_type=self.config.compute_type,
            cpu_threads=self.config.cpu_threads,
            num_workers=self.config.num_workers,
            download_root=self.config.download_root,
            local_files_only=self.config.local_files_only,
        )

    def _should_fallback_to_cpu(self, exc: Exception) -> bool:
        if self.active_device == "cpu":
            return False
        normalized = str(exc).lower()
        return any(marker in normalized for marker in CUDA_ERROR_MARKERS)

    def _switch_to_cpu(self) -> None:
        LOGGER.warning(
            "Whisper backend on device '%s' is unavailable. Falling back to CPU.",
            self.active_device,
        )
        self.active_device = "cpu"
        self.model = self._load_model("cpu")

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            return waveform

        waveform = waveform - float(np.mean(waveform))
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0

        if peak < self.config.min_peak_level or rms < self.config.min_rms_level:
            LOGGER.debug(
                "Skipping low-energy chunk before STT. peak=%.6f rms=%.6f",
                peak,
                rms,
            )
            return np.array([], dtype=np.float32)

        if self.config.normalize_audio and peak > 0.0:
            waveform = 0.95 * waveform / peak

        return waveform.astype(np.float32, copy=False)

    def _transcribe_with_model(self, waveform: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            waveform,
            language=self.config.language,
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
            word_timestamps=False,
            initial_prompt=self.config.initial_prompt,
            no_speech_threshold=self.config.no_speech_threshold,
            log_prob_threshold=self.config.log_prob_threshold,
            compression_ratio_threshold=self.config.compression_ratio_threshold,
        )
        parts = [segment.text.strip() for segment in segments if segment.text.strip()]
        return " ".join(parts).strip()

    def transcribe(self, audio: np.ndarray) -> str:
        waveform = self._prepare_audio(audio)
        if waveform.size == 0:
            return ""

        try:
            return self._transcribe_with_model(waveform)
        except Exception as exc:
            if not self._should_fallback_to_cpu(exc):
                raise
            self._switch_to_cpu()
            return self._transcribe_with_model(waveform)
