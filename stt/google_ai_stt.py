import base64
import io
import logging
import time
import wave
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Final, TypeVar

import httpx
import numpy as np

from config import GoogleSTTConfig, WhisperSTTConfig


RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({408, 409, 429, 500, 502, 503, 504})
RETRY_BACKOFF_SECONDS: Final[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0, 12.0)
MIN_RETRY_DELAY_SECONDS: Final[float] = 0.5
MAX_RETRY_DELAY_SECONDS: Final[float] = 60.0
NO_SPEECH_MARKERS: Final[tuple[str, ...]] = (
    'нет речи',
    'речь отсутствует',
    'тишина',
    'no speech',
    'silent',
)
ResponseT = TypeVar('ResponseT')
logger = logging.getLogger(__name__)


class GoogleAITranscriptionSTT:
    active_device = 'google_ai'

    def __init__(
        self,
        config: GoogleSTTConfig,
        audio_gate_config: WhisperSTTConfig,
        sample_rate: int,
    ) -> None:
        self.config = config
        self.audio_gate_config = audio_gate_config
        self.sample_rate = sample_rate
        self.client = httpx.Client(timeout=config.timeout_seconds)

    def _validate_api_key(self) -> None:
        if not self.config.api_key:
            raise RuntimeError('GOOGLE_STT_API_KEY or GOOGLE_AI_API_KEY is not configured.')

    def _endpoint_url(self) -> str:
        base_url = self.config.base_url.rstrip('/')
        return f'{base_url}/models/{self.config.model}:generateContent'

    def _retry_after_seconds(self, response: httpx.Response) -> float | None:
        retry_after = response.headers.get('retry-after')
        if not retry_after:
            return None

        try:
            delay_seconds = float(retry_after)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
            except (TypeError, ValueError, IndexError, OverflowError):
                return None
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            delay_seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()

        return max(MIN_RETRY_DELAY_SECONDS, min(delay_seconds, MAX_RETRY_DELAY_SECONDS))

    def _retry_delay_seconds(self, attempt_index: int, response: httpx.Response | None = None) -> float:
        if response is not None:
            retry_after_seconds = self._retry_after_seconds(response)
            if retry_after_seconds is not None:
                return retry_after_seconds

        return RETRY_BACKOFF_SECONDS[attempt_index]

    def _max_status_retries(self, response: httpx.Response) -> int:
        if response.status_code == 429:
            return max(0, min(self.config.rate_limit_retries, len(RETRY_BACKOFF_SECONDS)))
        return max(0, min(self.config.retry_attempts, len(RETRY_BACKOFF_SECONDS)))

    def _status_error_message(self, response: httpx.Response) -> str:
        provider_message = ''
        try:
            provider_message = response.json().get('error', {}).get('message', '')
        except ValueError:
            provider_message = response.text[:300]

        provider_suffix = f': {provider_message}' if provider_message else ''
        return (
            f"Google AI STT model '{self.config.model}' is unavailable "
            f'(HTTP {response.status_code}){provider_suffix}'
        )

    def _call_with_retry(self, request_fn) -> ResponseT:
        last_error: Exception | None = None

        for attempt_index in range(len(RETRY_BACKOFF_SECONDS) + 1):
            try:
                self._validate_api_key()
                response = request_fn()
                if response.status_code < 400:
                    return response

                is_retryable = response.status_code in RETRYABLE_STATUS_CODES
                has_more_attempts = attempt_index < self._max_status_retries(response)
                if not is_retryable or not has_more_attempts:
                    raise RuntimeError(self._status_error_message(response))

                delay_seconds = self._retry_delay_seconds(attempt_index, response)
                logger.info(
                    'Google AI STT retry after %.1f seconds because provider returned HTTP %s.',
                    delay_seconds,
                    response.status_code,
                )
                time.sleep(delay_seconds)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = exc
                max_retries = max(0, min(self.config.retry_attempts, len(RETRY_BACKOFF_SECONDS)))
                has_more_attempts = attempt_index < max_retries
                if not has_more_attempts:
                    raise RuntimeError(f'Google AI STT request failed: {exc}') from exc

                delay_seconds = self._retry_delay_seconds(attempt_index)
                logger.info('Google AI STT retry after %.1f seconds: %s', delay_seconds, exc)
                time.sleep(delay_seconds)

        raise RuntimeError(f"Google AI STT model '{self.config.model}' did not return a response.") from last_error

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            return waveform

        waveform = waveform - float(np.mean(waveform))
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0

        if peak < self.audio_gate_config.min_peak_level or rms < self.audio_gate_config.min_rms_level:
            logger.debug('Skipping low-energy chunk before Google STT. peak=%.6f rms=%.6f', peak, rms)
            return np.array([], dtype=np.float32)

        if self.audio_gate_config.normalize_audio and peak > 0.0:
            waveform = 0.95 * waveform / peak

        return waveform.astype(np.float32, copy=False)

    def _wav_bytes(self, waveform: np.ndarray) -> bytes:
        audio_i16 = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype('<i2')
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_i16.tobytes())
        return buffer.getvalue()

    def _prompt(self) -> str:
        language_hint = self.config.language_hint
        language_line = f'Expected language: {language_hint}.' if language_hint else 'Detect the spoken language.'
        return (
            'Transcribe the speech in this audio clip. '
            f'{language_line} '
            'Return only the transcript text, without quotes, markdown, comments, timestamps, speaker labels, '
            'translations, summaries, or explanations. If there is no clear speech, return an empty string.'
        )

    def _build_payload(self, waveform: np.ndarray) -> dict:
        wav_data = self._wav_bytes(waveform)
        encoded_audio = base64.b64encode(wav_data).decode('ascii')
        return {
            'contents': [
                {
                    'role': 'user',
                    'parts': [
                        {'text': self._prompt()},
                        {
                            'inline_data': {
                                'mime_type': 'audio/wav',
                                'data': encoded_audio,
                            },
                        },
                    ],
                },
            ],
            'generationConfig': {
                'temperature': 0.0,
                'maxOutputTokens': 120,
            },
        }

    def _generate_once(self, waveform: np.ndarray) -> httpx.Response:
        return self.client.post(
            self._endpoint_url(),
            headers={
                'Content-Type': 'application/json',
                'x-goog-api-key': self.config.api_key or '',
            },
            json=self._build_payload(waveform),
        )

    def _extract_transcript(self, response: httpx.Response) -> str:
        data = response.json()
        candidates = data.get('candidates') or []
        if not candidates:
            prompt_feedback = data.get('promptFeedback') or {}
            raise RuntimeError(f'Google AI STT returned no candidates: {prompt_feedback}')

        parts = candidates[0].get('content', {}).get('parts') or []
        transcript = ''.join(str(part.get('text', '')) for part in parts).strip()
        transcript = transcript.strip(' "\'`')

        normalized = transcript.lower().strip(' .,!?:;"\'`')
        if normalized in NO_SPEECH_MARKERS or any(marker == normalized for marker in NO_SPEECH_MARKERS):
            return ''
        return transcript

    def transcribe(self, audio: np.ndarray) -> str:
        waveform = self._prepare_audio(audio)
        if waveform.size == 0:
            return ''

        response = self._call_with_retry(lambda: self._generate_once(waveform))
        return self._extract_transcript(response)
