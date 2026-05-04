import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Final, TypeVar

import httpx

from config import GoogleAIConfig


RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({408, 409, 429, 500, 502, 503, 504})
RETRY_BACKOFF_SECONDS: Final[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0)
MIN_RETRY_DELAY_SECONDS: Final[float] = 0.5
MAX_RETRY_DELAY_SECONDS: Final[float] = 120.0
ResponseT = TypeVar('ResponseT')
logger = logging.getLogger(__name__)


class GoogleAIChatClient:
    def __init__(self, config: GoogleAIConfig) -> None:
        self.config = config
        self.client = httpx.Client(timeout=config.timeout_seconds)
        self._warmed_up = False
        self._warmup_attempted = False
        self._last_warmup_error: str | None = None

    @property
    def last_warmup_error(self) -> str | None:
        return self._last_warmup_error

    def _validate_api_key(self) -> None:
        if not self.config.api_key:
            raise RuntimeError('GOOGLE_AI_API_KEY is not configured.')

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

    def _status_error_message(self, request_name: str, response: httpx.Response) -> str:
        provider_message = ''
        try:
            provider_message = response.json().get('error', {}).get('message', '')
        except ValueError:
            provider_message = response.text[:300]

        provider_suffix = f': {provider_message}' if provider_message else ''
        if response.status_code == 429:
            return (
                f"Google AI Studio rate limit during {request_name} for model '{self.config.model}' "
                f'(HTTP 429){provider_suffix}'
            )

        return (
            f"Google AI Studio model '{self.config.model}' is unavailable during {request_name} "
            f'(HTTP {response.status_code}){provider_suffix}'
        )

    def _call_with_retry(self, request_name: str, request_fn) -> ResponseT:
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
                    raise RuntimeError(self._status_error_message(request_name, response))

                delay_seconds = self._retry_delay_seconds(attempt_index, response)
                logger.info(
                    "Google AI %s retry after %.1f seconds because provider returned HTTP %s.",
                    request_name,
                    delay_seconds,
                    response.status_code,
                )
                time.sleep(delay_seconds)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = exc
                max_retries = max(0, min(self.config.retry_attempts, len(RETRY_BACKOFF_SECONDS)))
                has_more_attempts = attempt_index < max_retries
                if not has_more_attempts:
                    raise RuntimeError(f'Google AI Studio request failed during {request_name}: {exc}') from exc

                delay_seconds = self._retry_delay_seconds(attempt_index)
                logger.info("Google AI %s retry after %.1f seconds: %s", request_name, delay_seconds, exc)
                time.sleep(delay_seconds)

        raise RuntimeError(
            f"Google AI Studio model '{self.config.model}' did not return a response during {request_name}."
        ) from last_error

    def _append_content(self, contents: list[dict], role: str, text: str) -> None:
        if contents and contents[-1]['role'] == role:
            contents[-1]['parts'].append({'text': text})
            return
        contents.append({'role': role, 'parts': [{'text': text}]})

    def _build_payload(self, messages: list[dict[str, str]]) -> dict:
        system_texts: list[str] = []
        contents: list[dict] = []

        for message in messages:
            text = str(message.get('content', '')).strip()
            if not text:
                continue

            role = message.get('role', 'user')
            if role == 'system':
                system_texts.append(text)
            elif role == 'assistant':
                self._append_content(contents, 'model', text)
            else:
                self._append_content(contents, 'user', text)

        if not contents:
            self._append_content(contents, 'user', 'ping')

        instruction_text = '\n\n'.join(system_texts)
        if instruction_text and not self.config.system_instruction_enabled:
            instruction_prefix = (
                'Instructions for the assistant. Follow them throughout this conversation:\n'
                f'{instruction_text}\n\nConversation:'
            )
            if contents[0]['role'] == 'user':
                contents[0]['parts'].insert(0, {'text': instruction_prefix})
            else:
                contents.insert(0, {'role': 'user', 'parts': [{'text': instruction_prefix}]})

        payload = {
            'contents': contents,
            'generationConfig': {
                'temperature': self.config.temperature,
                'maxOutputTokens': self.config.max_tokens,
            },
        }
        if instruction_text and self.config.system_instruction_enabled:
            payload['system_instruction'] = {'parts': [{'text': instruction_text}]}

        return payload

    def _generate_once(self, messages: list[dict[str, str]]) -> httpx.Response:
        return self.client.post(
            self._endpoint_url(),
            headers={
                'Content-Type': 'application/json',
                'x-goog-api-key': self.config.api_key or '',
            },
            json=self._build_payload(messages),
        )

    def _extract_content(self, response: httpx.Response) -> str:
        data = response.json()
        candidates = data.get('candidates') or []
        if not candidates:
            prompt_feedback = data.get('promptFeedback') or {}
            raise RuntimeError(f"Google AI Studio returned no candidates: {prompt_feedback}")

        parts = candidates[0].get('content', {}).get('parts') or []
        text = ''.join(str(part.get('text', '')) for part in parts).strip()
        if not text:
            finish_reason = candidates[0].get('finishReason', 'unknown')
            raise RuntimeError(f"Google AI Studio returned no text. finish_reason={finish_reason}")
        return text

    def warm_up(self) -> bool:
        if self._warmed_up:
            return True
        if self._warmup_attempted:
            return False

        self._warmup_attempted = True

        try:
            self._validate_api_key()
            self._warmed_up = True
            self._last_warmup_error = None
            return True
        except RuntimeError as exc:
            self._last_warmup_error = str(exc)
            logger.warning("Google AI warm-up failed: %s", exc)
            return False

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.warm_up()
        response = self._call_with_retry('chat', lambda: self._generate_once(messages))
        return self._extract_content(response)
