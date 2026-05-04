import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Final, TypeVar

from config import DeepSeekConfig

try:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
except ImportError:  # pragma: no cover - depends on the local virtual environment
    OpenAI = None

    class APIStatusError(Exception):
        status_code: int | None = None

    class APIConnectionError(Exception):
        pass

    class APITimeoutError(Exception):
        pass


RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({408, 409, 429, 500, 502, 503, 504})
RETRY_BACKOFF_SECONDS: Final[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0)
MIN_RETRY_DELAY_SECONDS: Final[float] = 0.5
MAX_RETRY_DELAY_SECONDS: Final[float] = 120.0
ResponseT = TypeVar('ResponseT')
logger = logging.getLogger(__name__)


class DeepSeekChatClient:
    def __init__(self, config: DeepSeekConfig) -> None:
        if OpenAI is None:
            raise RuntimeError("DeepSeek provider requires the 'openai' package. Run: python -m pip install openai")

        self.config = config
        self.client = OpenAI(
            api_key=config.api_key or 'missing-deepseek-api-key',
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=0,
        )
        self._warmed_up = False
        self._warmup_attempted = False
        self._last_warmup_error: str | None = None

    @property
    def last_warmup_error(self) -> str | None:
        return self._last_warmup_error

    def _validate_api_key(self) -> None:
        if not self.config.api_key:
            raise RuntimeError('DEEPSEEK_API_KEY is not configured.')

    def _retry_after_seconds(self, exc: APIStatusError) -> float | None:
        response = getattr(exc, 'response', None)
        retry_after = response.headers.get('retry-after') if response is not None else None
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

    def _retry_delay_seconds(self, attempt_index: int, exc: APIStatusError | None = None) -> float:
        if exc is not None:
            retry_after_seconds = self._retry_after_seconds(exc)
            if retry_after_seconds is not None:
                return retry_after_seconds

        return RETRY_BACKOFF_SECONDS[attempt_index]

    def _max_status_retries(self, exc: APIStatusError) -> int:
        if exc.status_code == 429:
            return max(0, min(self.config.rate_limit_retries, len(RETRY_BACKOFF_SECONDS)))
        return max(0, min(self.config.retry_attempts, len(RETRY_BACKOFF_SECONDS)))

    def _rate_limit_message(self, request_name: str, exc: APIStatusError) -> str:
        provider_hint = ''
        if 'openrouter.ai' in self.config.base_url.lower() and self.config.model.endswith(':free'):
            provider_hint = (
                ' OpenRouter free models can be unavailable under load; wait a bit, choose another model, '
                'or use the non-free model id without ":free".'
            )

        return (
            f"Provider rate limit during {request_name} for model '{self.config.model}' "
            f"(HTTP {exc.status_code}).{provider_hint}"
        )

    def _call_with_retry(self, request_name: str, request_fn) -> ResponseT:
        last_error: Exception | None = None

        for attempt_index in range(len(RETRY_BACKOFF_SECONDS) + 1):
            try:
                self._validate_api_key()
                return request_fn()
            except APIStatusError as exc:
                last_error = exc
                is_retryable = exc.status_code in RETRYABLE_STATUS_CODES
                has_more_attempts = attempt_index < self._max_status_retries(exc)
                if not is_retryable or not has_more_attempts:
                    if exc.status_code == 429:
                        raise RuntimeError(self._rate_limit_message(request_name, exc)) from exc
                    raise RuntimeError(
                        f"DeepSeek model '{self.config.model}' is unavailable during {request_name}: {exc}"
                    ) from exc
                delay_seconds = self._retry_delay_seconds(attempt_index, exc)
                logger.info(
                    "DeepSeek %s retry after %.1f seconds because provider returned HTTP %s.",
                    request_name,
                    delay_seconds,
                    exc.status_code,
                )
                time.sleep(delay_seconds)
            except (APIConnectionError, APITimeoutError) as exc:
                last_error = exc
                max_retries = max(0, min(self.config.retry_attempts, len(RETRY_BACKOFF_SECONDS)))
                has_more_attempts = attempt_index < max_retries
                if not has_more_attempts:
                    raise RuntimeError(
                        f"DeepSeek request failed during {request_name}: {exc}"
                    ) from exc
                delay_seconds = self._retry_delay_seconds(attempt_index)
                logger.info("DeepSeek %s retry after %.1f seconds: %s", request_name, delay_seconds, exc)
                time.sleep(delay_seconds)

        raise RuntimeError(
            f"DeepSeek model '{self.config.model}' did not return a response during {request_name}."
        ) from last_error

    def _chat_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        return self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
            stream=False,
        )

    def _extract_content(self, response) -> str:
        if not response.choices:
            return ''
        content = response.choices[0].message.content
        return content.strip() if content else ''

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
            logger.warning("DeepSeek warm-up failed: %s", exc)
            return False

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.warm_up()
        response = self._call_with_retry('chat', lambda: self._chat_once(messages))
        return self._extract_content(response)
