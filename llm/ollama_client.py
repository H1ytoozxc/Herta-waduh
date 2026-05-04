import re
import logging
import time
from typing import Final, TypeVar

from ollama import Client, ResponseError

from config import OllamaConfig


THINK_BLOCK_RE: Final[re.Pattern[str]] = re.compile(r'<think>.*?</think>', re.IGNORECASE | re.DOTALL)
BASE_QWEN3_MODEL_PREFIXES: Final[tuple[str, ...]] = ('qwen3:',)
BASE_QWEN3_MODEL_EXACT: Final[tuple[str, ...]] = ('qwen3',)
RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({502, 503, 504})
RETRY_BACKOFF_SECONDS: Final[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 45.0, 60.0)
ResponseT = TypeVar('ResponseT')
logger = logging.getLogger(__name__)


class OllamaChatClient:
    def __init__(self, config: OllamaConfig) -> None:
        self.config = config
        self.client = Client(
            host=config.host,
            timeout=config.timeout_seconds,
        )
        self._warmed_up = False
        self._warmup_attempted = False
        self._last_warmup_error: str | None = None

    @property
    def last_warmup_error(self) -> str | None:
        return self._last_warmup_error

    def _is_base_qwen3_model(self) -> bool:
        model_name = self.config.model.strip().lower()
        return model_name in BASE_QWEN3_MODEL_EXACT or any(
            model_name.startswith(prefix) for prefix in BASE_QWEN3_MODEL_PREFIXES
        )

    def _effective_think_flag(self) -> bool:
        if self._is_base_qwen3_model():
            return True
        return self.config.think

    def _is_model_loaded(self) -> bool:
        try:
            running_models = self.client.ps().models
        except Exception as exc:
            logger.debug("Could not check loaded Ollama models: %s", exc)
            return False

        expected_model = self.config.model.strip().lower()
        for running_model in running_models:
            model_names = (
                getattr(running_model, 'model', None),
                getattr(running_model, 'name', None),
            )
            if any(str(model_name).strip().lower() == expected_model for model_name in model_names if model_name):
                return True
        return False

    def _normalize_response_text(self, text: str) -> str:
        stripped_text = text.strip()
        if not stripped_text:
            return stripped_text

        if stripped_text.startswith('<think>') and '</think>' not in stripped_text:
            return ''

        if '</think>' in stripped_text:
            after_closing_tag = stripped_text.rsplit('</think>', maxsplit=1)[-1].strip()
            if after_closing_tag:
                return after_closing_tag

        without_thinking = THINK_BLOCK_RE.sub('', stripped_text).strip()
        return without_thinking or stripped_text

    def _call_with_retry(self, request_name: str, request_fn) -> ResponseT:
        last_error: ResponseError | None = None

        for attempt_index in range(len(RETRY_BACKOFF_SECONDS) + 1):
            try:
                return request_fn()
            except ResponseError as exc:
                last_error = exc
                is_retryable = exc.status_code in RETRYABLE_STATUS_CODES
                has_more_attempts = attempt_index < len(RETRY_BACKOFF_SECONDS)
                if not is_retryable or not has_more_attempts:
                    raise RuntimeError(
                        f"Ollama model '{self.config.model}' is temporarily unavailable during {request_name}: {exc}"
                    ) from exc
                time.sleep(RETRY_BACKOFF_SECONDS[attempt_index])

        raise RuntimeError(
            f"Ollama model '{self.config.model}' did not return a response during {request_name}."
        ) from last_error

    def _build_options(self, **overrides) -> dict[str, float | int | bool]:
        options: dict[str, float | int | bool] = {
            'temperature': self.config.temperature,
            'num_ctx': self.config.num_ctx,
        }
        if self.config.num_gpu is not None:
            options['num_gpu'] = self.config.num_gpu
        options.update(overrides)
        return options

    def _chat_once(self, messages: list[dict[str, str]]):
        return self.client.chat(
            model=self.config.model,
            messages=messages,
            think=self._effective_think_flag(),
            keep_alive=self.config.keep_alive,
            options=self._build_options(),
        )

    def _warm_up_with_generate(self) -> None:
        self._call_with_retry(
            'warm-up',
            lambda: self.client.generate(
                model=self.config.model,
                prompt='',
                keep_alive=self.config.keep_alive,
                options=self._build_options(num_predict=0),
            ),
        )

    def _warm_up_with_chat(self) -> None:
        self._call_with_retry(
            'warm-up fallback chat',
            lambda: self.client.chat(
                model=self.config.model,
                messages=[
                    {'role': 'user', 'content': 'Привет.'},
                ],
                think=self._effective_think_flag(),
                keep_alive=self.config.keep_alive,
                options=self._build_options(temperature=0.0, num_predict=1),
            ),
        )

    def warm_up(self) -> bool:
        if self._warmed_up:
            return True
        if self._warmup_attempted:
            return False

        self._warmup_attempted = True

        try:
            self._warm_up_with_chat()
            self._warmed_up = True
            self._last_warmup_error = None
            return True
        except RuntimeError as exc:
            self._last_warmup_error = str(exc)
            if self._is_model_loaded():
                self._warmed_up = True
                self._last_warmup_error = None
                logger.info("Ollama model is loaded despite chat warm-up failure; continuing.")
                return True
            logger.warning("Ollama chat warm-up failed: %s", exc)
            return False

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.warm_up()
        response = self._call_with_retry('chat', lambda: self._chat_once(messages))
        return self._normalize_response_text(response.message.content)
