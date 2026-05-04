import json
import logging
from pathlib import Path
from typing import Any

from config import MemoryConfig


logger = logging.getLogger(__name__)
VALID_ROLES = {'user', 'assistant'}


class DialogueMemory:
    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.path = self._resolve_path(config.path)

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        return Path.cwd() / candidate

    def _read_payload(self) -> dict[str, Any]:
        if not self.path.exists():
            return {'version': 1, 'messages': []}

        try:
            with self.path.open('r', encoding='utf-8') as file:
                payload = json.load(file)
        except Exception as exc:
            logger.warning("Failed to read dialogue memory from %s: %s", self.path, exc)
            return {'version': 1, 'messages': []}

        if not isinstance(payload, dict):
            return {'version': 1, 'messages': []}
        return payload

    def _sanitize_messages(self, raw_messages: Any) -> list[dict[str, str]]:
        if not isinstance(raw_messages, list):
            return []

        messages: list[dict[str, str]] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                continue

            role = item.get('role')
            content = item.get('content')
            if role not in VALID_ROLES or not isinstance(content, str):
                continue

            stripped_content = content.strip()
            if stripped_content:
                messages.append({'role': role, 'content': stripped_content})

        return messages

    def _write_messages(self, messages: list[dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'version': 1,
            'messages': messages[-self.config.max_messages:] if self.config.max_messages > 0 else [],
        }

        with self.path.open('w', encoding='utf-8') as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write('\n')

    def load_context_messages(self) -> list[dict[str, str]]:
        if self.config.context_messages <= 0:
            return []

        payload = self._read_payload()
        messages = self._sanitize_messages(payload.get('messages'))
        context_messages = messages[-self.config.context_messages:]
        while context_messages and context_messages[0]['role'] != 'user':
            context_messages = context_messages[1:]
        return context_messages

    def append_turn(self, user_text: str, assistant_text: str) -> None:
        user_text = user_text.strip()
        assistant_text = assistant_text.strip()
        if not user_text or not assistant_text:
            return

        payload = self._read_payload()
        messages = self._sanitize_messages(payload.get('messages'))
        messages.extend([
            {'role': 'user', 'content': user_text},
            {'role': 'assistant', 'content': assistant_text},
        ])
        self._write_messages(messages)
