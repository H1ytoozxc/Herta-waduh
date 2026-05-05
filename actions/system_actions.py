from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

from config import SystemActionsConfig


URL_RE = re.compile(r'\b(?:https?://[^\s]+|www\.[^\s]+|[a-z0-9-]+\.[a-z]{2,}(?:/[^\s]*)?)', re.IGNORECASE)
INVALID_PATH_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
DESTRUCTIVE_RE = re.compile(
    r'\b('
    r'удали|удалить|удаляй|сотри|стереть|стирай|очисти|снеси|уничтожь|'
    r'перемести|замени|перезапиши|форматируй|'
    r'delete|remove|erase|rm|rmdir|del|format|move|overwrite'
    r')\b',
    re.IGNORECASE,
)

OPEN_WORDS = ('открой', 'открыть', 'запусти', 'запустить')
CREATE_WORDS = ('создай', 'создать', 'сделай', 'заведи')
WRITE_WORDS = ('запиши', 'допиши', 'добавь', 'внеси', 'заполни', 'напиши')
RENAME_WORDS = ('переименуй', 'переименовать')
SEARCH_MARKERS = ('загугли', 'найди в интернете', 'найди в браузере', 'поиск в браузере')
VAGUE_NAME_PATTERNS = (
    'как нибудь',
    'как-нибудь',
    'что нибудь',
    'что-нибудь',
    'нибудь',
    'как хочешь',
    'хочешь',
    'как угодно',
    'угодно',
    'любое',
    'любую',
    'любой',
    'сама придумай',
    'сам придумай',
    'придумай',
    'на свое усмотрение',
    'на своё усмотрение',
)


@dataclass(slots=True)
class SystemActionResult:
    action_name: str
    message: str
    executed: bool


class SystemActionRunner:
    def __init__(self, config: SystemActionsConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.root_dir = _resolve_managed_root(config.document_dir)
        self.registry_path = _resolve_registry_path(config.registry_path)

    def handle(self, user_text: str) -> SystemActionResult | None:
        normalized = _normalize(user_text)
        if not normalized:
            return None

        if _is_destructive_request(normalized):
            return SystemActionResult(
                action_name='blocked_destructive_request',
                message='Нет. Удалять, перезаписывать, перемещать, очищать или форматировать файлы мне запрещено.',
                executed=False,
            )

        detected_action = self._detect_action(user_text, normalized)
        if detected_action is None:
            return None

        if not self.config.enabled:
            return SystemActionResult(
                action_name='system_actions_disabled',
                message="Системные действия отключены. Включи SYSTEM_ACTIONS_ENABLED='true' и запусти Герту заново.",
                executed=False,
            )

        try:
            return detected_action()
        except Exception as exc:
            self.logger.warning("System action failed: %s", exc)
            return SystemActionResult(
                action_name='system_action_failed',
                message=f'Не вышло выполнить системное действие: {exc}',
                executed=False,
            )

    def _detect_action(self, original_text: str, normalized: str):
        if _has_any(normalized, OPEN_WORDS):
            url = _extract_url(original_text)
            if url is not None:
                return lambda: self._open_url(url)

            if 'браузер' in normalized:
                return lambda: self._open_url(self.config.browser_home_url)

            if _mentions_vscode(normalized):
                return self._open_vscode

        search_query = _extract_search_query(original_text, normalized)
        if search_query is not None:
            return lambda: self._open_url(f'https://www.google.com/search?q={quote_plus(search_query)}', 'open_search')

        if _has_any(normalized, RENAME_WORDS) and (_mentions_text_document(normalized) or _mentions_folder(normalized)):
            return lambda: self._rename_created_item(original_text)

        if _has_any(normalized, WRITE_WORDS) and (_mentions_text_document(normalized) or _mentions_folder(normalized)):
            return lambda: self._append_text_document(original_text)

        if _has_any(normalized, CREATE_WORDS) and _mentions_folder(normalized) and _mentions_text_document(normalized):
            return lambda: self._create_folder_with_document(original_text)

        if _has_any(normalized, CREATE_WORDS) and _mentions_folder(normalized):
            return lambda: self._create_folder(original_text)

        if _has_any(normalized, CREATE_WORDS) and _mentions_text_document(normalized):
            return lambda: self._create_text_document(original_text)

        return None

    def _open_url(self, url: str, action_name: str = 'open_browser') -> SystemActionResult:
        safe_url = _normalize_url(url)
        webbrowser.open(safe_url, new=2)
        return SystemActionResult(
            action_name=action_name,
            message=f'Открываю: {safe_url}',
            executed=True,
        )

    def _open_vscode(self) -> SystemActionResult:
        command = _resolve_vscode_command(self.config.vscode_command)
        if command is None:
            return SystemActionResult(
                action_name='open_vscode',
                message="Не нашла команду VS Code. В VS Code включи команду 'code' в PATH или задай SYSTEM_ACTIONS_VSCODE_COMMAND.",
                executed=False,
            )

        args = [command]
        if self.config.vscode_open_workspace:
            args.append(str(Path.cwd()))
        subprocess.Popen(args, shell=False)
        return SystemActionResult(
            action_name='open_vscode',
            message='Открываю VS Code.',
            executed=True,
        )

    def _create_folder(self, user_text: str) -> SystemActionResult:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        folder_name = _extract_folder_name(user_text) or _generated_folder_name()
        target_path = _safe_folder_path(self.root_dir, folder_name)

        existing = target_path.exists()
        if existing and not self._is_registered(target_path, 'folder'):
            return SystemActionResult(
                action_name='create_folder',
                message=f'Папка уже существует, но она не создана Гертой, поэтому я не буду ее трогать: {target_path}',
                executed=False,
            )

        target_path.mkdir(parents=True, exist_ok=True)
        self._register(target_path, 'folder')
        return SystemActionResult(
            action_name='create_folder',
            message=f"{'Папка уже была создана' if existing else 'Создана папка'}: {target_path}",
            executed=not existing,
        )

    def _create_folder_with_document(self, user_text: str) -> SystemActionResult:
        folder_result = self._create_folder(user_text)
        if not folder_result.executed and 'уже была создана' not in folder_result.message:
            return folder_result

        folder_name = _extract_folder_name(user_text)
        folder_path = self._resolve_registered_folder(folder_name)
        return self._create_text_document(user_text, parent_dir=folder_path)

    def _create_text_document(self, user_text: str, parent_dir: Path | None = None) -> SystemActionResult:
        documents_dir = parent_dir or self._resolve_target_folder(user_text)
        documents_dir.mkdir(parents=True, exist_ok=True)

        title = _extract_document_title(user_text) or _generated_document_name()
        filename = _text_filename(title)
        content = _extract_document_content(user_text)

        target_path = _unique_child_path(documents_dir, filename)
        with target_path.open('x', encoding='utf-8') as file:
            if content:
                file.write(content)
                if not content.endswith('\n'):
                    file.write('\n')

        self._register(target_path, 'file')
        return SystemActionResult(
            action_name='create_text_document',
            message=f'Создан текстовый документ: {target_path}',
            executed=True,
        )

    def _append_text_document(self, user_text: str) -> SystemActionResult:
        documents_dir = self._resolve_target_folder(user_text)
        documents_dir.mkdir(parents=True, exist_ok=True)

        title = _extract_write_document_title(user_text) or _extract_document_title(user_text) or 'herta_note'
        filename = _text_filename(title)
        target_path = _safe_text_path(documents_dir, filename)
        content = _extract_write_document_content(user_text) or _extract_document_content(user_text)
        if not content:
            return SystemActionResult(
                action_name='append_text_document',
                message='Не вижу текст, который нужно записать. Скажи: "допиши в документ план текст купить чай".',
                executed=False,
            )

        if target_path.exists() and not self._is_registered(target_path, 'file'):
            return SystemActionResult(
                action_name='append_text_document',
                message=f'Файл уже существует, но он не создан Гертой, поэтому я не буду его менять: {target_path}',
                executed=False,
            )

        created = not target_path.exists()
        with target_path.open('a', encoding='utf-8') as file:
            if not created and target_path.stat().st_size > 0:
                file.write('\n')
            file.write(content)
            if not content.endswith('\n'):
                file.write('\n')

        self._register(target_path, 'file')
        return SystemActionResult(
            action_name='append_text_document',
            message=f"{'Создан и заполнен' if created else 'Записано в'} текстовый документ: {target_path}",
            executed=True,
        )

    def _rename_created_item(self, user_text: str) -> SystemActionResult:
        parsed = _extract_rename_request(user_text)
        if parsed is None:
            return SystemActionResult(
                action_name='rename_created_item',
                message='Не поняла, что во что переименовать. Пример: "переименуй документ план в задачи".',
                executed=False,
            )

        kind, old_name, new_name = parsed
        old_path = self._find_registered_by_name(old_name, kind)
        if old_path is None:
            return SystemActionResult(
                action_name='rename_created_item',
                message='Я могу переименовывать только файлы и папки, которые создала сама.',
                executed=False,
            )

        new_path = _safe_folder_path(old_path.parent, new_name) if kind == 'folder' else _safe_text_path(old_path.parent, _text_filename(new_name))
        if new_path.exists():
            return SystemActionResult(
                action_name='rename_created_item',
                message=f'Новое имя уже занято, поэтому не переименовываю: {new_path}',
                executed=False,
            )

        old_path.rename(new_path)
        self._replace_registered_path(old_path, new_path, kind)
        return SystemActionResult(
            action_name='rename_created_item',
            message=f'Переименовано: {old_path.name} -> {new_path.name}',
            executed=True,
        )

    def _resolve_target_folder(self, user_text: str) -> Path:
        folder_name = _extract_folder_name(user_text)
        if folder_name is None:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            return self.root_dir
        return self._resolve_registered_folder(folder_name)

    def _resolve_registered_folder(self, folder_name: str | None) -> Path:
        if folder_name is None:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            return self.root_dir

        folder_path = _safe_folder_path(self.root_dir, folder_name)
        if folder_path.exists() and not self._is_registered(folder_path, 'folder'):
            raise RuntimeError(f'Папка существует, но она не создана Гертой: {folder_path}')

        folder_path.mkdir(parents=True, exist_ok=True)
        self._register(folder_path, 'folder')
        return folder_path

    def _registry_entries(self) -> list[dict[str, str]]:
        if not self.registry_path.exists():
            return []
        try:
            raw_data = json.loads(self.registry_path.read_text(encoding='utf-8'))
        except Exception as exc:
            self.logger.warning("Failed to read system actions registry: %s", exc)
            return []
        if not isinstance(raw_data, list):
            return []
        return [entry for entry in raw_data if isinstance(entry, dict)]

    def _save_registry_entries(self, entries: list[dict[str, str]]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def _register(self, path: Path, kind: str) -> None:
        resolved = str(path.resolve())
        entries = [entry for entry in self._registry_entries() if entry.get('path') != resolved]
        entries.append(
            {
                'path': resolved,
                'kind': kind,
                'created_at': datetime.now().isoformat(timespec='seconds'),
            }
        )
        self._save_registry_entries(entries)

    def _replace_registered_path(self, old_path: Path, new_path: Path, kind: str) -> None:
        old_resolved = str(old_path.resolve())
        entries = self._registry_entries()
        replaced = False
        for entry in entries:
            if entry.get('path') == old_resolved and entry.get('kind') == kind:
                entry['path'] = str(new_path.resolve())
                entry['renamed_at'] = datetime.now().isoformat(timespec='seconds')
                replaced = True
                break
        if not replaced:
            entries.append(
                {
                    'path': str(new_path.resolve()),
                    'kind': kind,
                    'created_at': datetime.now().isoformat(timespec='seconds'),
                }
            )
        self._save_registry_entries(entries)

    def _is_registered(self, path: Path, kind: str | None = None) -> bool:
        resolved = str(path.resolve())
        return any(
            entry.get('path') == resolved and (kind is None or entry.get('kind') == kind)
            for entry in self._registry_entries()
        )

    def _find_registered_by_name(self, name: str, kind: str) -> Path | None:
        expected_name = _sanitize_folder_name(name) if kind == 'folder' else _text_filename(name)
        matches: list[Path] = []
        for entry in self._registry_entries():
            if entry.get('kind') != kind:
                continue
            path = Path(str(entry.get('path', ''))).resolve()
            if not path.exists():
                continue
            if path.name.lower() == expected_name.lower() or path.stem.lower() == Path(expected_name).stem.lower():
                matches.append(path)

        if len(matches) > 1:
            raise RuntimeError(f'Нашла несколько созданных объектов с именем {name!r}. Уточни имя.')
        return matches[0] if matches else None


def build_system_actions_instruction() -> str:
    return (
        'The local assistant has a limited safe OS action runner. It can open the default browser, open HTTP/HTTPS URLs, '
        'open VS Code, create folders, create new .txt files, append text to .txt files, and rename only files/folders '
        'that it created itself. It cannot delete, move, overwrite files, format drives, or run arbitrary shell commands. '
        'For supported OS action requests, answer briefly as if the action will be handled locally. For destructive '
        'requests, refuse briefly.'
    )


def _normalize(text: str) -> str:
    return ' '.join(text.strip().lower().split())


def _has_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _is_destructive_request(normalized: str) -> bool:
    return DESTRUCTIVE_RE.search(normalized) is not None


def _mentions_vscode(normalized: str) -> bool:
    return 'vscode' in normalized or 'vs code' in normalized or 'visual studio code' in normalized


def _mentions_text_document(normalized: str) -> bool:
    return any(word in normalized for word in ('текстовый документ', 'документ', 'файл', 'txt', 'заметк'))


def _mentions_folder(normalized: str) -> bool:
    return any(word in normalized for word in ('папк', 'каталог', 'директор'))


def _extract_url(text: str) -> str | None:
    match = URL_RE.search(text)
    if match is None:
        return None
    return match.group(0).rstrip('.,;:!?)]}')


def _normalize_url(url: str) -> str:
    stripped = url.strip()
    if stripped.lower().startswith(('http://', 'https://')):
        return stripped
    return f'https://{stripped}'


def _extract_search_query(original_text: str, normalized: str) -> str | None:
    for marker in SEARCH_MARKERS:
        marker_index = normalized.find(marker)
        if marker_index < 0:
            continue
        query_start = marker_index + len(marker)
        query = original_text[query_start:].strip(' .,:;-')
        return query or None
    return None


def _extract_folder_name(text: str) -> str | None:
    explicit_name = _extract_named_as(text)
    if explicit_name is not None:
        return explicit_name

    patterns = (
        r'(?:папк[ауе]|каталог|директори[юи])\s+(.+)$',
        r'(?:в|к)\s+(?:папк[еу]|каталог|директори[юи])\s+(.+)$',
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match is None:
            continue
        name = _meaningful_name_or_none(_trim_folder_name_tail(match.group(1)))
        if name is not None:
            return name
    return None


def _trim_folder_name_tail(raw_name: str) -> str:
    name = raw_name.strip(' "\'.,:;-')
    name = re.sub(
        r'^(?:и\s+)?(?:назови|назвать)\s+(?:ее|её|его|их)?\s*(?:как|в|на)?\s+',
        '',
        name,
        flags=re.IGNORECASE,
    )
    name = re.split(
        r'\s+(?:'
        r'и\s+(?:создай\s+)?(?:документ|файл|заметк[ауи]?|текстовый\s+документ)|'
        r'с\s+(?:документом|файлом|заметк[оа]й|текстом)|'
        r'со\s+(?:документом|файлом|текстом)|'
        r'внутри|туда\s+(?:документ|файл|заметк[ауи]?|текст)'
        r')\b',
        name,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return name.strip(' "\'.,:;-')


def _extract_document_title(text: str) -> str | None:
    patterns = (
        r'(?:с названием|под названием|назови)\s+(.+?)(?:\s+с текстом|\s+и текстом|\s+текстом|\s+в папк|\s+в каталог|$)',
        r'(?:документ|файл|заметку)\s+(.+?)(?:\s+с текстом|\s+и текстом|\s+текстом|\s+текст|\s+в папк|\s+в каталог|$)',
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match is None:
            continue
        title = _meaningful_name_or_none(match.group(1))
        if title and title.lower() not in {'текстовый', 'документ', 'файл', 'заметку'}:
            return title
    return None


def _extract_document_content(text: str) -> str:
    patterns = (
        r':\s*(.+)$',
        r'(?:с текстом|и текстом|текстом|текст|напиши туда)\s+(.+)$',
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match is None:
            continue
        content = match.group(1).strip()
        if content:
            return content
    return ''


def _extract_write_document_title(text: str) -> str | None:
    patterns = (
        r'(?:в|к)\s+(?:текстовый\s+)?(?:документ|файл|заметку)\s+(.+?)(?:\s+текст|\s+с текстом|\s+текстом|\s*:|$)',
        r'([^\s"<>:/\\|?*]+\.txt)\b',
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match is None:
            continue
        title = _meaningful_name_or_none(match.group(1))
        if title and title.lower() not in {'текстовый документ', 'документ', 'файл', 'заметку'}:
            return title
    return None


def _extract_write_document_content(text: str) -> str:
    patterns = (
        r':\s*(.+)$',
        r'(?:текст|с текстом|текстом|напиши туда)\s+(.+)$',
        r'(?:запиши|допиши|добавь|внеси|заполни|напиши)\s+(.+?)\s+(?:в|к)\s+(?:текстовый\s+)?(?:документ|файл|заметку)\b.*$',
        r'(?:запиши|допиши|добавь|внеси|заполни|напиши)\s+(.+)$',
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match is None:
            continue
        content = match.group(1).strip(' "\'.,:;-')
        if content:
            return content
    return ''


def _extract_rename_request(text: str) -> tuple[str, str, str] | None:
    match = re.search(
        r'переименуй\s+(?:(папку|каталог|директорию|документ|файл|заметку)\s+)?(.+?)\s+(?:в|на)\s+(.+)$',
        text,
        re.IGNORECASE,
    )
    if match is None:
        return None

    raw_kind = (match.group(1) or '').lower()
    old_name = match.group(2).strip(' "\'.,:;-')
    new_name = match.group(3).strip(' "\'.,:;-')
    meaningful_old_name = _meaningful_name_or_none(old_name)
    if not meaningful_old_name:
        return None

    kind = 'folder' if raw_kind in {'папку', 'каталог', 'директорию'} else 'file'
    meaningful_new_name = _meaningful_name_or_none(new_name)
    if meaningful_new_name is None:
        meaningful_new_name = _generated_folder_name() if kind == 'folder' else _generated_document_name()
    return kind, meaningful_old_name, meaningful_new_name


def _text_filename(title: str) -> str:
    filename = _sanitize_folder_name(title)
    if not filename.lower().endswith('.txt'):
        filename = f'{filename}.txt'
    return filename


def _extract_named_as(text: str) -> str | None:
    match = re.search(
        r'(?:назови|назвать|имя|название)\s+(?:ее|её|его|их|папку|файл|документ|заметку)?\s*(?:как|в|на)?\s+(.+?)(?:\s+(?:с текстом|и текстом|текстом|документ|файл|папк|туда)|$)',
        text,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return _meaningful_name_or_none(match.group(1))


def _meaningful_name_or_none(raw_name: str) -> str | None:
    name = raw_name.strip(' "\'.,:;-')
    normalized = _normalize(name.replace('-', ' '))
    if not normalized:
        return None
    if normalized in {'и', 'ее', 'её', 'его', 'их', 'назови', 'назвать', 'папку', 'файл', 'документ'}:
        return None
    if any(pattern in normalized for pattern in VAGUE_NAME_PATTERNS):
        return None
    name = re.sub(r'^(?:и\s+)?(?:назови|назвать)\s+(?:ее|её|его|их)?\s*(?:как|в|на)?\s+', '', name, flags=re.IGNORECASE)
    name = name.strip(' "\'.,:;-')
    return name or None


def _generated_folder_name() -> str:
    names = (
        'Рабочие материалы Герты',
        'Заметки для наблюдений',
        'Архив идей Герты',
        'Черновики Герты',
        'Полезные материалы',
    )
    now = datetime.now()
    return f'{names[now.second % len(names)]} {now:%Y-%m-%d %H-%M}'


def _generated_document_name() -> str:
    return datetime.now().strftime('Заметка Герты %Y-%m-%d %H-%M')


def _sanitize_folder_name(name: str) -> str:
    cleaned = INVALID_PATH_CHARS_RE.sub('_', name).strip(' ._')
    if not cleaned:
        return datetime.now().strftime('herta_item_%Y%m%d_%H%M%S')
    return cleaned[:80]


def _resolve_managed_root(document_dir: str) -> Path:
    if document_dir.strip().lower() in {'desktop', 'рабочий стол'}:
        return _resolve_desktop_dir()

    path = Path(document_dir).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _resolve_registry_path(registry_path: str) -> Path:
    path = Path(registry_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _resolve_desktop_dir() -> Path:
    candidates = []
    user_profile = os.getenv('USERPROFILE')
    if user_profile:
        candidates.append(Path(user_profile) / 'Desktop')
    candidates.append(Path.home() / 'Desktop')

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _unique_child_path(parent: Path, filename: str) -> Path:
    base_path = _safe_text_path(parent, filename)
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    for index in range(2, 1000):
        candidate = _safe_text_path(parent, f'{stem}_{index}{suffix}')
        if not candidate.exists():
            return candidate
    raise RuntimeError('Could not create a unique text document name.')


def _safe_text_path(parent: Path, filename: str) -> Path:
    parent_path = parent.resolve()
    child_path = (parent_path / filename).resolve()
    if child_path.parent != parent_path:
        raise ValueError('Unsafe document path.')
    if child_path.suffix.lower() != '.txt':
        raise ValueError('Only .txt documents are allowed.')
    return child_path


def _safe_folder_path(parent: Path, folder_name: str) -> Path:
    parent_path = parent.resolve()
    child_path = (parent_path / _sanitize_folder_name(folder_name)).resolve()
    if child_path.parent != parent_path:
        raise ValueError('Unsafe folder path.')
    return child_path


def _resolve_vscode_command(command: str) -> str | None:
    configured = command.strip()
    if configured:
        configured_path = Path(configured)
        if configured_path.is_absolute() and configured_path.exists():
            return str(configured_path)

        found = shutil.which(configured)
        if found is not None:
            return found

    return shutil.which('code') or shutil.which('code.cmd')
