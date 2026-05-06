from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from actions.system_actions import SystemActionRunner
from brain.memory import DialogueMemory
from config import AppConfig


LIVE_MODEL_NAMES = {
    'gemini-3.1-flash-live-preview',
    'gemini-2.5-flash-native-audio-preview-12-2025',
}
VALID_LIVE_PLAYBACK_MODES = {'google', 'rvc'}
VALID_RVC_BACKENDS = {'persistent', 'worker', 'subprocess', 'cli'}
VALID_RVC_BASE_TTS = {'silero', 'piper'}
VALID_RVC_F0_METHODS = {'rmvpe', 'fcpe', 'crepe', 'crepe-tiny'}


@dataclass(slots=True)
class DoctorCheck:
    status: str
    name: str
    detail: str


class DoctorReport:
    def __init__(self) -> None:
        self.checks: list[DoctorCheck] = []

    def ok(self, name: str, detail: str) -> None:
        self.checks.append(DoctorCheck('OK', name, detail))

    def warn(self, name: str, detail: str) -> None:
        self.checks.append(DoctorCheck('WARN', name, detail))

    def fail(self, name: str, detail: str) -> None:
        self.checks.append(DoctorCheck('FAIL', name, detail))

    @property
    def has_failures(self) -> bool:
        return any(check.status == 'FAIL' for check in self.checks)

    def print(self) -> None:
        print("The Herta doctor")
        print("================")
        for check in self.checks:
            print(f"[{check.status}] {check.name}: {check.detail}")

        counts = {status: 0 for status in ('OK', 'WARN', 'FAIL')}
        for check in self.checks:
            counts[check.status] += 1
        print("----------------")
        print(f"Summary: ok={counts['OK']}, warn={counts['WARN']}, fail={counts['FAIL']}")


def run_doctor(config: AppConfig) -> int:
    report = DoctorReport()
    _check_python(report)
    _check_imports(report, config)
    _check_models(report, config)
    _check_audio(report, config)
    _check_memory(report, config)
    _check_system_actions(report, config)
    _check_rvc(report, config)
    report.print()
    return 1 if report.has_failures else 0


def _check_python(report: DoctorReport) -> None:
    version = sys.version_info
    detail = f"{platform.python_implementation()} {version.major}.{version.minor}.{version.micro}"
    if version < (3, 11):
        report.fail('Python', f'{detail}; use Python 3.11 or newer.')
        return
    if version >= (3, 14):
        report.warn('Python', f'{detail}; some audio/ML packages may not support this version yet.')
        return
    report.ok('Python', detail)


def _check_imports(report: DoctorReport, config: AppConfig) -> None:
    required_modules = [
        ('dotenv', 'python-dotenv'),
        ('httpx', 'httpx'),
        ('numpy', 'numpy'),
        ('sounddevice', 'sounddevice'),
        ('google.genai', 'google-genai'),
    ]

    if config.llm_provider == 'ollama':
        required_modules.append(('ollama', 'ollama'))
    if config.stt_provider in {'whisper', 'faster_whisper', 'faster-whisper'}:
        required_modules.append(('faster_whisper', 'faster-whisper'))
        required_modules.append(('silero_vad', 'silero-vad'))
    if config.rvc_tts.enabled or config.google_ai.live_playback == 'rvc':
        required_modules.append(('torch', 'torch'))
        if config.rvc_tts.base_tts.strip().lower() == 'piper':
            required_modules.append(('piper', 'piper-tts'))

    seen: set[str] = set()
    for module_name, package_name in required_modules:
        if module_name in seen:
            continue
        seen.add(module_name)
        if _module_exists(module_name):
            report.ok(f'Package {package_name}', f"module '{module_name}' is importable.")
        else:
            report.fail(f'Package {package_name}', f"module '{module_name}' is missing. Run: python -m pip install -r requirements.txt")


def _check_models(report: DoctorReport, config: AppConfig) -> None:
    if config.google_ai.api_key:
        report.ok('Google AI API key', 'configured via GOOGLE_AI_API_KEY or GEMINI_API_KEY.')
    else:
        report.warn('Google AI API key', 'not configured. Google AI and Gemini Live modes will fail.')

    live_model = config.google_ai.live_model
    if live_model in LIVE_MODEL_NAMES:
        report.ok('Gemini Live model', live_model)
    else:
        report.warn(
            'Gemini Live model',
            f"{live_model!r}; preferred models are {', '.join(sorted(LIVE_MODEL_NAMES))}.",
        )

    live_playback = config.google_ai.live_playback
    if live_playback in VALID_LIVE_PLAYBACK_MODES:
        report.ok('Gemini Live playback', live_playback)
    else:
        report.fail('Gemini Live playback', f"{live_playback!r}; use 'google' or 'rvc'.")

    if live_playback == 'rvc' and not config.rvc_tts.enabled:
        report.fail('Live RVC playback', "GOOGLE_AI_LIVE_PLAYBACK='rvc' requires RVC_TTS_ENABLED='true'.")
    elif live_playback == 'rvc':
        report.ok('Live RVC playback', 'enabled.')

    if config.llm_provider not in {'ollama', 'deepseek', *{'google_ai', 'google', 'gemini'}}:
        report.fail('LLM provider', f"{config.llm_provider!r}; use ollama, deepseek, or google_ai.")
    else:
        report.ok('LLM provider', config.llm_provider)

    if config.stt_provider not in {'whisper', 'faster_whisper', 'faster-whisper', 'google_ai', 'google', 'gemini'}:
        report.fail('STT provider', f"{config.stt_provider!r}; use whisper or google_ai.")
    else:
        report.ok('STT provider', config.stt_provider)


def _check_audio(report: DoctorReport, config: AppConfig) -> None:
    try:
        import sounddevice as sd
    except Exception as exc:
        report.fail('Audio devices', f'sounddevice import failed: {exc}')
        return

    try:
        devices = sd.query_devices()
    except Exception as exc:
        report.fail('Audio devices', f'could not query devices: {exc}')
        return

    input_count = sum(1 for device in devices if int(device.get('max_input_channels', 0)) > 0)
    output_count = sum(1 for device in devices if int(device.get('max_output_channels', 0)) > 0)
    report.ok('Audio device scan', f'inputs={input_count}, outputs={output_count}.')
    _check_audio_device(report, 'Input device', config.audio.device, 'input')
    _check_audio_device(report, 'Output device', config.audio_output.device, 'output')


def _check_audio_device(report: DoctorReport, name: str, device: int | str | None, kind: str) -> None:
    try:
        import sounddevice as sd

        if device is None:
            default_index = sd.default.device[0 if kind == 'input' else 1]
            if default_index is None or int(default_index) < 0:
                report.warn(name, f'not configured and no default {kind} device is available.')
                return
            selected = sd.query_devices(default_index)
            report.ok(name, f"default {kind}: {default_index}: {selected['name']}")
            return

        selected = sd.query_devices(device)
        channels_key = 'max_input_channels' if kind == 'input' else 'max_output_channels'
        channels = int(selected.get(channels_key, 0))
        if channels <= 0:
            report.fail(name, f"{device!r}: {selected['name']} has no {kind} channels.")
            return
        report.ok(name, f"{selected['index']}: {selected['name']} | {kind}_channels={channels}")
    except Exception as exc:
        report.fail(name, f'{device!r} is not usable: {exc}')


def _check_memory(report: DoctorReport, config: AppConfig) -> None:
    if not config.memory.enabled:
        report.warn('Dialogue memory', 'disabled.')
        return

    memory = DialogueMemory(config.memory)
    try:
        context_messages = memory.load_context_messages()
    except Exception as exc:
        report.fail('Dialogue memory', f'could not load context: {exc}')
        return

    stored_messages = _count_json_messages(memory.path)
    if stored_messages is None:
        report.ok('Dialogue memory', f'path={memory.path}; no file yet; context_loaded={len(context_messages)}.')
    else:
        report.ok(
            'Dialogue memory',
            f'path={memory.path}; stored={stored_messages}; context_loaded={len(context_messages)}; max={config.memory.max_messages}.',
        )

    if config.memory.context_messages > config.memory.max_messages and config.memory.max_messages > 0:
        report.warn('Dialogue memory limits', 'MEMORY_CONTEXT_MESSAGES is greater than MEMORY_MAX_MESSAGES.')
    else:
        report.ok('Dialogue memory limits', f'context={config.memory.context_messages}, max={config.memory.max_messages}.')


def _check_system_actions(report: DoctorReport, config: AppConfig) -> None:
    if not config.system_actions.enabled:
        report.warn('System actions', 'disabled.')
        return

    try:
        runner = SystemActionRunner(config.system_actions)
    except Exception as exc:
        report.fail('System actions', f'could not initialize: {exc}')
        return

    report.ok('System actions', f'enabled; root={runner.root_dir}; registry={runner.registry_path}.')
    if shutil.which(config.system_actions.vscode_command):
        report.ok('VS Code command', f"{config.system_actions.vscode_command!r} found in PATH.")
    else:
        report.warn('VS Code command', f"{config.system_actions.vscode_command!r} was not found in PATH.")


def _check_rvc(report: DoctorReport, config: AppConfig) -> None:
    needs_rvc = config.rvc_tts.enabled or config.google_ai.live_playback == 'rvc'
    if not needs_rvc:
        report.warn('RVC TTS', 'disabled.')
        return

    backend = config.rvc_tts.backend.strip().lower()
    if backend in VALID_RVC_BACKENDS:
        report.ok('RVC backend', backend)
    else:
        report.fail('RVC backend', f"{backend!r}; use persistent or subprocess.")

    base_tts = config.rvc_tts.base_tts.strip().lower()
    if base_tts in VALID_RVC_BASE_TTS:
        report.ok('RVC base TTS', base_tts)
    else:
        report.fail('RVC base TTS', f"{base_tts!r}; use silero or piper.")

    f0_method = config.rvc_tts.f0_method.strip().lower()
    if f0_method in VALID_RVC_F0_METHODS:
        report.ok('RVC f0 method', f0_method)
    else:
        report.warn('RVC f0 method', f"{f0_method!r}; known values: {', '.join(sorted(VALID_RVC_F0_METHODS))}.")

    _check_path_exists(report, 'Applio root', Path(config.rvc_tts.applio_root), expect_dir=True)
    _check_path_exists(report, 'Applio Python', Path(config.rvc_tts.applio_python), expect_dir=False)
    _check_path_exists(report, 'RVC model', Path(config.rvc_tts.model_path), expect_dir=False)
    if config.rvc_tts.index_path:
        _check_path_exists(report, 'RVC index', Path(config.rvc_tts.index_path), expect_dir=False)
    else:
        report.warn('RVC index', 'not configured; conversion will run without index retrieval.')


def _check_path_exists(report: DoctorReport, name: str, path: Path, *, expect_dir: bool) -> None:
    if not path.exists():
        report.fail(name, f'not found: {path}')
        return
    if expect_dir and not path.is_dir():
        report.fail(name, f'expected directory, got file: {path}')
        return
    if not expect_dir and not path.is_file():
        report.fail(name, f'expected file, got directory: {path}')
        return
    report.ok(name, str(path))


def _module_exists(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _count_json_messages(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        payload: Any = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return 0
    if not isinstance(payload, dict):
        return 0
    messages = payload.get('messages')
    return len(messages) if isinstance(messages, list) else 0
