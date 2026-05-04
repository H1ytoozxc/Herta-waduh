import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


load_dotenv()
PREFERRED_AUDIO_NAME = 'fifine'
DEFAULT_PIPER_MODEL_PATH = 'models/piper/ru_RU-irina-medium.onnx'
DEFAULT_RVC_MODEL_PATH = 'Z:\\' + '\u0413\u0415\u0420\u0422\u0410\u0410\u0410\u0410' + '\\model.pth'


def _get_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _get_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _get_optional_int(value: str | None) -> int | None:
    parsed = _get_optional_str(value)
    return int(parsed) if parsed is not None else None


def _detect_sounddevice(kind: str, preferred_name: str = PREFERRED_AUDIO_NAME) -> int | None:
    try:
        import sounddevice as sd
    except Exception:
        return None

    preferred = preferred_name.strip().lower()
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        name = str(device.get('name', '')).lower()
        if preferred not in name:
            continue
        if kind == 'input' and int(device.get('max_input_channels', 0)) > 0:
            return index
        if kind == 'output' and int(device.get('max_output_channels', 0)) > 0:
            return index
    return None


def _get_device(value: str | None, *, kind: str) -> int | str | None:
    parsed = _get_optional_str(value)
    if parsed is not None:
        return int(parsed) if parsed.isdigit() else parsed
    return _detect_sounddevice(kind)


@dataclass(slots=True)
class OllamaConfig:
    host: str = 'http://127.0.0.1:11434'
    model: str = 'qwen3:4b'
    timeout_seconds: float = 300.0
    keep_alive: str = '10m'
    think: bool = False
    temperature: float = 0.55
    num_ctx: int = 2048
    num_gpu: int | None = None


@dataclass(slots=True)
class DeepSeekConfig:
    api_key: str | None = None
    base_url: str = 'https://api.deepseek.com'
    model: str = 'deepseek-v4-flash'
    timeout_seconds: float = 120.0
    temperature: float = 0.55
    max_tokens: int = 700
    retry_attempts: int = 4
    rate_limit_retries: int = 2


@dataclass(slots=True)
class GoogleAIConfig:
    api_key: str | None = None
    base_url: str = 'https://generativelanguage.googleapis.com/v1beta'
    model: str = 'gemma-3-27b-it'
    timeout_seconds: float = 120.0
    temperature: float = 0.55
    max_tokens: int = 700
    retry_attempts: int = 4
    rate_limit_retries: int = 2
    system_instruction_enabled: bool = False
    live_model: str = 'gemini-3.1-flash-live-preview'
    live_api_version: str = 'v1beta'
    live_voice_name: str | None = 'Kore'
    live_thinking_level: str | None = 'minimal'
    live_thinking_budget: int | None = None
    live_affective_dialog: bool = False
    live_proactive_audio: bool = False
    live_input_transcription: bool = True
    live_output_transcription: bool = True


@dataclass(slots=True)
class EdgeTTSConfig:
    enabled: bool = True
    prefer_local: bool = True
    voice: str = 'ru-RU-DariyaNeural'
    rate: str = '-6%'
    volume: str = '+0%'
    pitch: str = '+8Hz'
    sapi_voice: str | None = 'Microsoft Irina Desktop - Russian'
    sapi_rate: int = 0
    sapi_volume: int = 100
    piper_model_path: str | None = DEFAULT_PIPER_MODEL_PATH
    piper_config_path: str | None = None
    piper_use_cuda: bool = False


@dataclass(slots=True)
class RvcTTSConfig:
    enabled: bool = False
    backend: str = 'persistent'
    warm_up: bool = True
    base_tts: str = 'silero'
    applio_root: str = r'Z:\APPLIO'
    applio_python: str = r'Z:\APPLIO\env\python.exe'
    model_path: str = DEFAULT_RVC_MODEL_PATH
    index_path: str | None = None
    pitch: int = 0
    f0_method: str = 'rmvpe'
    index_rate: float = 0.3
    protect: float = 0.33
    silero_model: str = 'v4_ru'
    silero_speaker: str = 'xenia'
    silero_sample_rate: int = 24000
    silero_device: str = 'cpu'
    piper_model_path: str | None = DEFAULT_PIPER_MODEL_PATH
    piper_config_path: str | None = None
    piper_use_cuda: bool = False
    worker_start_timeout_seconds: float = 120.0
    conversion_timeout_seconds: float = 300.0


@dataclass(slots=True)
class AudioInputConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = 'float32'
    block_size: int = 512
    device: int | str | None = None
    queue_max_chunks: int = 128


@dataclass(slots=True)
class AudioOutputConfig:
    sample_rate: int = 44100
    channels: int = 2
    dtype: str = 'float32'
    device: int | str | None = None
    tone_frequency_hz: float = 523.25
    tone_duration_seconds: float = 0.7
    tone_volume: float = 0.2


@dataclass(slots=True)
class VadConfig:
    threshold: float = 0.5
    min_silence_duration_ms: int = 600
    speech_pad_ms: int = 200
    min_utterance_duration_ms: int = 450
    max_utterance_seconds: float = 20.0


@dataclass(slots=True)
class WhisperSTTConfig:
    model_size: str = 'small'
    device: str = 'cpu'
    compute_type: str = 'int8'
    cpu_threads: int = 4
    num_workers: int = 1
    language: str | None = None
    beam_size: int = 5
    best_of: int = 5
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -0.8
    compression_ratio_threshold: float = 2.2
    min_peak_level: float = 0.01
    min_rms_level: float = 0.0015
    normalize_audio: bool = True
    local_files_only: bool = False
    download_root: str | None = None
    initial_prompt: str | None = None


@dataclass(slots=True)
class MemoryConfig:
    enabled: bool = True
    path: str = 'data/dialogue_memory.json'
    max_messages: int = 80
    context_messages: int = 12


@dataclass(slots=True)
class AppConfig:
    log_level: str = 'INFO'
    llm_provider: str = 'ollama'
    max_history_messages: int = 8
    persona_rewrite_enabled: bool = False
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    deepseek: DeepSeekConfig = field(default_factory=DeepSeekConfig)
    google_ai: GoogleAIConfig = field(default_factory=GoogleAIConfig)
    tts: EdgeTTSConfig = field(default_factory=EdgeTTSConfig)
    rvc_tts: RvcTTSConfig = field(default_factory=RvcTTSConfig)
    audio: AudioInputConfig = field(default_factory=AudioInputConfig)
    audio_output: AudioOutputConfig = field(default_factory=AudioOutputConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    stt: WhisperSTTConfig = field(default_factory=WhisperSTTConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def load_config() -> AppConfig:
    return AppConfig(
        log_level=os.getenv('LOG_LEVEL', 'INFO').upper(),
        llm_provider=os.getenv('LLM_PROVIDER', 'ollama').strip().lower(),
        max_history_messages=int(os.getenv('MAX_HISTORY_MESSAGES', '8')),
        persona_rewrite_enabled=_get_bool(os.getenv('PERSONA_REWRITE_ENABLED'), False),
        ollama=OllamaConfig(
            host=os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434'),
            model=os.getenv('OLLAMA_MODEL', 'qwen3:4b'),
            timeout_seconds=float(os.getenv('OLLAMA_TIMEOUT_SECONDS', '300')),
            keep_alive=os.getenv('OLLAMA_KEEP_ALIVE', '10m'),
            think=_get_bool(os.getenv('OLLAMA_THINK'), False),
            temperature=float(os.getenv('OLLAMA_TEMPERATURE', '0.55')),
            num_ctx=int(os.getenv('OLLAMA_NUM_CTX', '2048')),
            num_gpu=_get_optional_int(os.getenv('OLLAMA_NUM_GPU')),
        ),
        deepseek=DeepSeekConfig(
            api_key=_get_optional_str(os.getenv('DEEPSEEK_API_KEY')),
            base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
            model=os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-flash'),
            timeout_seconds=float(os.getenv('DEEPSEEK_TIMEOUT_SECONDS', '120')),
            temperature=float(os.getenv('DEEPSEEK_TEMPERATURE', '0.55')),
            max_tokens=int(os.getenv('DEEPSEEK_MAX_TOKENS', '700')),
            retry_attempts=int(os.getenv('DEEPSEEK_RETRY_ATTEMPTS', '4')),
            rate_limit_retries=int(os.getenv('DEEPSEEK_RATE_LIMIT_RETRIES', '2')),
        ),
        google_ai=GoogleAIConfig(
            api_key=_get_optional_str(os.getenv('GOOGLE_AI_API_KEY')) or _get_optional_str(os.getenv('GEMINI_API_KEY')),
            base_url=os.getenv('GOOGLE_AI_BASE_URL', 'https://generativelanguage.googleapis.com/v1beta'),
            model=os.getenv('GOOGLE_AI_MODEL', 'gemma-3-27b-it'),
            timeout_seconds=float(os.getenv('GOOGLE_AI_TIMEOUT_SECONDS', '120')),
            temperature=float(os.getenv('GOOGLE_AI_TEMPERATURE', '0.55')),
            max_tokens=int(os.getenv('GOOGLE_AI_MAX_TOKENS', '700')),
            retry_attempts=int(os.getenv('GOOGLE_AI_RETRY_ATTEMPTS', '4')),
            rate_limit_retries=int(os.getenv('GOOGLE_AI_RATE_LIMIT_RETRIES', '2')),
            system_instruction_enabled=_get_bool(os.getenv('GOOGLE_AI_SYSTEM_INSTRUCTION_ENABLED'), False),
            live_model=os.getenv('GOOGLE_AI_LIVE_MODEL', 'gemini-3.1-flash-live-preview'),
            live_api_version=os.getenv('GOOGLE_AI_LIVE_API_VERSION', 'v1beta'),
            live_voice_name=_get_optional_str(os.getenv('GOOGLE_AI_LIVE_VOICE', 'Kore')),
            live_thinking_level=_get_optional_str(os.getenv('GOOGLE_AI_LIVE_THINKING_LEVEL', 'minimal')),
            live_thinking_budget=_get_optional_int(os.getenv('GOOGLE_AI_LIVE_THINKING_BUDGET')),
            live_affective_dialog=_get_bool(os.getenv('GOOGLE_AI_LIVE_AFFECTIVE_DIALOG'), False),
            live_proactive_audio=_get_bool(os.getenv('GOOGLE_AI_LIVE_PROACTIVE_AUDIO'), False),
            live_input_transcription=_get_bool(os.getenv('GOOGLE_AI_LIVE_INPUT_TRANSCRIPTION'), True),
            live_output_transcription=_get_bool(os.getenv('GOOGLE_AI_LIVE_OUTPUT_TRANSCRIPTION'), True),
        ),
        tts=EdgeTTSConfig(
            enabled=_get_bool(os.getenv('EDGE_TTS_ENABLED'), True),
            prefer_local=_get_bool(os.getenv('TTS_PREFER_LOCAL'), True),
            voice=os.getenv('EDGE_TTS_VOICE', 'ru-RU-DariyaNeural'),
            rate=os.getenv('EDGE_TTS_RATE', '-6%'),
            volume=os.getenv('EDGE_TTS_VOLUME', '+0%'),
            pitch=os.getenv('EDGE_TTS_PITCH', '+8Hz'),
            sapi_voice=_get_optional_str(os.getenv('SAPI_VOICE', 'Microsoft Irina Desktop - Russian')),
            sapi_rate=int(os.getenv('SAPI_RATE', '0')),
            sapi_volume=int(os.getenv('SAPI_VOLUME', '100')),
            piper_model_path=_get_optional_str(os.getenv('PIPER_MODEL_PATH', DEFAULT_PIPER_MODEL_PATH)),
            piper_config_path=_get_optional_str(os.getenv('PIPER_CONFIG_PATH')),
            piper_use_cuda=_get_bool(os.getenv('PIPER_USE_CUDA'), False),
        ),
        rvc_tts=RvcTTSConfig(
            enabled=_get_bool(os.getenv('RVC_TTS_ENABLED'), False),
            backend=os.getenv('RVC_BACKEND', 'persistent'),
            warm_up=_get_bool(os.getenv('RVC_WARM_UP'), True),
            base_tts=os.getenv('RVC_BASE_TTS', 'silero'),
            applio_root=os.getenv('RVC_APPLIO_ROOT', r'Z:\APPLIO'),
            applio_python=os.getenv('RVC_APPLIO_PYTHON', r'Z:\APPLIO\env\python.exe'),
            model_path=os.getenv('RVC_MODEL_PATH', DEFAULT_RVC_MODEL_PATH),
            index_path=_get_optional_str(os.getenv('RVC_INDEX_PATH')),
            pitch=int(os.getenv('RVC_PITCH', '0')),
            f0_method=os.getenv('RVC_F0_METHOD', 'rmvpe'),
            index_rate=float(os.getenv('RVC_INDEX_RATE', '0.3')),
            protect=float(os.getenv('RVC_PROTECT', '0.33')),
            silero_model=os.getenv('SILERO_TTS_MODEL', 'v4_ru'),
            silero_speaker=os.getenv('SILERO_TTS_SPEAKER', 'xenia'),
            silero_sample_rate=int(os.getenv('SILERO_TTS_SAMPLE_RATE', '24000')),
            silero_device=os.getenv('SILERO_TTS_DEVICE', 'cpu'),
            piper_model_path=_get_optional_str(os.getenv('RVC_PIPER_MODEL_PATH', DEFAULT_PIPER_MODEL_PATH)),
            piper_config_path=_get_optional_str(os.getenv('RVC_PIPER_CONFIG_PATH')),
            piper_use_cuda=_get_bool(os.getenv('RVC_PIPER_USE_CUDA'), False),
            worker_start_timeout_seconds=float(os.getenv('RVC_WORKER_START_TIMEOUT_SECONDS', '120')),
            conversion_timeout_seconds=float(os.getenv('RVC_CONVERSION_TIMEOUT_SECONDS', '300')),
        ),
        audio=AudioInputConfig(
            sample_rate=int(os.getenv('AUDIO_SAMPLE_RATE', '16000')),
            channels=int(os.getenv('AUDIO_CHANNELS', '1')),
            dtype=os.getenv('AUDIO_DTYPE', 'float32'),
            block_size=int(os.getenv('AUDIO_BLOCK_SIZE', '512')),
            device=_get_device(os.getenv('AUDIO_DEVICE'), kind='input'),
            queue_max_chunks=int(os.getenv('AUDIO_QUEUE_MAX_CHUNKS', '128')),
        ),
        audio_output=AudioOutputConfig(
            sample_rate=int(os.getenv('AUDIO_OUTPUT_SAMPLE_RATE', '44100')),
            channels=int(os.getenv('AUDIO_OUTPUT_CHANNELS', '2')),
            dtype=os.getenv('AUDIO_OUTPUT_DTYPE', 'float32'),
            device=_get_device(os.getenv('AUDIO_OUTPUT_DEVICE'), kind='output'),
            tone_frequency_hz=float(os.getenv('AUDIO_OUTPUT_TONE_FREQUENCY_HZ', '523.25')),
            tone_duration_seconds=float(os.getenv('AUDIO_OUTPUT_TONE_DURATION_SECONDS', '0.7')),
            tone_volume=float(os.getenv('AUDIO_OUTPUT_TONE_VOLUME', '0.2')),
        ),
        vad=VadConfig(
            threshold=float(os.getenv('VAD_THRESHOLD', '0.5')),
            min_silence_duration_ms=int(os.getenv('VAD_MIN_SILENCE_MS', '600')),
            speech_pad_ms=int(os.getenv('VAD_SPEECH_PAD_MS', '200')),
            min_utterance_duration_ms=int(os.getenv('VAD_MIN_UTTERANCE_MS', '450')),
            max_utterance_seconds=float(os.getenv('VAD_MAX_UTTERANCE_SECONDS', '20')),
        ),
        stt=WhisperSTTConfig(
            model_size=os.getenv('WHISPER_MODEL_SIZE', 'small'),
            device=os.getenv('WHISPER_DEVICE', 'cpu'),
            compute_type=os.getenv('WHISPER_COMPUTE_TYPE', 'int8'),
            cpu_threads=int(os.getenv('WHISPER_CPU_THREADS', '4')),
            num_workers=int(os.getenv('WHISPER_NUM_WORKERS', '1')),
            language=_get_optional_str(os.getenv('WHISPER_LANGUAGE')),
            beam_size=int(os.getenv('WHISPER_BEAM_SIZE', '5')),
            best_of=int(os.getenv('WHISPER_BEST_OF', '5')),
            no_speech_threshold=float(os.getenv('WHISPER_NO_SPEECH_THRESHOLD', '0.6')),
            log_prob_threshold=float(os.getenv('WHISPER_LOG_PROB_THRESHOLD', '-0.8')),
            compression_ratio_threshold=float(os.getenv('WHISPER_COMPRESSION_RATIO_THRESHOLD', '2.2')),
            min_peak_level=float(os.getenv('WHISPER_MIN_PEAK_LEVEL', '0.01')),
            min_rms_level=float(os.getenv('WHISPER_MIN_RMS_LEVEL', '0.0015')),
            normalize_audio=_get_bool(os.getenv('WHISPER_NORMALIZE_AUDIO'), True),
            local_files_only=_get_bool(os.getenv('WHISPER_LOCAL_FILES_ONLY'), False),
            download_root=_get_optional_str(os.getenv('WHISPER_DOWNLOAD_ROOT')),
            initial_prompt=_get_optional_str(os.getenv('WHISPER_INITIAL_PROMPT')),
        ),
        memory=MemoryConfig(
            enabled=_get_bool(os.getenv('MEMORY_ENABLED'), True),
            path=os.getenv('MEMORY_PATH', 'data/dialogue_memory.json'),
            max_messages=int(os.getenv('MEMORY_MAX_MESSAGES', '80')),
            context_messages=int(os.getenv('MEMORY_CONTEXT_MESSAGES', '12')),
        ),
    )
