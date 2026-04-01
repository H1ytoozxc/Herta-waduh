import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


load_dotenv()


def _get_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}



def _get_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None



def _get_device(value: str | None) -> int | str | None:
    parsed = _get_optional_str(value)
    if parsed is None:
        return None
    return int(parsed) if parsed.isdigit() else parsed


@dataclass(slots=True)
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen3:4b"
    timeout_seconds: float = 120.0
    keep_alive: str = "10m"
    think: bool = False
    temperature: float = 0.55


@dataclass(slots=True)
class EdgeTTSConfig:
    enabled: bool = True
    voice: str = "ru-RU-SvetlanaNeural"
    rate: str = "+0%"
    volume: str = "+0%"
    pitch: str = "+0Hz"


@dataclass(slots=True)
class AudioInputConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    block_size: int = 512
    device: int | str | None = None
    queue_max_chunks: int = 128


@dataclass(slots=True)
class VadConfig:
    threshold: float = 0.5
    min_silence_duration_ms: int = 600
    speech_pad_ms: int = 200
    min_utterance_duration_ms: int = 450
    max_utterance_seconds: float = 20.0


@dataclass(slots=True)
class WhisperSTTConfig:
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
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
class AppConfig:
    log_level: str = "INFO"
    max_history_messages: int = 8
    persona_rewrite_enabled: bool = False
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    tts: EdgeTTSConfig = field(default_factory=EdgeTTSConfig)
    audio: AudioInputConfig = field(default_factory=AudioInputConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    stt: WhisperSTTConfig = field(default_factory=WhisperSTTConfig)



def load_config() -> AppConfig:
    return AppConfig(
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        max_history_messages=int(os.getenv("MAX_HISTORY_MESSAGES", "8")),
        persona_rewrite_enabled=_get_bool(os.getenv("PERSONA_REWRITE_ENABLED"), False),
        ollama=OllamaConfig(
            host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            model=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "10m"),
            think=_get_bool(os.getenv("OLLAMA_THINK"), False),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.55")),
        ),
        tts=EdgeTTSConfig(
            enabled=_get_bool(os.getenv("EDGE_TTS_ENABLED"), True),
            voice=os.getenv("EDGE_TTS_VOICE", "ru-RU-SvetlanaNeural"),
            rate=os.getenv("EDGE_TTS_RATE", "+0%"),
            volume=os.getenv("EDGE_TTS_VOLUME", "+0%"),
            pitch=os.getenv("EDGE_TTS_PITCH", "+0Hz"),
        ),
        audio=AudioInputConfig(
            sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
            channels=int(os.getenv("AUDIO_CHANNELS", "1")),
            dtype=os.getenv("AUDIO_DTYPE", "float32"),
            block_size=int(os.getenv("AUDIO_BLOCK_SIZE", "512")),
            device=_get_device(os.getenv("AUDIO_DEVICE")),
            queue_max_chunks=int(os.getenv("AUDIO_QUEUE_MAX_CHUNKS", "128")),
        ),
        vad=VadConfig(
            threshold=float(os.getenv("VAD_THRESHOLD", "0.5")),
            min_silence_duration_ms=int(os.getenv("VAD_MIN_SILENCE_MS", "600")),
            speech_pad_ms=int(os.getenv("VAD_SPEECH_PAD_MS", "200")),
            min_utterance_duration_ms=int(os.getenv("VAD_MIN_UTTERANCE_MS", "450")),
            max_utterance_seconds=float(os.getenv("VAD_MAX_UTTERANCE_SECONDS", "20")),
        ),
        stt=WhisperSTTConfig(
            model_size=os.getenv("WHISPER_MODEL_SIZE", "small"),
            device=os.getenv("WHISPER_DEVICE", "cpu"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            cpu_threads=int(os.getenv("WHISPER_CPU_THREADS", "4")),
            num_workers=int(os.getenv("WHISPER_NUM_WORKERS", "1")),
            language=_get_optional_str(os.getenv("WHISPER_LANGUAGE")),
            beam_size=int(os.getenv("WHISPER_BEAM_SIZE", "5")),
            best_of=int(os.getenv("WHISPER_BEST_OF", "5")),
            no_speech_threshold=float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.6")),
            log_prob_threshold=float(os.getenv("WHISPER_LOG_PROB_THRESHOLD", "-0.8")),
            compression_ratio_threshold=float(os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.2")),
            min_peak_level=float(os.getenv("WHISPER_MIN_PEAK_LEVEL", "0.01")),
            min_rms_level=float(os.getenv("WHISPER_MIN_RMS_LEVEL", "0.0015")),
            normalize_audio=_get_bool(os.getenv("WHISPER_NORMALIZE_AUDIO"), True),
            local_files_only=_get_bool(os.getenv("WHISPER_LOCAL_FILES_ONLY"), False),
            download_root=_get_optional_str(os.getenv("WHISPER_DOWNLOAD_ROOT")),
            initial_prompt=_get_optional_str(os.getenv("WHISPER_INITIAL_PROMPT")),
        ),
    )
