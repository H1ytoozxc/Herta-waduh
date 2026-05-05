import argparse
import asyncio
import logging
import time
from typing import Protocol

from actions.system_actions import SystemActionRunner, build_system_actions_instruction
from brain.memory import DialogueMemory
from config import AppConfig, load_config
from llm.deepseek_client import DeepSeekChatClient
from llm.google_ai_client import GoogleAIChatClient
from llm.google_live_client import GoogleLiveVoiceClient
from llm.ollama_client import OllamaChatClient
from persona.the_herta import (
    build_bootstrap_messages,
    build_conversational_hint,
    build_identity_reply,
    build_persona_polish_messages,
    build_persona_repair_messages,
    is_identity_query,
    needs_persona_repair,
)
from tts.edge_tts_engine import EdgeTTSEngine
from utils.logger import configure_logging


EXIT_COMMANDS = {"exit", "quit", "q", "выход"}
TTS_TEST_PHRASE = "Это Великая Герта. Проверка голосового вывода завершена."
GOOGLE_AI_PROVIDER_NAMES = {"google_ai", "google", "gemini"}


class ChatClient(Protocol):
    @property
    def last_warmup_error(self) -> str | None: ...

    def warm_up(self) -> bool: ...

    def chat(self, messages: list[dict[str, str]]) -> str: ...


class TTSEngine(Protocol):
    def speak(self, text: str) -> None: ...


class STTEngine(Protocol):
    @property
    def active_device(self) -> str: ...

    def transcribe(self, audio) -> str: ...


class FallbackSTTEngine:
    def __init__(self, primary: STTEngine, fallback_factory, logger: logging.Logger) -> None:
        self.primary = primary
        self.fallback_factory = fallback_factory
        self.logger = logger
        self._fallback: STTEngine | None = None

    @property
    def active_device(self) -> str:
        fallback_suffix = f"+{self._fallback.active_device}" if self._fallback is not None else "+fallback"
        return f"{self.primary.active_device}{fallback_suffix}"

    def _get_fallback(self) -> STTEngine:
        if self._fallback is None:
            self.logger.warning("Loading fallback Whisper STT after primary STT failure.")
            self._fallback = self.fallback_factory()
        return self._fallback

    def transcribe(self, audio) -> str:
        try:
            return self.primary.transcribe(audio)
        except Exception as exc:
            self.logger.warning("Primary STT failed, trying fallback Whisper: %s", exc)
            return self._get_fallback().transcribe(audio)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal local voice assistant scaffold powered by Ollama.",
    )
    parser.add_argument(
        "--text",
        help="Run a single prompt and exit instead of starting interactive mode.",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Run microphone -> VAD -> STT -> LLM -> TTS loop.",
    )
    parser.add_argument(
        "--live-voice",
        action="store_true",
        help="Run Google Live API native audio loop. Bypasses local Whisper and TTS.",
    )
    parser.add_argument(
        "--tts-test",
        action="store_true",
        help="Play a short TTS test phrase and exit.",
    )
    parser.add_argument(
        "--output-test",
        action="store_true",
        help="Play a short sine-wave tone through the configured output device and exit.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available input audio devices and exit.",
    )
    parser.add_argument(
        "--list-output-devices",
        action="store_true",
        help="List available output audio devices and exit.",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable speech output for this run.",
    )
    return parser



def trim_history(
    messages: list[dict[str, str]],
    max_history_messages: int,
    locked_prefix_count: int,
) -> list[dict[str, str]]:
    if len(messages) <= locked_prefix_count:
        return messages

    locked_prefix = messages[:locked_prefix_count]
    history = messages[locked_prefix_count:]
    trimmed_history = history[-max_history_messages:]
    return [*locked_prefix, *trimmed_history]



def build_inference_messages(user_text: str, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    hint = build_conversational_hint(user_text)
    if hint is None or not messages:
        return messages

    return [*messages[:-1], {"role": "system", "content": hint}, messages[-1]]



def generate_assistant_reply(
    *,
    user_text: str,
    messages: list[dict[str, str]],
    chat_client: ChatClient,
    config: AppConfig,
) -> str:
    if is_identity_query(user_text):
        return build_identity_reply(user_text)

    inference_messages = build_inference_messages(user_text, messages)
    draft_reply = chat_client.chat(inference_messages)

    if needs_persona_repair(draft_reply):
        repaired_messages = build_persona_repair_messages(user_text, draft_reply)
        repaired_reply = chat_client.chat(repaired_messages).strip()
        return repaired_reply or draft_reply

    if not config.persona_rewrite_enabled:
        return draft_reply

    polished_messages = build_persona_polish_messages(user_text, draft_reply)
    polished_reply = chat_client.chat(polished_messages).strip()
    return polished_reply or draft_reply



def run_turn(
    *,
    user_text: str,
    messages: list[dict[str, str]],
    chat_client: ChatClient,
    tts_engine: TTSEngine | None,
    config: AppConfig,
    logger: logging.Logger,
    locked_prefix_count: int,
    memory_store: DialogueMemory | None,
    system_action_runner: SystemActionRunner | None,
) -> str:
    messages.append({"role": "user", "content": user_text})

    action_result = system_action_runner.handle(user_text) if system_action_runner is not None else None
    if action_result is not None:
        assistant_reply = action_result.message
        logger.info("System action handled: action=%s, executed=%s.", action_result.action_name, action_result.executed)
        messages.append({"role": "assistant", "content": assistant_reply})
        if memory_store is not None:
            try:
                memory_store.append_turn(user_text, assistant_reply)
            except Exception as exc:
                logger.warning("Failed to save dialogue memory: %s", exc)

        messages[:] = trim_history(messages, config.max_history_messages, locked_prefix_count)

        if tts_engine is not None:
            try:
                tts_engine.speak(assistant_reply)
            except Exception as exc:  # pragma: no cover - depends on local audio/network state
                logger.warning("TTS playback failed: %s", exc)

        return assistant_reply

    logger.info(
        "Generating reply with %s model '%s'...",
        config.llm_provider,
        _selected_model_name(config),
    )
    started_at = time.perf_counter()
    assistant_reply = generate_assistant_reply(
        user_text=user_text,
        messages=messages,
        chat_client=chat_client,
        config=config,
    )
    elapsed_seconds = time.perf_counter() - started_at
    logger.info("Assistant reply ready in %.1fs.", elapsed_seconds)

    messages.append({"role": "assistant", "content": assistant_reply})
    if memory_store is not None:
        try:
            memory_store.append_turn(user_text, assistant_reply)
        except Exception as exc:
            logger.warning("Failed to save dialogue memory: %s", exc)

    messages[:] = trim_history(messages, config.max_history_messages, locked_prefix_count)

    if tts_engine is not None:
        try:
            tts_engine.speak(assistant_reply)
        except Exception as exc:  # pragma: no cover - depends on local audio/network state
            logger.warning("TTS playback failed: %s", exc)

    return assistant_reply



def interactive_loop(
    *,
    messages: list[dict[str, str]],
    chat_client: ChatClient,
    tts_engine: TTSEngine | None,
    config: AppConfig,
    logger: logging.Logger,
    locked_prefix_count: int,
    memory_store: DialogueMemory | None,
    system_action_runner: SystemActionRunner | None,
) -> None:
    print("The Herta assistant ready. Type a message or 'exit' to quit.")

    while True:
        try:
            user_text = input("You> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_text:
            continue

        if user_text.lower() in EXIT_COMMANDS:
            break

        try:
            assistant_reply = run_turn(
                user_text=user_text,
                messages=messages,
                chat_client=chat_client,
                tts_engine=tts_engine,
                config=config,
                logger=logger,
                locked_prefix_count=locked_prefix_count,
                memory_store=memory_store,
                system_action_runner=system_action_runner,
            )
        except Exception as exc:
            logger.error("Assistant turn failed: %s", exc)
            continue

        print(f"The Herta> {assistant_reply}")


def _prepare_stt_engine(config: AppConfig, logger: logging.Logger) -> STTEngine:
    provider = config.stt_provider.strip().lower()

    if provider in {"whisper", "faster_whisper", "faster-whisper"}:
        from stt.whisper_stt import FasterWhisperSTT

        print(
            f"Preparing voice mode. Loading Whisper model '{config.stt.model_size}' on device '{config.stt.device}'. "
            "On first run this may download files and take a few minutes."
        )
        if config.stt.model_size == "tiny":
            print("Warning: Whisper 'tiny' is fast, but its Russian STT quality is limited. Prefer 'small' for better accuracy.")
        return FasterWhisperSTT(config.stt)

    if provider in {"google_ai", "google", "gemini"}:
        from stt.google_ai_stt import GoogleAITranscriptionSTT
        from stt.whisper_stt import FasterWhisperSTT

        print(
            f"Preparing Google AI STT model '{config.google_stt.model}'. "
            "This sends each detected utterance to Google for transcription."
        )
        primary = GoogleAITranscriptionSTT(
            config.google_stt,
            audio_gate_config=config.stt,
            sample_rate=config.audio.sample_rate,
        )
        if not config.google_stt.fallback_to_whisper:
            return primary

        return FallbackSTTEngine(
            primary,
            fallback_factory=lambda: FasterWhisperSTT(config.stt),
            logger=logger,
        )

    raise ValueError("Unsupported STT_PROVIDER. Use 'whisper' or 'google_ai'.")



def voice_loop(
    *,
    messages: list[dict[str, str]],
    chat_client: ChatClient,
    tts_engine: TTSEngine | None,
    config: AppConfig,
    logger: logging.Logger,
    locked_prefix_count: int,
    memory_store: DialogueMemory | None,
    system_action_runner: SystemActionRunner | None,
) -> None:
    from audio.input import MicrophoneInput
    from audio.vad import StreamingVADSegmenter

    provider_name = config.llm_provider
    model_name = _selected_model_name(config)

    print(f"Preparing {provider_name} model '{model_name}'. This may take a few seconds.")
    warmed_up = chat_client.warm_up()
    if warmed_up:
        print(f"{provider_name} model ready.")
    else:
        warmup_error = chat_client.last_warmup_error
        warmup_error_suffix = f": {warmup_error}" if warmup_error else ""
        logger.info(
            "%s model is not ready%s. The first reply may take longer.",
            provider_name,
            warmup_error_suffix,
        )

    try:
        stt_engine = _prepare_stt_engine(config, logger)
    except KeyboardInterrupt:
        print("\nSTT model loading interrupted.")
        return

    print(f"STT ready. provider={config.stt_provider}, active_device={stt_engine.active_device}.")

    microphone = MicrophoneInput(config.audio)
    vad_segmenter = StreamingVADSegmenter(config.audio, config.vad)

    print(
        f"Voice mode ready. device={config.audio.device!r}, sample_rate={config.audio.sample_rate}, "
        f"block_size={config.audio.block_size}. Speak into the microphone. Press Ctrl+C to stop."
    )

    with microphone:
        while True:
            try:
                chunk = microphone.read_chunk(timeout=1.0)
            except (KeyboardInterrupt, EOFError):
                print()
                break

            if chunk is None:
                continue

            utterance = vad_segmenter.process_chunk(chunk)
            if utterance is None:
                continue

            microphone.clear_queue()

            try:
                transcript = stt_engine.transcribe(utterance)
            except Exception as exc:
                logger.error("STT failed: %s", exc)
                continue

            if not transcript:
                continue

            print(f"You> {transcript}")

            try:
                assistant_reply = run_turn(
                    user_text=transcript,
                    messages=messages,
                    chat_client=chat_client,
                    tts_engine=tts_engine,
                    config=config,
                    logger=logger,
                    locked_prefix_count=locked_prefix_count,
                    memory_store=memory_store,
                    system_action_runner=system_action_runner,
                )
            except Exception as exc:
                logger.error("Assistant turn failed: %s", exc)
                continue

            print(f"The Herta> {assistant_reply}")


def google_live_voice_loop(
    *,
    messages: list[dict[str, str]],
    config: AppConfig,
    logger: logging.Logger,
    memory_store: DialogueMemory | None,
    system_action_runner: SystemActionRunner | None,
    tts_engine: TTSEngine | None,
) -> None:
    client = GoogleLiveVoiceClient(config.google_ai)
    model_name = config.google_ai.live_model

    print(f"Preparing Google Live model '{model_name}'. This may take a few seconds.")
    warmed_up = client.warm_up()
    if warmed_up:
        print("Google Live model ready.")
    else:
        warmup_error = client.last_warmup_error
        warmup_error_suffix = f": {warmup_error}" if warmup_error else ""
        logger.info("Google Live model is not ready%s.", warmup_error_suffix)

    print(
        "Google Live voice mode ready. This bypasses local Whisper and local TTS. "
        "Speak into the microphone. Press Ctrl+C to stop."
    )
    print(
        f"Google Live output: device={_describe_output_device(config.audio_output.device)}, "
        f"sample_rate=24000, channels={config.audio_output.channels}."
    )
    print(f"Google Live voice preset: {config.google_ai.live_voice_name or 'auto'}.")
    if config.google_ai.live_playback == 'rvc':
        print("Google Live playback mode: RVC. Google audio is ignored; output transcript is spoken by local RVC.")
    else:
        print("Google Live playback mode: Google native audio.")
    asyncio.run(
        client.run_voice_loop(
            messages=messages,
            audio_config=config.audio,
            audio_output_config=config.audio_output,
            memory_store=memory_store,
            system_action_runner=(system_action_runner.handle if system_action_runner is not None else None),
            transcript_tts=tts_engine if config.google_ai.live_playback == 'rvc' else None,
        )
    )



def print_input_devices() -> int:
    from audio.input import list_input_devices

    devices = list_input_devices()
    if not devices:
        print("No input audio devices found.")
        return 1

    for description in devices:
        print(description)
    return 0



def print_output_devices() -> int:
    from audio.output import list_output_devices

    devices = list_output_devices()
    if not devices:
        print("No output audio devices found.")
        return 1

    for description in devices:
        print(description)
    return 0



def run_output_test(config: AppConfig, logger: logging.Logger) -> int:
    from audio.output import SpeakerOutput, describe_output_device

    print(f"Configured output device: {describe_output_device(config.audio_output.device)}")
    try:
        SpeakerOutput(config.audio_output).play_test_tone()
    except Exception as exc:
        logger.error("Output test failed: %s", exc)
        return 1

    print("Output tone test completed.")
    return 0



def _is_base_qwen3_model(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    return normalized.startswith("qwen3:") or normalized == "qwen3"



def _describe_output_device(device: int | str | None) -> str:
    try:
        from audio.output import describe_output_device

        return describe_output_device(device)
    except Exception:
        return "unknown"


def _selected_model_name(config: AppConfig) -> str:
    if config.llm_provider == "deepseek":
        return config.deepseek.model
    if config.llm_provider in GOOGLE_AI_PROVIDER_NAMES:
        return config.google_ai.model
    return config.ollama.model


def _build_chat_client(config: AppConfig) -> ChatClient:
    if config.llm_provider == "ollama":
        return OllamaChatClient(config.ollama)
    if config.llm_provider == "deepseek":
        return DeepSeekChatClient(config.deepseek)
    if config.llm_provider in GOOGLE_AI_PROVIDER_NAMES:
        return GoogleAIChatClient(config.google_ai)
    raise ValueError("Unsupported LLM_PROVIDER. Use 'ollama', 'deepseek', or 'google_ai'.")


def _build_tts_engine(config: AppConfig, *, no_tts: bool, live_voice: bool) -> TTSEngine | None:
    if no_tts:
        return None
    if live_voice and config.google_ai.live_playback != 'rvc':
        return None
    if live_voice and not config.rvc_tts.enabled:
        return None
    if config.rvc_tts.enabled:
        from tts.rvc_tts_engine import RvcTTSEngine

        return RvcTTSEngine(config.rvc_tts, config.audio_output)
    if not config.tts.enabled:
        return None
    return EdgeTTSEngine(config.tts, config.audio_output)


def _build_memory_store(config: AppConfig, logger: logging.Logger) -> DialogueMemory | None:
    if not config.memory.enabled:
        return None

    memory_store = DialogueMemory(config.memory)
    try:
        memory_store.load_context_messages()
    except Exception as exc:
        logger.warning("Dialogue memory is disabled for this run: %s", exc)
        return None

    logger.info("Dialogue memory ready. path=%s.", memory_store.path)
    return memory_store



def main() -> int:
    args = build_parser().parse_args()

    if args.voice and args.live_voice:
        print("Use either --voice or --live-voice, not both.")
        return 1

    if args.list_devices:
        return print_input_devices()

    if args.list_output_devices:
        return print_output_devices()

    config = load_config()
    configure_logging(config.log_level)
    logger = logging.getLogger("the_herta.main")

    if args.output_test:
        return run_output_test(config, logger)

    tts_engine = _build_tts_engine(config, no_tts=args.no_tts, live_voice=args.live_voice)
    live_uses_rvc_tts = args.live_voice and config.google_ai.live_playback == 'rvc'

    if live_uses_rvc_tts and not config.rvc_tts.enabled:
        print("GOOGLE_AI_LIVE_PLAYBACK='rvc' requires RVC_TTS_ENABLED='true'.")
        return 1

    if args.live_voice and not live_uses_rvc_tts:
        print("Local TTS disabled for Google Live native audio mode.")
    elif tts_engine is None:
        print("TTS disabled for this run.")
    elif config.rvc_tts.enabled:
        print(
            f"RVC TTS ready. model={config.rvc_tts.model_path!r}, pitch={config.rvc_tts.pitch}, "
            f"f0_method={config.rvc_tts.f0_method}, base_tts={config.rvc_tts.base_tts}, "
            f"backend={config.rvc_tts.backend}."
        )
        print(f"Configured output device: {_describe_output_device(config.audio_output.device)}")
        if config.rvc_tts.warm_up:
            print("Preparing RVC voice cache. First startup can take a bit; replies after that should be faster.")
            try:
                warm_up = getattr(tts_engine, "warm_up")
                warm_up()
            except Exception as exc:
                logger.warning("RVC TTS warm-up failed: %s", exc)
            else:
                print("RVC voice cache ready.")
    else:
        print(
            f"TTS ready. preferred_local={config.tts.prefer_local}, sapi_voice={config.tts.sapi_voice!r}, edge_voice={config.tts.voice}."
        )
        print(f"Configured output device: {_describe_output_device(config.audio_output.device)}")
        print(f"Piper model: {config.tts.piper_model_path!r}")

    system_action_runner = SystemActionRunner(config.system_actions, logger)
    if config.system_actions.enabled:
        print(
            "System actions enabled: browser, VS Code, and safe .txt creation only. "
            "Delete/move/overwrite requests are blocked; rename is limited to Herta-created items."
        )

    if args.tts_test:
        if tts_engine is None:
            print("TTS is disabled. Remove --no-tts or enable TTS in config.")
            return 1
        try:
            tts_engine.speak(TTS_TEST_PHRASE)
        except Exception as exc:
            logger.error("TTS test failed: %s", exc)
            return 1
        print(TTS_TEST_PHRASE)
        return 0

    if not args.live_voice and config.llm_provider == "ollama" and _is_base_qwen3_model(config.ollama.model):
        print(
            "Warning: base qwen3 models in Ollama are usable now, but they remain much slower than gemma in live voice mode. "
            "Use gemma for realtime speech, and qwen3 when you want higher-quality but slower answers."
        )

    selected_model_name = config.google_ai.live_model if args.live_voice else _selected_model_name(config)
    messages = build_bootstrap_messages(selected_model_name)
    if config.system_actions.enabled:
        messages.append({"role": "system", "content": build_system_actions_instruction()})
    locked_prefix_count = len(messages)
    memory_store = _build_memory_store(config, logger)
    if memory_store is not None:
        context_messages = memory_store.load_context_messages()
        messages.extend(context_messages)
        logger.info("Loaded %s messages from dialogue memory.", len(context_messages))

    if args.live_voice:
        try:
            google_live_voice_loop(
                messages=messages,
                config=config,
                logger=logger,
                memory_store=memory_store,
                system_action_runner=system_action_runner,
                tts_engine=tts_engine,
            )
        except KeyboardInterrupt:
            print()
        except Exception as exc:
            logger.error("Google Live voice loop failed: %s", exc)
            return 1
        return 0

    chat_client = _build_chat_client(config)

    if args.text:
        try:
            assistant_reply = run_turn(
                user_text=args.text,
                messages=messages,
                chat_client=chat_client,
                tts_engine=tts_engine,
                config=config,
                logger=logger,
                locked_prefix_count=locked_prefix_count,
                memory_store=memory_store,
                system_action_runner=system_action_runner,
            )
        except Exception as exc:
            logger.error("Assistant turn failed: %s", exc)
            return 1

        print(assistant_reply)
        return 0

    if args.voice:
        try:
            voice_loop(
                messages=messages,
                chat_client=chat_client,
                tts_engine=tts_engine,
                config=config,
                logger=logger,
                locked_prefix_count=locked_prefix_count,
                memory_store=memory_store,
                system_action_runner=system_action_runner,
            )
        except Exception as exc:
            logger.error("Voice loop failed: %s", exc)
            return 1
        return 0

    interactive_loop(
        messages=messages,
        chat_client=chat_client,
        tts_engine=tts_engine,
        config=config,
        logger=logger,
        locked_prefix_count=locked_prefix_count,
        memory_store=memory_store,
        system_action_runner=system_action_runner,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
