import argparse
import logging

from config import AppConfig, load_config
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
        "--list-devices",
        action="store_true",
        help="List available input audio devices and exit.",
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
    chat_client: OllamaChatClient,
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
    chat_client: OllamaChatClient,
    tts_engine: EdgeTTSEngine | None,
    config: AppConfig,
    logger: logging.Logger,
    locked_prefix_count: int,
) -> str:
    messages.append({"role": "user", "content": user_text})
    assistant_reply = generate_assistant_reply(
        user_text=user_text,
        messages=messages,
        chat_client=chat_client,
        config=config,
    )
    messages.append({"role": "assistant", "content": assistant_reply})

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
    chat_client: OllamaChatClient,
    tts_engine: EdgeTTSEngine | None,
    config: AppConfig,
    logger: logging.Logger,
    locked_prefix_count: int,
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
            )
        except Exception as exc:
            logger.error("Assistant turn failed: %s", exc)
            continue

        print(f"The Herta> {assistant_reply}")



def voice_loop(
    *,
    messages: list[dict[str, str]],
    chat_client: OllamaChatClient,
    tts_engine: EdgeTTSEngine | None,
    config: AppConfig,
    logger: logging.Logger,
    locked_prefix_count: int,
) -> None:
    from audio.input import MicrophoneInput
    from audio.vad import StreamingVADSegmenter
    from stt.whisper_stt import FasterWhisperSTT

    print(
        f"Preparing voice mode. Loading Whisper model '{config.stt.model_size}' on device '{config.stt.device}'. "
        "On first run this may download files and take a few minutes."
    )

    if config.stt.model_size == "tiny":
        print("Warning: Whisper 'tiny' is fast, but its Russian STT quality is limited. Prefer 'small' for better accuracy.")

    try:
        stt_engine = FasterWhisperSTT(config.stt)
    except KeyboardInterrupt:
        print("\nSTT model loading interrupted.")
        return

    print(
        f"STT model ready. active_device={stt_engine.active_device}, beam_size={config.stt.beam_size}, "
        f"language={config.stt.language!r}."
    )

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
                )
            except Exception as exc:
                logger.error("Assistant turn failed: %s", exc)
                continue

            print(f"The Herta> {assistant_reply}")



def print_audio_devices() -> int:
    from audio.input import list_input_devices

    devices = list_input_devices()
    if not devices:
        print("No input audio devices found.")
        return 1

    for description in devices:
        print(description)
    return 0



def main() -> int:
    args = build_parser().parse_args()

    if args.list_devices:
        return print_audio_devices()

    config = load_config()
    configure_logging(config.log_level)
    logger = logging.getLogger("the_herta.main")

    messages = build_bootstrap_messages()
    locked_prefix_count = len(messages)
    chat_client = OllamaChatClient(config.ollama)
    tts_engine = None if args.no_tts or not config.tts.enabled else EdgeTTSEngine(config.tts)

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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
