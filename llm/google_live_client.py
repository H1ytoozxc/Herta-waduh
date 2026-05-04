import asyncio
import base64
import logging
from typing import Any

from audio.live import LIVE_INPUT_SAMPLE_RATE, LiveAudioOutput, LiveMicrophoneInput
from brain.memory import DialogueMemory
from config import AudioInputConfig, AudioOutputConfig, GoogleAIConfig


logger = logging.getLogger(__name__)


class GoogleLiveVoiceClient:
    def __init__(self, config: GoogleAIConfig) -> None:
        self.config = config
        self._last_warmup_error: str | None = None

    @property
    def last_warmup_error(self) -> str | None:
        return self._last_warmup_error

    def _validate_api_key(self) -> None:
        if not self.config.api_key:
            raise RuntimeError('GOOGLE_AI_API_KEY is not configured.')

    def warm_up(self) -> bool:
        try:
            self._validate_api_key()
            self._last_warmup_error = None
            return True
        except RuntimeError as exc:
            self._last_warmup_error = str(exc)
            logger.warning("Google Live warm-up failed: %s", exc)
            return False

    def _api_version(self) -> str:
        if self.config.live_affective_dialog or self.config.live_proactive_audio:
            return 'v1alpha'
        return self.config.live_api_version

    def _build_live_config(self) -> dict[str, Any]:
        live_config: dict[str, Any] = {
            'response_modalities': ['AUDIO'],
        }

        if self.config.live_input_transcription:
            live_config['input_audio_transcription'] = {}
        if self.config.live_output_transcription:
            live_config['output_audio_transcription'] = {}
        if self.config.live_voice_name:
            live_config['speech_config'] = {
                'voice_config': {'prebuilt_voice_config': {'voice_name': self.config.live_voice_name}}
            }

        thinking_config: dict[str, Any] = {}
        if self.config.live_thinking_level and self.config.live_model.startswith('gemini-3'):
            thinking_config['thinking_level'] = self.config.live_thinking_level
        if self.config.live_thinking_budget is not None and not self.config.live_model.startswith('gemini-3'):
            thinking_config['thinking_budget'] = self.config.live_thinking_budget
        if thinking_config:
            live_config['thinking_config'] = thinking_config

        if self.config.live_affective_dialog:
            live_config['enable_affective_dialog'] = True
        if self.config.live_proactive_audio:
            live_config['proactivity'] = {'proactive_audio': True}
        if self.config.live_model.startswith('gemini-3'):
            live_config['history_config'] = {'initial_history_in_client_content': True}

        return live_config

    def _to_live_turns(self, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        turns: list[dict[str, Any]] = []
        system_texts: list[str] = []

        for message in messages:
            text = str(message.get('content', '')).strip()
            if not text:
                continue

            role = message.get('role', 'user')
            if role == 'system':
                system_texts.append(text)
                continue

            live_role = 'model' if role == 'assistant' else 'user'
            turns.append({'role': live_role, 'parts': [{'text': text}]})

        if system_texts:
            joined_system_text = '\n\n'.join(system_texts)
            instruction_text = (
                'Instructions for the assistant. Follow them throughout this live voice conversation:\n'
                f'{joined_system_text}'
            )
            turns.insert(0, {'role': 'user', 'parts': [{'text': instruction_text}]})

        return turns

    async def _seed_context(self, session: Any, messages: list[dict[str, str]]) -> None:
        turns = self._to_live_turns(messages)
        if not turns:
            return
        await session.send_client_content(turns=turns, turn_complete=False)

    async def _send_microphone_audio(self, session: Any, microphone: LiveMicrophoneInput, types: Any) -> None:
        while True:
            chunk = await asyncio.to_thread(microphone.read_chunk, 0.25)
            if not chunk:
                continue
            await session.send_realtime_input(
                audio=types.Blob(
                    data=chunk,
                    mime_type=f'audio/pcm;rate={LIVE_INPUT_SAMPLE_RATE}',
                )
            )

    def _decode_audio_data(self, raw_data: bytes | str) -> bytes:
        if isinstance(raw_data, str):
            return base64.b64decode(raw_data)
        return bytes(raw_data)

    def _append_transcription(self, chunks: list[str], text: str | None) -> None:
        if text:
            chunks.append(text)

    def _print_and_save_turn(
        self,
        input_chunks: list[str],
        output_chunks: list[str],
        memory_store: DialogueMemory | None,
    ) -> None:
        user_text = ''.join(input_chunks).strip()
        assistant_text = ''.join(output_chunks).strip()
        input_chunks.clear()
        output_chunks.clear()

        if user_text:
            print(f"You> {user_text}")
        if assistant_text:
            print(f"The Herta> {assistant_text}")
        if user_text and assistant_text and memory_store is not None:
            try:
                memory_store.append_turn(user_text, assistant_text)
            except Exception as exc:
                logger.warning("Failed to save dialogue memory: %s", exc)

    async def _receive_model_audio(
        self,
        session: Any,
        output: LiveAudioOutput,
        memory_store: DialogueMemory | None,
    ) -> None:
        input_chunks: list[str] = []
        output_chunks: list[str] = []
        audio_chunk_count = 0
        audio_byte_count = 0

        while True:
            async for response in session.receive():
                raw_audio = getattr(response, 'data', None)
                wrote_response_audio = raw_audio is not None
                if raw_audio is not None:
                    audio_data = self._decode_audio_data(raw_audio)
                    audio_chunk_count += 1
                    audio_byte_count += len(audio_data)
                    output.write_pcm16_mono(audio_data)

                text = getattr(response, 'text', None)
                self._append_transcription(output_chunks, text)

                server_content = getattr(response, 'server_content', None)
                if server_content is None:
                    continue

                input_transcription = getattr(server_content, 'input_transcription', None)
                self._append_transcription(input_chunks, getattr(input_transcription, 'text', None))

                output_transcription = getattr(server_content, 'output_transcription', None)
                self._append_transcription(output_chunks, getattr(output_transcription, 'text', None))

                model_turn = getattr(server_content, 'model_turn', None)
                parts = getattr(model_turn, 'parts', None) or []
                for part in parts:
                    inline_data = getattr(part, 'inline_data', None)
                    if inline_data is not None and not wrote_response_audio:
                        audio_data = self._decode_audio_data(inline_data.data)
                        audio_chunk_count += 1
                        audio_byte_count += len(audio_data)
                        output.write_pcm16_mono(audio_data)

                    part_text = getattr(part, 'text', None)
                    self._append_transcription(output_chunks, part_text)

                if getattr(server_content, 'turn_complete', False):
                    if audio_chunk_count > 0:
                        logger.info(
                            "Google Live audio received: chunks=%s, bytes=%s.",
                            audio_chunk_count,
                            audio_byte_count,
                        )
                    else:
                        logger.warning(
                            "Google Live turn completed without audio chunks. Check live model and response_modalities."
                        )
                    self._print_and_save_turn(input_chunks, output_chunks, memory_store)
                    audio_chunk_count = 0
                    audio_byte_count = 0

    async def run_voice_loop(
        self,
        *,
        messages: list[dict[str, str]],
        audio_config: AudioInputConfig,
        audio_output_config: AudioOutputConfig,
        memory_store: DialogueMemory | None,
    ) -> None:
        self._validate_api_key()

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Google Live API requires the 'google-genai' package. Run: python -m pip install google-genai"
            ) from exc

        client = genai.Client(
            api_key=self.config.api_key,
            http_options={'api_version': self._api_version()},
        )

        with LiveMicrophoneInput(audio_config) as microphone, LiveAudioOutput(audio_output_config) as output:
            async with client.aio.live.connect(
                model=self.config.live_model,
                config=self._build_live_config(),
            ) as session:
                await self._seed_context(session, messages)

                sender_task = asyncio.create_task(self._send_microphone_audio(session, microphone, types))
                receiver_task = asyncio.create_task(self._receive_model_audio(session, output, memory_store))
                done, pending = await asyncio.wait(
                    {sender_task, receiver_task},
                    return_when=asyncio.FIRST_EXCEPTION,
                )

                for task in pending:
                    task.cancel()
                for task in done:
                    task.result()
