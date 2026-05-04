import asyncio
import base64
import subprocess
import tempfile
from pathlib import Path

import edge_tts
import numpy as np
from edge_playback.win32_playback import play_mp3_win32
from piper import PiperVoice

from audio.output import SpeakerOutput
from config import AudioOutputConfig, EdgeTTSConfig


WINDOWS_POWERSHELL = r'C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe'


class EdgeTTSEngine:
    def __init__(self, config: EdgeTTSConfig, output_config: AudioOutputConfig) -> None:
        self.config = config
        self.output = SpeakerOutput(output_config)
        self._piper_voice: PiperVoice | None = None

    async def _save_to_file(self, text: str, target_path: Path) -> None:
        communicator = edge_tts.Communicate(
            text=text,
            voice=self.config.voice,
            rate=self.config.rate,
            volume=self.config.volume,
            pitch=self.config.pitch,
        )
        await communicator.save(str(target_path))

    def _resolve_path(self, raw_path: str | None) -> Path | None:
        if not raw_path:
            return None
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        return Path.cwd() / candidate

    def _get_piper_voice(self) -> PiperVoice:
        if self._piper_voice is not None:
            return self._piper_voice

        model_path = self._resolve_path(self.config.piper_model_path)
        if model_path is None:
            raise RuntimeError('Piper model path is not configured.')
        if not model_path.exists():
            raise FileNotFoundError(f'Piper model not found: {model_path}')

        config_path = self._resolve_path(self.config.piper_config_path)
        self._piper_voice = PiperVoice.load(
            model_path,
            config_path=config_path,
            use_cuda=self.config.piper_use_cuda,
        )
        return self._piper_voice

    def _speak_with_piper(self, text: str) -> None:
        voice = self._get_piper_voice()
        audio_chunks = list(voice.synthesize(text))
        if not audio_chunks:
            raise RuntimeError('Piper returned no audio chunks.')

        audio_arrays = [chunk.audio_float_array for chunk in audio_chunks if chunk.audio_float_array.size > 0]
        if not audio_arrays:
            raise RuntimeError('Piper returned empty audio data.')

        merged_audio = np.concatenate(audio_arrays, axis=0).astype(np.float32, copy=False)
        self.output.play_audio(merged_audio, sample_rate=audio_chunks[0].sample_rate)

    def _build_sapi_script(self, text: str) -> str:
        encoded_text = base64.b64encode(text.encode('utf-8')).decode('ascii')
        desired_voice = (self.config.sapi_voice or '').replace("'", "''")
        return f"""
$voice = New-Object -ComObject SAPI.SpVoice
$voice.Rate = {self.config.sapi_rate}
$voice.Volume = {self.config.sapi_volume}
$desiredVoice = '{desired_voice}'
if ($desiredVoice) {{
    $match = $voice.GetVoices() | Where-Object {{ $_.GetDescription() -eq $desiredVoice }} | Select-Object -First 1
    if ($null -ne $match) {{
        $voice.Voice = $match
    }}
}}
$bytes = [Convert]::FromBase64String('{encoded_text}')
$text = [System.Text.Encoding]::UTF8.GetString($bytes)
[void]$voice.Speak($text)
""".strip()

    def _speak_with_sapi(self, text: str) -> None:
        script = self._build_sapi_script(text)
        subprocess.run(
            [WINDOWS_POWERSHELL, '-STA', '-NoProfile', '-NonInteractive', '-Command', '-'],
            input=script,
            text=True,
            encoding='utf-8',
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _speak_with_edge(self, text: str) -> None:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            mp3_path = Path(tmp_file.name)

        try:
            asyncio.run(self._save_to_file(text, mp3_path))
            play_mp3_win32(str(mp3_path))
        finally:
            mp3_path.unlink(missing_ok=True)

    def speak(self, text: str) -> None:
        normalized_text = text.strip()
        if not normalized_text:
            return

        attempts: list[tuple[str, callable]] = []
        piper_path = self._resolve_path(self.config.piper_model_path)
        if piper_path is not None:
            attempts.append(('piper', self._speak_with_piper))

        if self.config.prefer_local:
            attempts.extend([
                ('sapi', self._speak_with_sapi),
                ('edge', self._speak_with_edge),
            ])
        else:
            attempts.extend([
                ('edge', self._speak_with_edge),
                ('sapi', self._speak_with_sapi),
            ])

        errors: list[str] = []
        for backend_name, backend in attempts:
            try:
                backend(normalized_text)
                return
            except Exception as exc:
                errors.append(f'{backend_name}={exc}')

        raise RuntimeError('All TTS backends failed. ' + '; '.join(errors))
