import asyncio
import tempfile
from pathlib import Path

import edge_tts
from edge_playback.win32_playback import play_mp3_win32

from config import EdgeTTSConfig


class EdgeTTSEngine:
    def __init__(self, config: EdgeTTSConfig) -> None:
        self.config = config

    async def _save_to_file(self, text: str, target_path: Path) -> None:
        communicator = edge_tts.Communicate(
            text=text,
            voice=self.config.voice,
            rate=self.config.rate,
            volume=self.config.volume,
            pitch=self.config.pitch,
        )
        await communicator.save(str(target_path))

    def speak(self, text: str) -> None:
        normalized_text = text.strip()
        if not normalized_text:
            return

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            mp3_path = Path(tmp_file.name)

        try:
            asyncio.run(self._save_to_file(normalized_text, mp3_path))
            play_mp3_win32(str(mp3_path))
        finally:
            mp3_path.unlink(missing_ok=True)
