"""Tamil voice in/out.

* Text -> speech: gTTS (needs internet).  Audio is cached by content hash.
* Speech -> text: OpenAI Whisper (optional, heavy; needs ffmpeg).

Both imports are lazy so the rest of the app runs without them.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

AUDIO_DIR = Path(os.getenv("SCAMSHIELD_AUDIO_DIR", Path(__file__).resolve().parent.parent / "audio_cache"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # use "medium" for better Tamil if your PC can take it

_whisper = None


def synthesize(text: str) -> str:
    """Return the cached mp3 file name for `text`, creating it if needed."""
    from gtts import gTTS  # lazy import

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] + ".mp3"
    path = AUDIO_DIR / name
    if not path.exists():
        gTTS(text=text, lang="ta").save(str(path))
    return name


def transcribe(audio_path: str) -> str:
    """Tamil speech-to-text with Whisper."""
    global _whisper
    import whisper  # lazy import

    if _whisper is None:
        _whisper = whisper.load_model(WHISPER_MODEL)
    result = _whisper.transcribe(audio_path, language="ta", fp16=False)
    return (result.get("text") or "").strip()
