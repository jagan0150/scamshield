"""ScamShield API.

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import csv
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import engine, voice
from .whatsapp import router as whatsapp_router

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
REPORTS = ROOT.parent / "data" / "reports.csv"

app = FastAPI(title="ScamShield", version="0.1.0")
app.include_router(whatsapp_router)


class CheckRequest(BaseModel):
    text: str = Field(..., max_length=4000)
    voice: bool = True  # also return a Tamil voice reply


class ReportRequest(BaseModel):
    text: str = Field(..., max_length=4000)
    label: str  # "scam" | "safe"
    consent: bool  # must be True: the person agrees to share the anonymised text


def _with_audio(result: dict, want_voice: bool) -> dict:
    result["audio_url"] = None
    if want_voice:
        try:
            result["audio_url"] = "/audio/" + voice.synthesize(result["spoken_ta"])
        except Exception as exc:  # gTTS missing / offline -> browser speech fallback
            result["audio_error"] = type(exc).__name__
    return result


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": engine.load_model() is not None}


@app.post("/api/check")
def check_text(req: CheckRequest):
    return _with_audio(engine.check(req.text), req.voice)


@app.post("/api/check-voice")
def check_voice(file: UploadFile = File(...)):
    """Voice note -> Whisper (Tamil) -> same pipeline."""
    suffix = Path(file.filename or "note.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        transcript = voice.transcribe(tmp_path)
    except ImportError:
        raise HTTPException(501, "Whisper is not installed. Run: pip install openai-whisper (needs ffmpeg).")
    finally:
        Path(tmp_path).unlink(missing_ok=True)  # audio is never kept
    result = _with_audio(engine.check(transcript), True)
    result["transcript"] = transcript
    return result


@app.get("/audio/{name}")
def get_audio(name: str):
    if not re.fullmatch(r"[a-f0-9]{16}\.mp3", name):
        raise HTTPException(404)
    path = voice.AUDIO_DIR / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/mpeg")


@app.post("/api/report")
def report(req: ReportRequest):
    """Feedback loop. Stores ONLY anonymised text, and ONLY with explicit consent."""
    if not req.consent:
        raise HTTPException(400, "Consent is required to store a report.")
    if req.label not in ("scam", "safe"):
        raise HTTPException(400, "label must be 'scam' or 'safe'")
    REPORTS.parent.mkdir(parents=True, exist_ok=True)
    new_file = not REPORTS.exists()
    with REPORTS.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "label", "text"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), req.label, engine.anonymise(req.text)])
    return {"stored": True}
