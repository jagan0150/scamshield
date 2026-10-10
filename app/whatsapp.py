"""WhatsApp bot via the Twilio WhatsApp Sandbox (free for demos).

Setup (see README):
  1. Twilio console -> Messaging -> Try it out -> WhatsApp sandbox.
  2. Expose the app:  ngrok http 8000      (gives https://xxxx.ngrok-free.app)
  3. Set the sandbox "When a message comes in" URL to  https://xxxx.ngrok-free.app/whatsapp
  4. export PUBLIC_BASE_URL=https://xxxx.ngrok-free.app   (so voice replies can be fetched)
     export TWILIO_ACCOUNT_SID=...  TWILIO_AUTH_TOKEN=...  (only needed for incoming voice notes)

STATUS: written but not tested against live Twilio. Test it before the demo.
"""
from __future__ import annotations

import base64
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

from fastapi import APIRouter, Form
from fastapi.responses import Response

from . import engine, voice

router = APIRouter()

ICON = {"scam": "🚨", "risky": "⚠️", "safe": "✅"}


def _download_media(url: str) -> str:
    req = urllib.request.Request(url)
    sid, token = os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN")
    if sid and token:
        cred = base64.b64encode(f"{sid}:{token}".encode()).decode()
        req.add_header("Authorization", f"Basic {cred}")
    with urllib.request.urlopen(req, timeout=30) as resp, tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(resp.read())
        return tmp.name


def _twiml(body: str, media_url: Optional[str] = None) -> Response:
    media = f"<Media>{escape(media_url)}</Media>" if media_url else ""
    xml = f"<?xml version='1.0' encoding='UTF-8'?><Response><Message><Body>{escape(body)}</Body>{media}</Message></Response>"
    return Response(content=xml, media_type="application/xml")


@router.post("/whatsapp")
def whatsapp(
    Body: str = Form(""),
    MediaUrl0: Optional[str] = Form(None),
    MediaContentType0: Optional[str] = Form(None),
):
    text = Body
    if MediaUrl0 and (MediaContentType0 or "").startswith("audio"):
        path = None
        try:
            path = _download_media(MediaUrl0)
            text = voice.transcribe(path)
        except Exception:
            return _twiml("குரல் செய்தியை புரிந்துகொள்ள முடியவில்லை. தயவுசெய்து செய்தியை எழுதி அனுப்புங்கள்.")
        finally:
            if path:
                Path(path).unlink(missing_ok=True)  # voice notes are never kept

    if not text.strip():
        return _twiml("சந்தேகமான SMS, லிங்க் அல்லது குரல் செய்தியை இங்கே அனுப்புங்கள். நான் சரிபார்க்கிறேன்.")

    r = engine.check(text)
    lines = [f"{ICON[r['verdict']]} {r['title_ta']}"]
    lines += [f"• {x['ta']}" for x in r["reasons"][:3]]
    lines.append(r["advice_ta"])

    media_url = None
    base = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    if base:
        try:
            media_url = f"{base}/audio/{voice.synthesize(r['spoken_ta'])}"
        except Exception:
            media_url = None  # text reply still goes out
    return _twiml("\n".join(lines), media_url)
