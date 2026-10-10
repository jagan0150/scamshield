# ScamShield

Tamil voice-based fraud alert for rural and elderly UPI users. Paste (or speak) a suspicious
SMS / link / voice note and get a **Safe / Risky / Scam** verdict, the reasons in Tamil,
a spoken Tamil reply, and the **1930** cybercrime helpline.

Team ELITE KINGS - Hack for Social Cause (HSC|TN|00306)

## How it works

```
message / link / voice note
   -> (voice note) Whisper speech-to-text, Tamil
   -> risk engine = scam-pattern rules + URL checks + ML classifier
   -> verdict (Safe / Risky / Scam) + Tamil reasons
   -> Tamil voice reply (gTTS) + 1930 helpline
```

* `app/engine.py` - the risk engine. Rules cover OTP/PIN requests, fake collect requests,
  remote-access apps, KYC threats, electricity-disconnection threats, "digital arrest",
  lottery/loan/job bait, family impersonation. URL checks flag shorteners, odd TLDs, IP links,
  look-alike bank domains. Rules work in English, Tamil script and Tanglish.
* `app/train.py` - TF-IDF (character n-grams) + logistic regression. Character n-grams handle
  Tamil script and Tanglish spellings without a tokenizer.
* Genuine messages that say "Do not share your OTP" are NOT flagged (those sentences are
  stripped before the OTP-request rule runs).
* A message with no explainable reason can never be labelled "Scam" by the ML model alone.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m app.train                                     # creates app/model.joblib
uvicorn app.main:app --reload
```

Open http://localhost:8000 - use the sample chips for a quick demo.
Without `model.joblib` the engine still works in rules-only mode.

Tamil voice reply needs internet (gTTS). If it is unavailable, the page falls back to the
browser's own Tamil speech.

### Voice notes (optional)
```bash
pip install openai-whisper        # also install ffmpeg on your system
export WHISPER_MODEL=small        # "medium" is more accurate for Tamil but slower
```

### WhatsApp bot (optional, Twilio sandbox)
See the docstring in `app/whatsapp.py`. Needs ngrok + a free Twilio account.

## Tests
```bash
python -m pytest -q
```

## Demo script (1-2 minutes)
1. Tap **KYC செய்தி** -> Scam, Tamil voice plays, 1930 button shown.
2. Tap **மின் கட்டணம்** -> Scam (Tamil script input).
3. Tap **OTP கேட்கிறார்கள்** -> Scam (Tanglish input).
4. Tap **வங்கி OTP (உண்மையானது)** -> Safe. Shows it does not cry wolf.
5. Show the consent checkbox + report buttons (privacy story).

## Privacy
* Messages are NOT stored by default. Voice notes are deleted right after transcription.
* A message is stored only if the user ticks the consent box and taps a report button.
  Phone numbers, emails and long digit strings (accounts, OTPs) are masked first
  (`engine.anonymise`). Reports go to `data/reports.csv` and are used by `app.train`.

## Known limitations (be upfront about these with judges)
* The training set in `app/seed_data.py` is a small, hand-written **synthetic** seed set.
  The cross-validation score printed by `app.train` is optimistic. Extend it with real data
  (public SMS-spam sets, volunteer-collected messages, confirmed reports).
* Rules were checked on a small set of unseen messages by hand, not on a large benchmark.
* The official-domain allowlist in `engine.OFFICIAL` is a heuristic. Review it.
* Whisper's Tamil accuracy varies with dialect and noise; the text box is the fallback.
* `app/main.py` (FastAPI layer) and `app/whatsapp.py` were syntax-checked but not run
  end to end in the build environment. Run them once before recording the demo.
* The family-alert feature from the slides is not implemented yet.
