"""ScamShield risk engine.

Pipeline:  text -> scam-pattern rules + URL checks + ML classifier -> verdict.

Everything here is plain Python + scikit-learn so it can be tested without
FastAPI, gTTS or Whisper.

NOTE on regexes: Python's \\w does NOT match Tamil vowel signs, so patterns use
\\S / \\s instead of \\w / \\W.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

MODEL_PATH = Path(__file__).parent / "model.joblib"

HELPLINE = "1930"
# Spoken digit-by-digit so Tamil TTS reads it clearly.
HELPLINE_SPOKEN_TA = "ஒன்று ஒன்பது மூன்று பூஜ்ஜியம்"

SAFE_MAX = 0.30   # score below this  -> Safe
RISKY_MAX = 0.60  # score below this  -> Risky, otherwise Scam


# --------------------------------------------------------------------------
# Rule helpers
# --------------------------------------------------------------------------
def near(a: str, b: str, gap: int = 6) -> str:
    """Regex: `a` followed by `b` with at most `gap` words in between."""
    return rf"(?:{a})\S{{0,3}}\s+(?:\S+\s+){{0,{gap}}}?(?:{b})"


SECRET = r"\b(?:otp|pin|cvv|password|passcode)\b|ஓடிபி|ஒடிபி|பின்|சிவிவி|கடவுச்சொல்"
ASK = (
    r"\b(?:share|send|tell|give|forward|provide|enter|confirm|read out|reply|"
    r"sollu|sollunga|anuppu|anuppunga|kudunga|kodunga)\b"
    r"|சொல்லு|சொல்லுங்கள்|அனுப்பு|அனுப்புங்கள்|பகிருங்கள்|கொடுங்கள்|தாருங்கள்"
)

# Sentences like "Do not share OTP with anyone" appear in GENUINE bank SMS.
# They are removed before the OTP-request rule runs so real OTP messages stay Safe.
SAFE_PHRASES = [
    r"(?:do not|don'?t|never)\s+(?:\S+\s+){0,3}?(?:share|disclose|reveal|give|tell)[^.\n]*",
    r"[^.\n]*(?:பகிர|சொல்ல|கொடுக்க|தெரிவிக்க)\s*வேண்டாம்[^.\n]*",
    r"[^.\n]*(?:sollathinga|sollatheenga|share pannathinga)[^.\n]*",
]


@dataclass(frozen=True)
class Rule:
    id: str
    pattern: str
    weight: float
    ta: str
    en: str
    critical: bool = False
    use_clean: bool = False  # run on text with "do not share" sentences removed


RULES: list[Rule] = [
    Rule(
        "otp_ask",
        near(ASK, SECRET, 3) + "|" + near(SECRET, ASK, 3),
        0.60,
        "OTP / PIN-ஐ கேட்கிறார்கள். வங்கி ஒருபோதும் இதை கேட்காது.",
        "Asks for your OTP/PIN. Banks never ask for this.",
        critical=True,
        use_clean=True,
    ),
    Rule(
        "collect_request",
        r"(?:approve|accept|authori[sz]e)\s+(?:\S+\s+){0,3}?(?:collect|payment|money)\s+request"
        r"|(?:enter|put|type)\s+(?:\S+\s+){0,3}?(?:upi\s+)?pin\s+(?:\S+\s+){0,6}?(?:receive|get|credit|refund|claim)"
        r"|பணம்\s+(?:\S+\s+){0,5}?பெற\S*\s+(?:\S+\s+){0,4}?(?:பின்|pin)",
        0.60,
        "பணம் பெற PIN போடச் சொல்கிறார்கள். PIN போட்டால் பணம் போகும், வராது.",
        "Asks for UPI PIN to 'receive' money. A PIN is only needed to SEND money.",
        critical=True,
    ),
    Rule(
        "remote_app",
        r"\b(?:anydesk|any desk|teamviewer|team viewer|quicksupport|quick support|rustdesk|ammyy|alpemix)\b|screen\s*share",
        0.60,
        "AnyDesk / TeamViewer போன்ற ஆப்-ஐ பதிவிறக்கச் சொல்கிறார்கள். உங்கள் போன் அவர்கள் கையில் போகும்.",
        "Asks you to install a remote-access app. This gives them control of your phone.",
        critical=True,
    ),
    Rule(
        "kyc_update",
        near(r"\bkyc\b|\bpan\b|\baadhaar\b|\baadhar\b|கேஒய்சி|கேவைசி",
             r"update|expire|expired|suspend|block|blocked|verify|link|புதுப்பி\S*|காலாவதி\S*", 5),
        0.40,
        "KYC புதுப்பிக்கச் சொல்லி பயமுறுத்துகிறார்கள். வங்கி SMS லிங்க் மூலம் KYC கேட்காது.",
        "KYC-update threat by SMS/link. Banks do not do KYC through links.",
    ),
    Rule(
        "urgency",
        r"\b(?:immediately|urgent|urgently|asap|today only|last chance|expires?|expired|tonight|"
        r"within\s+\d+\s*(?:hours?|hrs?|minutes?|mins?)|udane|ippove)\b"
        r"|உடனே|உடனடியாக|இன்றே|இன்று இரவு|அவசரம்|இன்றுக்குள்",
        0.20,
        "அவசரப்படுத்துகிறார்கள். மோசடிக்காரர்கள் யோசிக்க நேரம் தர மாட்டார்கள்.",
        "Creates urgency. Scammers do not give you time to think.",
    ),
    Rule(
        "account_block",
        near(r"\baccount\b|\bcard\b|\bsim\b|\bupi\b|\bnumber\b|கணக்கு|சிம்|கார்டு",
             r"block|blocked|suspend|suspended|freeze|frozen|deactivat\S*|disconnect\S*|முடக்க\S*|நிறுத்த\S*|தடை\S*", 5),
        0.30,
        "கணக்கு / சிம் முடக்கப்படும் என்று மிரட்டுகிறார்கள்.",
        "Threatens to block your account/SIM.",
    ),
    Rule(
        "click_link",
        r"\b(?:click|tap|open|download|install)\b[^.\n]{0,25}\b(?:link|here|below|app|apk)\b"
        r"|கிளிக்|லிங்க்|இணைப்பை|\.apk\b",
        0.15,
        "லிங்கை தொடச் சொல்கிறார்கள்.",
        "Pushes you to click a link / install an app.",
    ),
    Rule(
        "prize",
        r"\b(?:lottery|lucky draw|you (?:have )?won|winner|prize|gift voucher|cashback of|reward points?)\b"
        r"|பரிசு|லாட்டரி|வெற்றி பெற்றீர்கள்|அதிர்ஷ்ட",
        0.45,
        "பரிசு / லாட்டரி விழுந்ததாக ஆசை காட்டுகிறார்கள்.",
        "Lottery / prize bait.",
    ),
    Rule(
        "loan_instant",
        r"\b(?:instant loan|pre-?approved loan|loan (?:is )?(?:approved|sanctioned)|loan without (?:documents|cibil)|no cibil)\b"
        r"|உடனடி கடன்|கடன் அனுமதி|கடன் கிடைக்கும்",
        0.30,
        "உடனடி கடன் என்று ஆசை காட்டுகிறார்கள்.",
        "Instant / pre-approved loan bait.",
    ),
    Rule(
        "advance_fee",
        r"\b(?:processing fee|advance fee|registration fee|insurance fee|release fee|gst fee|file charge)\b"
        r"|முன்பணம்|பதிவு கட்டணம்|கட்டணம் செலுத்து",
        0.40,
        "முன்பணம் / கட்டணம் கேட்கிறார்கள். உண்மையான கடனுக்கு முன்பணம் கிடையாது.",
        "Asks for an advance fee. Real lenders do not ask for money before a loan.",
    ),
    Rule(
        "eb_threat",
        near(r"\btneb\b|\btangedco\b|\belectricity\b|\beb\b|மின்சார\S*|மின் கட்டண\S*|மின் இணைப்பு",
             r"disconnect\S*|\bcut\b|pending|unpaid|overdue|துண்டிக்க\S*|நிலுவை", 8),
        0.45,
        "மின் இணைப்பு துண்டிக்கப்படும் என்று மிரட்டுகிறார்கள். கட்டணத்தை அதிகாரப்பூர்வ தளத்தில் மட்டும் சரிபாருங்கள்.",
        "Electricity disconnection threat. Check dues only on the official TANGEDCO site.",
    ),
    Rule(
        "parcel_arrest",
        near(r"\bparcel\b|\bcourier\b|\bfedex\b|\bcustoms\b|\bnarcotics\b|\bcbi\b|\btrai\b|\bcyber ?crime\b|கூரியர்|பார்சல்|போலீஸ்",
             r"arrest\S*|illegal|seized|drugs?|\bcase\b|\bfir\b|warrant|கைது|போதைப்\S*|வழக்கு", 10),
        0.55,
        "போலீஸ் / கூரியர் / சிபிஐ பெயரில் கைது என்று மிரட்டுகிறார்கள். இது 'டிஜிட்டல் கைது' மோசடி.",
        "'Digital arrest' style threat in the name of police/courier/CBI.",
    ),
    Rule(
        "job_task",
        r"\b(?:work from home|part[- ]?time job|youtube like|like and subscribe|telegram task|daily income|"
        r"earn\s+(?:rs\.?|₹)?\s*\d+\s*(?:per|/|a)\s*day)\b"
        r"|வீட்டிலிருந்தே வேலை|தினமும் சம்பாதி",
        0.50,
        "வீட்டிலிருந்தே சம்பாதிக்கலாம் என்று ஆசை காட்டுகிறார்கள்.",
        "Work-from-home / task-job bait.",
    ),
    Rule(
        "family_impersonation",
        r"\b(?:hi|hello)\s+(?:mom|dad|amma|appa)\b|\bthis is my new number\b|\bnew number\b[^.\n]{0,40}\b(?:send|need|urgent)\b"
        r"|\b(?:puthu|pudhu|puthiya)\s+number\b|புதிய (?:எண்|நம்பர்)",
        0.50,
        "புதிய எண்ணிலிருந்து குடும்பத்தினர் போல் பணம் கேட்கிறார்கள். பழைய எண்ணில் அழைத்து உறுதி செய்யுங்கள்.",
        "Pretends to be a relative on a new number. Call their old number to confirm.",
    ),
    Rule(
        "send_money",
        r"\b(?:send|transfer|gpay|pay)\b\s+(?:\S+\s+){0,4}?(?:rs\.?|₹|inr)\s?\d"
        r"|(?:rs\.?|₹|inr)\s?\d[\d,]*\s+(?:\S+\s+){0,2}?(?:gpay|phonepe|anuppu\S*|send|அனுப்பு\S*)",
        0.25,
        "பணம் அனுப்பச் சொல்கிறார்கள்.",
        "Asks you to send money.",
    ),
]
_RULES_COMPILED = [(r, re.compile(r.pattern, re.IGNORECASE)) for r in RULES]
_SAFE_PHRASES_COMPILED = [re.compile(p, re.IGNORECASE) for p in SAFE_PHRASES]


# --------------------------------------------------------------------------
# URL checks
# --------------------------------------------------------------------------
URL_RE = re.compile(
    r"(?i)\b((?:https?://|www\.)[^\s<>\"']+"
    r"|(?:[a-z0-9-]+\.)+(?:com|in|co\.in|net|org|xyz|top|click|live|icu|cfd|link|site|online|info|me|ly|gl|app|club|vip|shop|sbi|gov\.in)"
    r"(?:/[^\s<>\"']*)?)"
)

SHORTENERS = {"bit.ly", "tinyurl.com", "cutt.ly", "rb.gy", "is.gd", "t.ly", "goo.gl", "shorturl.at", "ow.ly", "tiny.cc", "rebrand.ly"}
SUSPICIOUS_TLDS = {"xyz", "top", "click", "live", "icu", "cfd", "link", "site", "online", "club", "vip", "shop", "info"}

# brand keyword -> official registered domains.  KEEP THIS LIST REVIEWED; it is a
# heuristic allowlist, not an authoritative registry.
OFFICIAL = {
    "sbi": {"sbi.co.in", "onlinesbi.sbi", "sbicard.com", "sbilife.co.in", "sbi.bank.in"},
    "hdfc": {"hdfcbank.com", "hdfc.com", "hdfclife.com"},
    "icici": {"icicibank.com", "icicilombard.com", "iciciprulife.com"},
    "axisbank": {"axisbank.com"},
    "kotak": {"kotak.com"},
    "paytm": {"paytm.com"},
    "phonepe": {"phonepe.com"},
    "npci": {"npci.org.in"},
    "uidai": {"uidai.gov.in"},
    "aadhaar": {"uidai.gov.in"},
    "incometax": {"incometax.gov.in", "incometaxindia.gov.in"},
    "tangedco": {"tangedco.gov.in", "tnebnet.org"},
    "tneb": {"tangedco.gov.in", "tnebnet.org"},
    "indiapost": {"indiapost.gov.in"},
    "amazon": {"amazon.in", "amazon.com"},
    "flipkart": {"flipkart.com"},
    "airtel": {"airtel.in"},
}
_SECOND_LEVEL = {"co", "com", "org", "gov", "ac", "net", "nic"}

# url rule id -> (weight, ta, en)
URL_FLAGS = {
    "url_shortener": (0.30, "சுருக்கப்பட்ட லிங்க். எங்கே போகும் என்று தெரியாது.", "Shortened link hides the real destination."),
    "url_odd_tld": (0.25, "சந்தேகமான இணையதள முகவரி.", "Unusual website ending often used by scam sites."),
    "url_ip": (0.40, "எண்களால் ஆன இணையதள முகவரி.", "Link uses a raw IP address."),
    "url_brand": (0.55, "வங்கி / அரசு பெயரை போல் தோன்றும் போலி இணையதளம்.", "Fake site imitating a bank/government brand."),
    "url_punycode": (0.40, "போலி எழுத்துகளை பயன்படுத்தும் முகவரி.", "Look-alike (punycode) domain."),
    "url_at": (0.30, "லிங்கில் '@' குறியீடு உள்ளது. இது ஏமாற்றும் தந்திரம்.", "'@' in a link is a classic trick to hide the real host."),
    "url_http": (0.10, "பாதுகாப்பற்ற (http) லிங்க்.", "Not using HTTPS."),
}


def extract_urls(text: str) -> list[str]:
    seen, out = set(), []
    for m in URL_RE.finditer(text):
        u = m.group(1).rstrip(".,;:!?)")
        if u.lower() not in seen:
            seen.add(u.lower())
            out.append(u)
    return out


def registered_domain(host: str) -> str:
    labels = host.lower().strip(".").split(".")
    if len(labels) >= 3 and labels[-2] in _SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 else host.lower()


def analyze_url(url: str) -> list[str]:
    """Return a list of URL_FLAGS ids that apply to `url`."""
    raw = url if re.match(r"(?i)https?://", url) else "http://" + url
    try:
        p = urlparse(raw)
    except ValueError:
        return ["url_odd_tld"]
    host = (p.hostname or "").lower()
    flags: list[str] = []
    if not host:
        return flags
    dom = registered_domain(host)

    # Official domain -> trusted, no flags (but still https check is skipped).
    for brand, doms in OFFICIAL.items():
        if dom in doms:
            return []

    if dom in SHORTENERS:
        flags.append("url_shortener")
    if dom.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS:
        flags.append("url_odd_tld")
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
        flags.append("url_ip")
    if "xn--" in host:
        flags.append("url_punycode")
    if "@" in p.netloc:
        flags.append("url_at")
    for brand in OFFICIAL:
        if brand in host.replace("-", ""):
            flags.append("url_brand")
            break
    if url.lower().startswith("http://"):
        flags.append("url_http")
    return flags


# --------------------------------------------------------------------------
# ML model (optional; engine works rules-only if the model file is missing)
# --------------------------------------------------------------------------
_model = None
_model_loaded = False


def load_model(force: bool = False):
    global _model, _model_loaded
    if _model_loaded and not force:
        return _model
    _model_loaded = True
    if MODEL_PATH.exists():
        try:
            import joblib

            _model = joblib.load(MODEL_PATH)
        except Exception:
            _model = None
    else:
        _model = None
    return _model


def ml_probability(text: str) -> float | None:
    model = load_model()
    if model is None:
        return None
    return float(model.predict_proba([text])[0][1])


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def _noisy_or(weights: list[float]) -> float:
    p = 1.0
    for w in weights:
        p *= 1.0 - w
    return 1.0 - p


def _clean_for_otp(text: str) -> str:
    out = text
    for rx in _SAFE_PHRASES_COMPILED:
        out = rx.sub(" ", out)
    return out


def check(text: str) -> dict:
    """Analyse a message and return a verdict dict."""
    text = (text or "").strip()
    if not text:
        return _result("safe", 0.0, 0.0, None, [], [])

    clean = _clean_for_otp(text)
    hits: list[dict] = []
    weights: list[float] = []
    critical = False

    for rule, rx in _RULES_COMPILED:
        target = clean if rule.use_clean else text
        if rx.search(target):
            hits.append({"id": rule.id, "weight": rule.weight, "ta": rule.ta, "en": rule.en})
            weights.append(rule.weight)
            critical = critical or rule.critical

    urls = extract_urls(text)
    url_report = []
    for u in urls:
        flags = analyze_url(u)
        url_report.append({"url": u, "flags": flags})
        for f in flags:
            w, ta, en = URL_FLAGS[f]
            if not any(h["id"] == f for h in hits):
                hits.append({"id": f, "weight": w, "ta": ta, "en": en})
            weights.append(w)

    rule_score = _noisy_or(weights)
    ml = ml_probability(text)
    score = rule_score if ml is None else 0.65 * rule_score + 0.35 * ml
    if critical:
        score = max(score, 0.85)
    # A message with zero rule hits is capped below "Scam" so ML alone can never
    # shout "scam" without any explainable reason (keeps false alarms down).
    if not hits:
        score = min(score, 0.45)

    verdict = "safe" if score < SAFE_MAX else "risky" if score < RISKY_MAX else "scam"
    hits.sort(key=lambda h: h["weight"], reverse=True)
    return _result(verdict, score, rule_score, ml, hits, url_report)


def _result(verdict, score, rule_score, ml, hits, url_report) -> dict:
    top = [h["ta"] for h in hits[:2]]
    if verdict == "scam":
        title = "மோசடி!"
        advice = (
            "லிங்கை தொடாதீர்கள். OTP, PIN யாரிடமும் சொல்லாதீர்கள். பணம் அனுப்பாதீர்கள். "
            f"பணம் இழந்திருந்தால் உடனே {HELPLINE} எண்ணுக்கு அழைக்கவும்."
        )
        spoken = "எச்சரிக்கை! இது மோசடி செய்தி. " + " ".join(top) + " " + advice.replace(HELPLINE, HELPLINE_SPOKEN_TA)
    elif verdict == "risky":
        title = "சந்தேகம்"
        advice = (
            "லிங்கை தொடாதீர்கள். உங்கள் வங்கியின் அதிகாரப்பூர்வ எண்ணில் அழைத்து உறுதி செய்யுங்கள். "
            f"சந்தேகம் இருந்தால் {HELPLINE} எண்ணுக்கு அழையுங்கள்."
        )
        spoken = "கவனம்! இந்த செய்தி சந்தேகமாக உள்ளது. " + " ".join(top) + " " + advice.replace(HELPLINE, HELPLINE_SPOKEN_TA)
    else:
        title = "பாதுகாப்பானது"
        advice = "இந்த செய்தியில் மோசடி அறிகுறிகள் தெரியவில்லை. இருந்தாலும் OTP, PIN யாரிடமும் பகிராதீர்கள்."
        spoken = advice
    return {
        "verdict": verdict,
        "title_ta": title,
        "score": round(float(score), 3),
        "rule_score": round(float(rule_score), 3),
        "ml_score": None if ml is None else round(float(ml), 3),
        "reasons": [{"id": h["id"], "ta": h["ta"], "en": h["en"]} for h in hits],
        "urls": url_report,
        "advice_ta": advice,
        "spoken_ta": spoken.strip(),
        "helpline": HELPLINE,
    }


# --------------------------------------------------------------------------
# Privacy helper for the feedback loop
# --------------------------------------------------------------------------
_PHONE = re.compile(r"(?:\+?91[\s-]?)?[6-9]\d{9}")
_LONGNUM = re.compile(r"\d{6,}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def anonymise(text: str) -> str:
    """Mask phone numbers, long digit strings (accounts/OTPs) and emails."""
    t = _EMAIL.sub("<EMAIL>", text)
    t = _PHONE.sub("<PHONE>", t)
    t = _LONGNUM.sub("<NUM>", t)
    return t
