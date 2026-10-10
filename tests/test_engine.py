"""Run:  python -m pytest -q   (or)   python tests/test_engine.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import engine  # noqa: E402

# (message, expected verdicts allowed)
SCAMS = [
    "Dear customer, your SBI account will be blocked today. Update KYC immediately: http://sbi-kyc-verify.xyz/login",
    "உங்கள் மின் இணைப்பு இன்று இரவு துண்டிக்கப்படும். மின் கட்டணம் நிலுவையில் உள்ளது. உடனே இந்த எண்ணை அழைக்கவும்.",
    "Please share your OTP to complete the refund.",
    "Enter your UPI PIN to receive Rs 5000 cashback.",
    "Install AnyDesk and tell me the code, I will fix your account.",
    "Ungal OTP sollunga, refund anuppa vendum.",
    "Congratulations! You have won a lucky draw. Pay processing fee Rs 4999 to claim.",
    "Hi mom, this is my new number. Send Rs 15000 on gpay urgently.",
    "Your parcel has illegal drugs. Police will arrest you. Call now.",
    "பணம் பெற உங்கள் UPI PIN-ஐ பதிவிடுங்கள்.",
]

SAFE = [
    "123456 is your OTP for login. Do not share this OTP with anyone. -SBI",
    "உங்கள் OTP 458231. இதை யாரிடமும் பகிர வேண்டாம்.",
    "Rs 500.00 debited from A/c XX1234 via UPI. If not done by you, call bank helpline.",
    "Amma, I will reach home by 7 pm.",
    "நாளை காலை கோயிலுக்கு போகலாம். தயாராக இருங்கள்.",
    "Your SBI statement is ready. Log in at https://onlinesbi.sbi to view it.",
    "Pay your electricity bill on https://www.tangedco.gov.in. Due date 25th.",
    "Happy birthday da! Have a great year ahead.",
]

URL_CASES = {
    "http://bit.ly/abc": {"url_shortener", "url_http"},
    "https://sbi-login-secure.xyz/x": {"url_odd_tld", "url_brand"},
    "http://185.23.44.9/verify": {"url_ip", "url_http"},
    "https://onlinesbi.sbi/login": set(),
    "https://www.amazon.in/orders": set(),
    "https://amazon-gift-claim.com": {"url_brand"},
}


def run_suite(label: str) -> int:
    failures = 0
    for m in SCAMS:
        r = engine.check(m)
        if r["verdict"] == "safe":
            failures += 1
            print(f"[{label}] MISSED SCAM  score={r['score']}: {m[:70]}")
    for m in SAFE:
        r = engine.check(m)
        if r["verdict"] == "scam":
            failures += 1
            print(f"[{label}] FALSE ALARM  score={r['score']}: {m[:70]}")
    return failures


def test_url_flags():
    for url, expected in URL_CASES.items():
        got = set(engine.analyze_url(url))
        assert got == expected, f"{url}: expected {expected}, got {got}"


def test_rules_only():
    engine._model, engine._model_loaded = None, True  # force rules-only
    assert run_suite("rules-only") == 0


def test_with_model():
    engine.load_model(force=True)
    if engine._model is None:
        return  # model not trained yet
    assert run_suite("rules+ml") == 0


def test_otp_notice_is_safe_but_otp_request_is_not():
    engine._model, engine._model_loaded = None, True
    assert engine.check("Your OTP is 998877. Do not share it with anyone.")["verdict"] == "safe"
    assert engine.check("Send me your OTP now")["verdict"] == "scam"


def test_anonymise():
    out = engine.anonymise("Call 9876543210 or mail a.b@x.com, A/c 123456789012")
    assert "9876543210" not in out and "a.b@x.com" not in out and "123456789012" not in out


def test_empty():
    assert engine.check("")["verdict"] == "safe"


if __name__ == "__main__":
    test_url_flags()
    print("url flags OK")
    engine._model, engine._model_loaded = None, True
    print("rules-only failures:", run_suite("rules-only"))
    engine.load_model(force=True)
    print("model loaded:", engine._model is not None)
    if engine._model is not None:
        print("rules+ml failures:", run_suite("rules+ml"))
