from complaints_trends.config import PIIConfig
from complaints_trends.pii_redaction import redact_pii


def test_redact_pii():
    cfg = PIIConfig()
    txt = "email a@b.com phone +79990001122 url https://x.ru"
    out = redact_pii(txt, cfg)
    assert "<EMAIL>" in out and "<PHONE>" in out and "<URL>" in out
