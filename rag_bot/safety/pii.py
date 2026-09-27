"""PII redaction at the input boundary (architecture section 6).

No PAN, Aadhaar, account number, OTP, email or phone number may reach the
embedder, the LLM, the logs or the screen. `scrub()` is the only way text enters
the pipeline, and the redacted string -- never the raw one -- is what flows on.

THE FALSE-POSITIVE PROBLEM
---------------------------
A blunt digit regex destroys this product. "expense ratio 1.05%", "exit load 1%",
"minimum SIP Rs 500" are the answers the assistant exists to give. A scrubber that
eats "1.05%" is a worse bug than one that leaks an email, because the damage is
silent: no test fails, the answer is just quietly wrong.

Two mechanisms prevent it:

1. **No bare digit sweep.** Nothing is ever redacted for being digits alone. Every
   digit rule needs either a PII-specific length (10-char PAN, 12-digit Aadhaar,
   10-digit phone) or a PII keyword within 40 characters. The keyword is what
   makes "otp 482913" PII and "minimum SIP 500" not.
2. **Protected spans.** Percentages and currency amounts are located first and no
   rule may touch them. This is not belt-and-braces: without it, a query like
   "my pin code, what is the minimum SIP, Rs 5000" would redact the 5000, because
   the keyword "code" is inside the 40-character window. The protection is what
   keeps a legitimate amount intact when a PII keyword happens to be nearby.

Each rule is a separate named function so tuning one cannot disturb the others.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# How far from a digit run a PII keyword may sit and still count.
KEYWORD_WINDOW = 40

# ---------------------------------------------------------------------------
# protected spans: the product's actual answers
# ---------------------------------------------------------------------------

_PERCENTAGE = re.compile(r"(?<![\w.])[0-9]+(?:\.[0-9]+)?\s?(?:%|percent\b)", re.IGNORECASE)
_CURRENCY = re.compile(
    r"(?<![\w])(?:rs\.?|inr|₹)\s?[0-9][0-9,]*(?:\.[0-9]+)?(?![\w])", re.IGNORECASE
)


def _protected_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges that no rule is allowed to modify."""
    spans: list[tuple[int, int]] = []
    for pattern in (_PERCENTAGE, _CURRENCY):
        spans.extend(m.span() for m in pattern.finditer(text))
    return spans


def _overlaps_protected(start: int, end: int, protected: list[tuple[int, int]]) -> bool:
    return any(start < pe and ps < end for ps, pe in protected)


# ---------------------------------------------------------------------------
# keyword proximity
# ---------------------------------------------------------------------------

ACCOUNT_KEYWORDS = ("a/c", "account", "acct", "folio", "acc no", "a/c no", "ledger")
OTP_KEYWORDS = ("otp", "one time password", "code", "pin", "passcode", "pass word")


def _has_keyword_near(
    text: str, start: int, end: int, keywords: tuple[str, ...],
    window: int = KEYWORD_WINDOW,
) -> str | None:
    """The keyword governing this digit run, if one is within `window` chars.

    Looks both before and after the run: users write "a/c no 12345678" but also
    "12345678 is my account number", and a rule that only reads leftwards leaks
    the second.
    """
    left = text[max(0, start - window):start]
    right = text[end:end + window]
    for keyword in keywords:
        if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", left, re.IGNORECASE):
            return keyword
    for keyword in keywords:
        if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", right, re.IGNORECASE):
            return keyword
    return None


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------

# 5 letters, 4 digits, 1 letter, not glued to a longer alphanumeric run.
# Case-insensitive on purpose: PANs are upper-case by format but users paste and
# retype them in lower case, and PII does not stop being PII because of a shift key.
_PAN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{5}[0-9]{4}[A-Za-z](?![A-Za-z0-9])")

# 12 digits, grouped 4-4-4 with space / hyphen / slash and any number of them, or
# X-masked. The separator class is broad because "4829-1396-2510" and
# "4829  1396  2510" are how people actually write an Aadhaar number; requiring a
# single space leaks the most common formats. Safe here because the corpus contains
# no 12-digit run of any kind, so a broad separator cannot eat a product answer.
_SEP = r"[\s./-]+"
_AADHAAR = re.compile(
    rf"(?<![A-Za-z0-9])(?:"
    rf"[0-9]{{4}}{_SEP}[0-9]{{4}}{_SEP}[0-9]{{4}}"
    rf"|[0-9]{{12}}"
    rf"|X{{4}}{_SEP}?X{{4}}{_SEP}?[0-9]{{4}}"
    rf")(?![A-Za-z0-9])"
)

_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
    r"(?![A-Za-z0-9-])"
)

# 10 digits with an optional +91 / 0 prefix. Tightened to a [6-9] leading digit
# because that is where Indian mobile numbers start: a bare "1234567890" is far
# more often an account number (caught by the account rule) than a phone number,
# and the loose form is what makes digit rules untrustworthy.
_PHONE = re.compile(r"(?<![0-9])(?:\+?91[-\s]?|0)?[6-9][0-9]{9}(?![0-9])")

_ACCOUNT = re.compile(r"(?<![0-9])[0-9]{8,16}(?![0-9])")
_OTP = re.compile(r"(?<![0-9])[0-9]{4,6}(?![0-9])")


@dataclass(frozen=True)
class ScrubResult:
    text: str                 # the REDACTED string -- this is what flows onward
    rules_fired: list[str] = field(default_factory=list)
    clean: bool = True

    @property
    def redacted(self) -> bool:
        return bool(self.rules_fired)


def _redact_pan(text: str) -> tuple[str, bool]:
    out, fired = _PAN.subn("[PAN redacted]", text)
    return out, bool(fired)


def _redact_aadhaar(text: str) -> tuple[str, bool]:
    out, fired = _AADHAAR.subn("[Aadhaar redacted]", text)
    return out, bool(fired)


def _redact_email(text: str) -> tuple[str, bool]:
    out, fired = _EMAIL.subn("[email redacted]", text)
    return out, bool(fired)


def _redact_phone(text: str) -> tuple[str, bool]:
    out, fired = _PHONE.subn("[phone redacted]", text)
    return out, bool(fired)


def _redact_account(text: str) -> tuple[str, bool]:
    """8-16 digits, but only with an account keyword within 40 chars."""
    out, n = _sub_with_keyword(text, _ACCOUNT, ACCOUNT_KEYWORDS, "[account redacted]")
    return out, n > 0


def _redact_otp(text: str) -> tuple[str, bool]:
    """4-6 digits, but only with an OTP keyword within 40 chars."""
    out, n = _sub_with_keyword(text, _OTP, OTP_KEYWORDS, "[OTP redacted]")
    return out, n > 0


def _sub_with_keyword(
    text: str, pattern: re.Pattern[str], keywords: tuple[str, ...], replacement: str
) -> tuple[str, int]:
    """Replace pattern matches that have a governing keyword and are not protected.

    Protected spans are recomputed per match rather than computed once, because
    an earlier replacement inside the same pass shifts every later offset.
    """
    out: list[str] = []
    cursor = 0
    count = 0
    for match in pattern.finditer(text):
        if match.start() < cursor:
            continue
        if _overlaps_protected(match.start(), match.end(), _protected_spans(text)):
            continue
        if _has_keyword_near(text, match.start(), match.end(), keywords) is None:
            continue
        out.append(text[cursor:match.start()])
        out.append(replacement)
        cursor = match.end()
        count += 1
    if not count:
        return text, 0
    out.append(text[cursor:])
    return "".join(out), count


# phone and aadhaar are unkeyed (their lengths are specific), so they are applied
# wholesale; account and otp go through the keyword gate.
#
# Order matters twice over. The 8-16 digit account rule is tried before the 4-6
# digit OTP rule, so a folio number is reported as an account rather than as an
# OTP. And phone runs before aadhaar because "+91 9876543210" is 12 digits once
# the country code is counted -- letting aadhaar match first redacts a phone
# number and labels it "[Aadhaar redacted]", which is wrong in the one field
# downstream logging reads.
_RULES = (
    ("pan", _redact_pan),
    ("email", _redact_email),
    ("phone", _redact_phone),
    ("aadhaar", _redact_aadhaar),
    ("account", _redact_account),
    ("otp", _redact_otp),
)


def scrub(text: str) -> ScrubResult:
    """Redact PII. Never raises, for any input.

    Returns the redacted string plus the rules that fired. `clean` is True only
    when nothing was redacted.
    """
    if not isinstance(text, str) or not text:
        return ScrubResult(text=text if isinstance(text, str) else "", rules_fired=[],
                           clean=True)

    rules_fired: list[str] = []
    current = text
    for name, rule in _RULES:
        try:
            current, fired = rule(current)
        except Exception:  # a broken rule must not leak the text it failed on
            continue
        if fired and name not in rules_fired:
            rules_fired.append(name)
    return ScrubResult(text=current, rules_fired=rules_fired, clean=not rules_fired)


def scrub_for_log(text: str) -> str:
    """Same rules, plain string. Used by the trace logger."""
    return scrub(text).text
