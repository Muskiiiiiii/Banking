"""Task 10 — input-side guardrails (PII masking, prompt-injection detection).

The output-side groundedness guardrail lives in agents/pipeline.py (`output_groundedness_guard`)
because it needs the retrieval result.

Masked (fixed-format) PII:
  * PAN       — 5 letters, 4 digits, 1 letter            e.g. ABCDE1234F
  * Aadhaar   — 12 digits starting 2-9, optional spaces/hyphens in groups of 4
  * Bank a/c  — 9 to 18 consecutive digits (checked after Aadhaar)
Out of scope (no reliable pattern under a keyless masker): applicant names and
income figures. Only fabricated examples are used in this repository.
"""
from __future__ import annotations

import re

PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)
AADHAAR_RE = re.compile(r"(?<![\d-])[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?![\d-])")
ACCOUNT_RE = re.compile(r"(?<![\d-])\d{9,18}(?![\d-])")

# A 12-digit number can be an Aadhaar OR a bank account. When the text labels it
# ("account no 9876...", "a/c: ..."), the label wins; unlabeled 12-digit numbers
# starting 2-9 are treated as Aadhaar. Either way the digits are masked.
ACCOUNT_LABELLED_RE = re.compile(
    r"(?P<label>\b(?:bank\s+)?(?:account|a/c|acct)\.?(?:\s*(?:no\.?|number|num|#))?\s*(?:is|:|-)?\s*)"
    r"(?P<num>(?<!\d)\d[\d\s-]{7,20}\d(?!\d))", re.IGNORECASE)

MASKS = [("PAN", PAN_RE, "[PAN_REDACTED]"),
         ("AADHAAR", AADHAAR_RE, "[AADHAAR_REDACTED]"),
         ("BANK_ACCOUNT", ACCOUNT_RE, "[ACCOUNT_REDACTED]")]


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Return (masked_text, list_of_pii_types_found). Idempotent."""
    found = []

    def _labelled(m):
        digits = re.sub(r"\D", "", m.group("num"))
        if 9 <= len(digits) <= 18:
            _labelled.n += 1
            return m.group("label") + "[ACCOUNT_REDACTED]"
        return m.group(0)
    _labelled.n = 0
    text = ACCOUNT_LABELLED_RE.sub(_labelled, text)
    acct_labelled = _labelled.n
    for label, rx, token in MASKS:
        text, n = rx.subn(token, text)
        if label == "BANK_ACCOUNT":
            n += acct_labelled
        if n:
            found.append(f"{label}x{n}")
    return text, found


INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|your)\s+(instructions|rules|prompts?)",
    r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|your)?\s*(instructions|rules|guidelines)",
    r"forget\s+(all\s+|your\s+)?(previous\s+)?(instructions|rules)",
    r"(reveal|show|print|leak)\s+(me\s+)?(your|the)\s+(system\s+prompt|instructions|hidden\s+prompt)",
    r"\bsystem\s+prompt\b",
    r"you\s+are\s+now\s+(a|an|in)\b",
    r"\b(developer|dan|god)\s+mode\b",
    r"\bjailbreak",
    r"act\s+as\s+(an?\s+)?(unrestricted|unfiltered|admin|administrator|root)",
    r"(override|bypass)\s+(the\s+|your\s+|all\s+)?(rules|guardrails|safety|policy|policies|restrictions)",
    r"approve\s+(this|my)\s+loan\s+(now|immediately)\s+regardless",
]
_INJ = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def detect_injection(text: str) -> list[str]:
    return [m.group(0) for rx in _INJ if (m := rx.search(text))]


INJECTION_REFUSAL = ("I can't act on instructions that try to change how I work. I can help with "
                     "Cred loan-policy questions or check a loan application status.")
