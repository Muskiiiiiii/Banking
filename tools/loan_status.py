"""Task 6 — check_loan_application_status with a designed escalation score.

escalation_score = 0.6 * fraud + 0.4 * staleness            (range [0, 1])
  fraud     = 1 if flagged_for_fraud_review else 0
  staleness = days_since_created / 30  if status is still open (Submitted, Under Review,
              Approved-but-not-disbursed), else 0   (Rejected / Disbursed are closed, so
              their age says nothing about a member waiting on us)

Threshold (escalate when score >= threshold) is derived from the generated data:
  threshold = 0.4 * P80(days_since_created over open applications) / 30
With SEED=2026 this is 0.4 * 25.2 / 30 = 0.336, which means
  * every fraud-flagged application escalates (0.6 alone clears the bar), and
  * a clean open application escalates only when it is older than the 80th percentile
    of open applications (> 25.2 days), i.e. the stalest ~20% of the open queue.
"""
from __future__ import annotations

import math
import re

import config
from dataset import LOAN_APPLICATIONS, MAX_DAYS, OPEN_STATUSES, RECORDS_BY_ID

RECORD_ID_RE = re.compile(r"\bCRED-LA-\d{4}\b", re.IGNORECASE)


def _percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


OPEN_DAYS = [r["days_since_created"] for r in LOAN_APPLICATIONS if r["status"] in OPEN_STATUSES]
P80_OPEN_DAYS = _percentile(OPEN_DAYS, 0.80)
ESCALATION_THRESHOLD = round(config.ESCALATION_W_STALENESS * P80_OPEN_DAYS / MAX_DAYS, 3)


def escalation_score(record: dict) -> float:
    fraud = 1.0 if record["flagged_for_fraud_review"] else 0.0
    staleness = (min(record["days_since_created"], MAX_DAYS) / MAX_DAYS
                 if record["status"] in OPEN_STATUSES else 0.0)
    return round(config.ESCALATION_W_FRAUD * fraud + config.ESCALATION_W_STALENESS * staleness, 4)


def check_loan_application_status(record_id: str) -> dict:
    rid = (record_id or "").strip().upper()
    rec = RECORDS_BY_ID.get(rid)
    if rec is None:
        return {"record_id": rid, "found": False,
                "error": f"No loan application found with id {rid!r}."}
    score = escalation_score(rec)
    return {
        "record_id": rid,
        "found": True,
        "category": rec["category"],
        "status": rec["status"],
        "loan_amount_inr": rec["loan_amount_inr"],
        "days_since_created": rec["days_since_created"],
        "flagged_for_fraud_review": rec["flagged_for_fraud_review"],
        "escalation_score": score,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "escalate": score >= ESCALATION_THRESHOLD,
    }


def extract_record_id(text: str) -> str | None:
    m = RECORD_ID_RE.search(text or "")
    return m.group(0).upper() if m else None
