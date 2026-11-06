"""Task 1 — seeded, deterministic loan-application dataset for the Cred track.

Run:  python dataset.py

Design choices (also stated in README.md):
  * SEED = 2026, N_RECORDS = 50, random.Random(SEED) — same output on every run.
  * Category weights: Personal 0.30, Home 0.20, Auto 0.20, Education 0.15, Business 0.15.
  * Status weights:   Submitted 0.20, Under Review 0.30, Approved 0.15, Rejected 0.15, Disbursed 0.20.
  * flagged_for_fraud_review ~ Bernoulli(0.18).
  * loan_amount_inr is drawn uniformly per category, rounded to the nearest Rs 5,000:
        Personal  50,000 – 25,00,000
        Home      10,00,000 – 1,50,00,000
        Auto      2,00,000 – 30,00,000
        Education 1,00,000 – 50,00,000
        Business  5,00,000 – 2,00,00,000
    Reasoning: each band follows the ticket sizes Indian lenders typically offer for that
    product, so a Rs 1.2 crore personal loan or a Rs 60,000 home loan can never appear.
  * days_since_created ~ uniform integer 0–30.

The generator never edits individual records. If a structural threshold fails,
validate() raises, and the fix is to change SEED or the weights and regenerate.
"""
from __future__ import annotations

import random
from collections import Counter

SEED = 2026
N_RECORDS = 50

CATEGORY_WEIGHTS = {
    "Personal Loan": 0.30,
    "Home Loan": 0.20,
    "Auto Loan": 0.20,
    "Education Loan": 0.15,
    "Business Loan": 0.15,
}
STATUS_WEIGHTS = {
    "Submitted": 0.20,
    "Under Review": 0.30,
    "Approved": 0.15,
    "Rejected": 0.15,
    "Disbursed": 0.20,
}
FRAUD_FLAG_PROBABILITY = 0.18
AMOUNT_RANGES_INR = {
    "Personal Loan": (50_000, 2_500_000),
    "Home Loan": (1_000_000, 15_000_000),
    "Auto Loan": (200_000, 3_000_000),
    "Education Loan": (100_000, 5_000_000),
    "Business Loan": (500_000, 20_000_000),
}
MAX_DAYS = 30

# Statuses where the application is still moving through the pipeline (used by the
# escalation score in tools/loan_status.py). Rejected and Disbursed are terminal.
OPEN_STATUSES = {"Submitted", "Under Review", "Approved"}


def generate(seed: int = SEED, n: int = N_RECORDS) -> list[dict]:
    rng = random.Random(seed)
    cats, cat_w = zip(*CATEGORY_WEIGHTS.items())
    stats, stat_w = zip(*STATUS_WEIGHTS.items())
    records = []
    for i in range(1, n + 1):
        category = rng.choices(cats, weights=cat_w, k=1)[0]
        status = rng.choices(stats, weights=stat_w, k=1)[0]
        lo, hi = AMOUNT_RANGES_INR[category]
        amount = int(round(rng.uniform(lo, hi) / 5_000) * 5_000)
        records.append(
            {
                "record_id": f"CRED-LA-{i:04d}",
                "category": category,
                "status": status,
                "loan_amount_inr": amount,
                "days_since_created": rng.randint(0, MAX_DAYS),
                "flagged_for_fraud_review": rng.random() < FRAUD_FLAG_PROBABILITY,
            }
        )
    return records


def summarize(records: list[dict]) -> dict:
    cat = Counter(r["category"] for r in records)
    st = Counter(r["status"] for r in records)
    flagged = sum(r["flagged_for_fraud_review"] for r in records)
    return {
        "n_records": len(records),
        "per_category": {c: cat.get(c, 0) for c in CATEGORY_WEIGHTS},
        "per_status": {s: st.get(s, 0) for s in STATUS_WEIGHTS},
        "flagged_count": flagged,
        "flagged_pct": round(100 * flagged / len(records), 2),
    }


def validate(records: list[dict]) -> dict:
    s = summarize(records)
    problems = []
    if s["n_records"] < 40:
        problems.append("fewer than 40 records")
    problems += [f"category {c} has {n} < 3" for c, n in s["per_category"].items() if n < 3]
    problems += [f"status {k} has 0 records" for k, n in s["per_status"].items() if n < 1]
    if not (10 <= s["flagged_pct"] <= 30):
        problems.append(f"flagged pct {s['flagged_pct']} outside 10-30%")
    for r in records:
        lo, hi = AMOUNT_RANGES_INR[r["category"]]
        if not (lo <= r["loan_amount_inr"] <= hi) or not (0 <= r["days_since_created"] <= MAX_DAYS):
            problems.append(f"{r['record_id']} out of range")
    if problems:
        raise ValueError("Dataset failed validation — change SEED or weights and regenerate: " + "; ".join(problems))
    return s


LOAN_APPLICATIONS: list[dict] = generate()
_SUMMARY = validate(LOAN_APPLICATIONS)
RECORDS_BY_ID: dict[str, dict] = {r["record_id"]: r for r in LOAN_APPLICATIONS}


if __name__ == "__main__":
    print(f"SEED={SEED}  N_RECORDS={N_RECORDS}")
    print(f"Category weights: {CATEGORY_WEIGHTS}")
    print(f"Status weights:   {STATUS_WEIGHTS}")
    print(f"Fraud-flag probability: {FRAUD_FLAG_PROBABILITY}")
    print("\nCount per category (each must be >= 3):")
    for c, n in _SUMMARY["per_category"].items():
        print(f"  {c:<15} {n:>3}")
    print("\nCount per status (each must be >= 1):")
    for k, n in _SUMMARY["per_status"].items():
        print(f"  {k:<15} {n:>3}")
    print(
        f"\nflagged_for_fraud_review=True: {_SUMMARY['flagged_count']}/{_SUMMARY['n_records']}"
        f" = {_SUMMARY['flagged_pct']}%  (required band 10%–30%)"
    )
    print("\nAll structural thresholds PASS.\n\nFirst 5 records:")
    for r in LOAN_APPLICATIONS[:5]:
        print(" ", r)