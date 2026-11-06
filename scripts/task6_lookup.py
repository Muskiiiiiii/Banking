"""Task 6 on its own: check_loan_application_status and the designed escalation score.
Run: python -m scripts.task6_lookup
"""
import json
from collections import Counter

import config
from dataset import LOAN_APPLICATIONS
from tools.loan_status import (ESCALATION_THRESHOLD, OPEN_DAYS, P80_OPEN_DAYS,
                               check_loan_application_status, escalation_score)


def main():
    print(f"escalation_score = {config.ESCALATION_W_FRAUD} * fraud + {config.ESCALATION_W_STALENESS} * staleness")
    print("  fraud     = 1 if flagged_for_fraud_review else 0")
    print("  staleness = days_since_created / 30 for open statuses (Submitted, Under Review, Approved); 0 for Rejected/Disbursed")
    print(f"\nOpen applications: {len(OPEN_DAYS)} of {len(LOAN_APPLICATIONS)}")
    print(f"Sorted days_since_created (open): {sorted(OPEN_DAYS)}")
    print(f"80th percentile = {P80_OPEN_DAYS:.1f} days")
    print(f"threshold = {config.ESCALATION_W_STALENESS} * {P80_OPEN_DAYS:.1f} / 30 = {ESCALATION_THRESHOLD}")

    esc = [r for r in LOAN_APPLICATIONS if escalation_score(r) >= ESCALATION_THRESHOLD]
    why = Counter("fraud flag" if r["flagged_for_fraud_review"] else "stale open application" for r in esc)
    print(f"\nEscalated: {len(esc)}/{len(LOAN_APPLICATIONS)} ({100 * len(esc) / len(LOAN_APPLICATIONS):.0f}%) -> {dict(why)}")

    print("\nSample lookups:")
    for rid in ["CRED-LA-0007", "CRED-LA-0001", "CRED-LA-0004", "CRED-LA-0042", "cred-la-0013", "CRED-LA-9999"]:
        print(" ", json.dumps(check_loan_application_status(rid)))


if __name__ == "__main__":
    main()
