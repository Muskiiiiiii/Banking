"""Task 14 on its own: the Autogen review stage after the CrewAI Composer's draft.
Shows one draft APPROVED unchanged and drafts REVISED after an ungrounded claim is
deliberately injected — each with a structured ReviewVerdict.
Run: python -m scripts.task14_review
"""
import config  # noqa: F401
from agents.memory import ConversationManager
from agents.review_team import REVIEW_STATS, build_review_team
from agents.schemas import ReviewVerdict
from rag.index import ensure_indexes


def show(r):
    print(f"  crew draft : {r['crew_draft']}")
    print("  review team messages:")
    for line in r["review_transcript"][1:]:  # [0] is the task message
        print(f"    - {line}")
    v = ReviewVerdict.model_validate(r["review"])  # structured verdict, validated
    print(f"  VERDICT    : approved={v.approved}")
    print(f"    final_answer : {v.final_answer}")
    print(f"    reason       : {v.reason}")
    return v


def main():
    ensure_indexes()
    team = build_review_team()
    print(f"Team: {type(team).__name__}  participants={[p.name for p in team._participants]}  max_turns=2")
    print("Final_Editor: output_content_type=ReviewVerdict; team: "
          "custom_message_types=[StructuredMessage[ReviewVerdict]]")
    print(f"ReviewVerdict fields: {list(ReviewVerdict.model_fields)}")

    cm = ConversationManager()
    q = "What is the foreclosure charge on a personal loan?"

    print("\n" + "=" * 90 + "\nCASE 1 — grounded draft -> expected APPROVED unchanged\n" + "=" * 90)
    r1 = cm.ask("review-1", q)
    v1 = show(r1)
    print(f"  unchanged: {v1.final_answer == r1['crew_draft']}")

    print("\n" + "=" * 90 + "\nCASE 2 — same question, false claim injected into the draft -> expected REVISED\n" + "=" * 90)
    fake = "Personal loans carry a 1% foreclosure charge with no lock-in period."
    r2 = cm.ask("review-2", q, inject_into_draft=fake)
    v2 = show(r2)
    print(f"  injected claim removed: {fake not in v2.final_answer}")

    print("\n" + "=" * 90 + "\nCASE 3 — status answer with a wrong amount injected -> expected REVISED\n" + "=" * 90)
    fake2 = "The sanctioned amount was later raised to Rs 50,00,000."
    r3 = cm.ask("review-3", "What is the status of CRED-LA-0004?", inject_into_draft=fake2)
    v3 = show(r3)
    print(f"  injected claim removed: {fake2 not in v3.final_answer}")

    print("\n" + "=" * 90 + "\nCASE 4 — KNOWN LIMITATION: a one-word falsehood with no number "
          "(expected to be MISSED)\n" + "=" * 90)
    fake3 = "Floating-rate home loans taken by companies carry no prepayment penalty."
    r4 = cm.ask("review-4", "Is there a prepayment penalty on a floating-rate home loan?",
                inject_into_draft=fake3)
    v4 = show(r4)
    print(f"  false claim ('companies' instead of 'individuals') still in final answer: "
          f"{fake3 in v4.final_answer}  <- the mock reviewer cannot catch this")

    print(f"\nreviewer calls={REVIEW_STATS['reviewer_calls']}  editor calls={REVIEW_STATS['editor_calls']} "
          f"(4 reviews; max_turns=2 lets both agents speak exactly once per review)")


if __name__ == "__main__":
    main()
