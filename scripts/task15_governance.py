"""Task 15 on its own — four-layer AI-governance model, the two layers the brief asks to show:
  Application layer: least autonomy (only the Lookup Agent may call the lookup tool) +
                     Low/Medium/High risk classification
  Runtime layer:     per-request token / cost budget cap that rejects an oversized request
Run: python -m scripts.task15_governance
"""
import config
from agents import budget
from agents.crew import build_crew
from agents.governance import RISK_CLASSIFICATION, LeastAutonomyViolation, audit_crew, build_agent
from agents.memory import ConversationManager
from rag import llm
from rag.index import ensure_indexes
from tools.crew_tools import COMPOSER_ROLE, LOOKUP_ROLE, RETRIEVAL_ROLE, LoanStatusTool


def hr(t):
    print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90)


def main():
    ensure_indexes()

    hr("APPLICATION LAYER — least autonomy")
    print(f"LoanStatusTool.restricted_to_roles = {sorted(LoanStatusTool.restricted_to_roles)}")
    crew = build_crew("x", use_retrieval=True, use_lookup=True)
    print(f"actual crew wiring (audit_crew): {audit_crew(crew)}\n")

    print("Attempt 1 — give the lookup tool to other agents at construction time:")
    for role in (RETRIEVAL_ROLE, COMPOSER_ROLE):
        try:
            build_agent(role=role, goal="g", backstory="b", tools=[LoanStatusTool()])
            print(f"  {role}: ALLOWED  <-- would be a violation")
        except LeastAutonomyViolation as e:
            print(f"  {role}: BLOCKED -> {e}")

    print("\nAttempt 2 — sneak it in afterwards through a Task-level tools override:")
    crew = build_crew("x", use_retrieval=True, use_lookup=False)
    crew.tasks[0].tools = [LoanStatusTool()]
    try:
        audit_crew(crew)
        print("  ALLOWED  <-- would be a violation")
    except LeastAutonomyViolation as e:
        print(f"  BLOCKED before kickoff -> {e}")

    print("\nControl — the Lookup Agent itself:")
    a = build_agent(role=LOOKUP_ROLE, goal="g", backstory="b", tools=[LoanStatusTool()])
    print(f"  {a.role}: allowed, tools={[t.name for t in a.tools]}")

    hr("APPLICATION LAYER — risk classification")
    print(f"Risk level: {RISK_CLASSIFICATION['level']}\n\n{RISK_CLASSIFICATION['justification']}")

    hr("RUNTIME LAYER — per-request token / cost budget cap")
    print(f"Caps per request: input <= {config.MAX_INPUT_TOKENS_PER_REQUEST} tokens; total <= "
          f"{config.MAX_TOTAL_TOKENS_PER_REQUEST} tokens and <= ${config.MAX_COST_PER_REQUEST_USD} "
          f"(notional ${config.PRICE_PER_1K_TOKENS_USD} per 1K tokens; ~4 characters per token)")
    cm = ConversationManager()

    r = cm.ask("budget-normal", "What is the late-payment fee on the credit card?")
    print(f"\n(1) normal request: used {r['budget']['used_tokens']} tokens "
          f"(${r['budget']['cost_usd']}) -> answered, flags={r['response'].guardrail_flags}")

    oversized = "Please explain every policy in full detail. " * 300
    calls_before = llm.LLM_STATS["calls"]
    r = cm.ask("budget-oversized", oversized)
    print(f"\n(2) deliberately oversized request: {len(oversized)} characters "
          f"(~{budget.estimate_tokens(oversized)} tokens)")
    print(f"    answer : {r['response'].answer}")
    print(f"    flags  : {r['response'].guardrail_flags}")
    print(f"    tokens charged = {r['budget']['used_tokens']}; LLM calls made = "
          f"{llm.LLM_STATS['calls'] - calls_before}")

    r = cm.ask("budget-tight", "What is the status of CRED-LA-0007?",
               budget_overrides={"max_total_tokens": 1500, "max_cost_usd": 0.0009})
    print("\n(3) normal-size request under a deliberately tiny total cap (1,500 tokens):")
    print(f"    answer : {r['response'].answer}")
    print(f"    flags  : {r['response'].guardrail_flags}")
    print(f"    tokens used when stopped = {r['budget']['used_tokens']} (cap 1,500 -> never exceeded)")


if __name__ == "__main__":
    main()
