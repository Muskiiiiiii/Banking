"""Task 7 on its own: build the 3-agent CrewAI crew and run it with .kickoff() on
different queries, showing which tool each query actually invoked.
Run: python -m scripts.task7_crew
"""
import os

import config
from agents.crew import build_crew, run_crew
from agents.mock_crewai_llm import MOCK_CREW_STATS, MockCrewLLM
from rag.index import ensure_indexes
from tools.crew_tools import LoanStatusTool, RagLookupTool, start_tool_log

QUERIES = [
    # (query, use_retrieval, use_lookup)
    ("Is there a prepayment penalty on a floating-rate home loan?", True, False),
    ("What is the status of CRED-LA-0007?", False, True),
    ("What is the status of CRED-LA-0001 and when will it be disbursed?", True, True),
]


def main():
    print(f"MOCK_LLM={config.MOCK_LLM}  CREWAI_DISABLE_TELEMETRY={os.environ['CREWAI_DISABLE_TELEMETRY']}  "
          f"OTEL_SDK_DISABLED={os.environ['OTEL_SDK_DISABLED']}")
    ensure_indexes()

    crew = build_crew("demo", use_retrieval=True, use_lookup=True)
    print("\nCrew agents:")
    for a in crew.agents:
        print(f"  {a.role:<24} tools={[t.name for t in a.tools]}  llm={type(a.llm).__name__}")

    print("\nMOCK_LLM pitfall checks from the brief:")
    template_only = [
        {"role": "system", "content": "...Action Input: {...}\nObservation: the result of the action\n..."},
        {"role": "user", "content": "Current Task: ... USER_QUERY: <<<x>>>\n\nThought:"}]
    print(f"  (1) tool result read from the system-prompt template on the first call -> "
          f"{MockCrewLLM._observation_from_own_output(template_only)!r}  (None = template correctly ignored)")
    for tool in (RagLookupTool(), LoanStatusTool()):
        args = MockCrewLLM._resolve_args(tool, "Can I prepay my loan CRED-LA-0004?")
        print(f"  (2) {tool.name!r:<34} declared fields={list(tool.args_schema.model_fields)} -> args {args}")

    for q, use_r, use_l in QUERIES:
        log = start_tool_log()
        before = MOCK_CREW_STATS["calls"]
        resp = run_crew(q, use_retrieval=use_r, use_lookup=use_l)  # crew.kickoff() inside
        print(f"\nQUERY : {q}")
        print(f"tools invoked : {[c['tool'] + str(c['args']) for c in log]}")
        print(f"LLM calls     : {MOCK_CREW_STATS['calls'] - before}")
        print(f"intent={resp.intent}  grounded={resp.grounded}  escalate={resp.escalate}  sources={resp.sources}")
        print(f"DRAFT ANSWER  : {resp.answer}")


if __name__ == "__main__":
    main()
