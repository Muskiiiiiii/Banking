"""Task 15 — Application-layer governance: principle of least autonomy + risk classification.

Guard: a tool can declare `restricted_to_roles`. Every agent in this project is created
through `build_agent()`, which refuses to attach a restricted tool to any other role,
and `audit_crew()` re-checks every agent AND every task-level tool override right before
`crew.kickoff()`. CrewAI lets a Task carry its own `tools=[...]` that override the agent's,
so checking agents alone would leave a gap — the audit closes it. The lookup tool is
restricted to the Lookup Agent, so no other agent can ever receive it, and a crew that
somehow did would be refused before it runs.
"""
from __future__ import annotations

RISK_CLASSIFICATION = {
    "level": "High",
    "justification": (
        "Under the Low/Medium/High scheme this system is High risk. It reads loan-application "
        "records that contain financial data (loan amounts, application status and fraud-review "
        "flags) and the answers it gives to support staff can shape how a member's loan or fraud "
        "dispute is handled, which is exactly the 'financial data' case the scheme lists as High. "
        "Members may also type PAN, Aadhaar or bank-account numbers into the chat. It is not a "
        "Medium-risk ticket summariser because a wrong answer about prepayment charges, KYC or "
        "dispute deadlines can cost a member money or legal protection. The controls that follow "
        "from High risk are: PII masking before any model or log sees the text, least-autonomy "
        "tool access, a groundedness refusal, a second independent review team, a per-request "
        "cost cap, structured JSONL audit logs, and human escalation via the escalation score; "
        "the agent informs staff and never approves, rejects or disburses a loan itself."
    ),
}


class LeastAutonomyViolation(PermissionError):
    pass


def check_tool_permitted(role: str, tool) -> None:
    allowed = getattr(tool, "restricted_to_roles", None)
    if allowed and role not in allowed:
        raise LeastAutonomyViolation(
            f"Least-autonomy guard: agent '{role}' may not be given tool '{tool.name}'. "
            f"Only {sorted(allowed)} may call it.")


def build_agent(*, role: str, tools: list | None = None, **kwargs):
    from crewai import Agent
    for t in tools or []:
        check_tool_permitted(role, t)
    return Agent(role=role, tools=tools or [], **kwargs)


def audit_crew(crew) -> list[str]:
    """Return a list of (role, tool) wirings; raise if any violates least autonomy."""
    wiring = []
    for agent in crew.agents:
        for t in agent.tools or []:
            check_tool_permitted(agent.role, t)
            wiring.append(f"{agent.role} -> {t.name}")
    for task in crew.tasks:
        for t in task.tools or []:
            role = task.agent.role if task.agent else "<unassigned>"
            check_tool_permitted(role, t)
    return wiring
