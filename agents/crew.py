"""Task 7 — the CrewAI crew: Retrieval Agent, Lookup Agent, Response Composer."""
from __future__ import annotations

import re

import config  # noqa: F401  (telemetry off before crewai import)
from crewai import Crew, Process, Task

from agents.governance import audit_crew, build_agent
from agents.schemas import CrewResponse
from tools.crew_tools import (COMPOSER_ROLE, LOOKUP_ROLE, RETRIEVAL_ROLE, LoanStatusTool,
                              RagLookupTool)


def make_llm():
    if config.MOCK_LLM:
        from agents.mock_crewai_llm import MockCrewLLM
        return MockCrewLLM(model="mock-crew-llm")
    from crewai import LLM  # pragma: no cover - optional real path
    return LLM(model=config.REAL_LLM_MODEL, temperature=0)


_RAG_TOOL = RagLookupTool()
_LOOKUP_TOOL = LoanStatusTool()


def build_agents(llm=None):
    llm = llm or make_llm()
    common = dict(llm=llm, verbose=False, allow_delegation=False, max_iter=4)
    retrieval = build_agent(
        role=RETRIEVAL_ROLE, tools=[_RAG_TOOL],
        goal="Answer Cred loan and banking policy questions strictly from the policy knowledge base.",
        backstory="You search Cred's policy documents and never answer from memory.", **common)
    lookup = build_agent(
        role=LOOKUP_ROLE, tools=[_LOOKUP_TOOL],
        goal="Fetch the current status, amount and escalation score of a specific loan application.",
        backstory="You are the only agent allowed to read loan-application records.", **common)
    composer = build_agent(
        role=COMPOSER_ROLE, tools=[],
        goal="Combine the retrieval and lookup results into one accurate, grounded draft answer.",
        backstory="You write the reply support staff will send. You add no facts of your own.", **common)
    return retrieval, lookup, composer


def _safe(q: str) -> str:
    # CrewAI interpolates {placeholders} in task text; neutralise braces from user input.
    return re.sub(r"[{}]", "", q)


def build_crew(query: str, *, use_retrieval: bool, use_lookup: bool, llm=None) -> Crew:
    retrieval, lookup, composer = build_agents(llm)
    q = _safe(query)
    tasks, ctx = [], []
    if use_retrieval:
        t = Task(description=("Use the rag_lookup tool to answer the member's policy question. "
                              f"Return the tool's JSON output unchanged.\nUSER_QUERY: <<<{q}>>>"),
                 expected_output="The rag_lookup tool's JSON output.", agent=retrieval)
        tasks.append(t); ctx.append(t)
    if use_lookup:
        t = Task(description=("Use the check_loan_application_status tool for the application id in "
                              f"the member's message. Return the tool's JSON output unchanged.\n"
                              f"USER_QUERY: <<<{q}>>>"),
                 expected_output="The check_loan_application_status tool's JSON output.", agent=lookup)
        tasks.append(t); ctx.append(t)
    tasks.append(Task(
        description=("Combine the results in your context into one final answer for the member. "
                     "Use only facts present in the context and conform to the CrewResponse schema.\n"
                     f"USER_QUERY: <<<{q}>>>"),
        expected_output="A JSON object matching the CrewResponse schema.",
        agent=composer, context=ctx, output_pydantic=CrewResponse))
    crew = Crew(agents=[retrieval, lookup, composer], tasks=tasks, process=Process.sequential,
                verbose=False)
    audit_crew(crew)  # least-autonomy check right before kickoff
    return crew


def run_crew(query: str, *, use_retrieval: bool, use_lookup: bool) -> CrewResponse:
    crew = build_crew(query, use_retrieval=use_retrieval, use_lookup=use_lookup)
    result = crew.kickoff()
    raw = result.pydantic if result.pydantic is not None else result.raw
    # Validate in code regardless of what CrewAI already parsed.
    if isinstance(raw, CrewResponse):
        return CrewResponse.model_validate(raw.model_dump())
    return CrewResponse.model_validate_json(raw)
