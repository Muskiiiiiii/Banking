"""MOCK_LLM for CrewAI: a deterministic, keyless, offline LLM built on CrewAI's documented
extension point `crewai.llms.base_llm.BaseLLM`.

It speaks CrewAI's ReAct text protocol (supports_function_calling() -> False):
  1st call for an agent that has tools  -> "Action: <tool>\nAction Input: {...}"
  after the tool ran                    -> "Final Answer: <observation passed through>"
  agent without tools (Composer)        -> "Final Answer: <CrewResponse JSON>"

Two pitfalls from the brief are handled explicitly:
  (1) CrewAI's ReAct system prompt literally contains "Observation: the result of the action".
      We never search the whole conversation for "Observation:". We read ONLY assistant-role
      messages (text this model generated, to which CrewAI appends the real tool result) and
      take the text after "Observation:" there. Before any tool has run there is no assistant
      message, so the template text can never be mistaken for a tool result.
  (2) Tool arguments are dispatched from each tool's declared `args_schema` fields (a `query`
      field gets the member's question; a `record_id` field gets the extracted id), never by
      matching the tool's name — so a tool called `rag_lookup` is not mistaken for the lookup tool.
"""
from __future__ import annotations

import json
import re
from typing import Any

from crewai.llms.base_llm import BaseLLM

from agents import budget
from agents.composer_logic import compose
from tools.loan_status import RECORD_ID_RE, extract_record_id

MEMORY_SUFFIX_RE = re.compile(r"\(regarding application [^)]*\)", re.IGNORECASE)
USER_QUERY_RE = re.compile(r"USER_QUERY:\s*<<<(.*?)>>>", re.DOTALL)
MOCK_CREW_STATS = {"calls": 0}


def _content(m: Any) -> str:
    c = m.get("content", "") if isinstance(m, dict) else str(m)
    if isinstance(c, list):  # multimodal-style content blocks
        c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def _json_objects(text: str) -> list[dict]:
    """Extract every top-level JSON object embedded in free text."""
    dec, out, i = json.JSONDecoder(), [], 0
    while (j := text.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(text, j)
            if isinstance(obj, dict):
                out.append(obj)
            i = end
        except json.JSONDecodeError:
            i = j + 1
    return out


class MockCrewLLM(BaseLLM):
    model: str = "mock-crew-llm"
    llm_type: str = "mock"

    def supports_function_calling(self) -> bool:  # force the ReAct text protocol
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 128_000

    def get_input_tokens_count(self, messages, tools=None, response_model=None) -> int:
        text = messages if isinstance(messages, str) else "\n".join(_content(m) for m in messages)
        return budget.estimate_tokens(text)  # same ~4 chars/token heuristic the budget uses

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _observation_from_own_output(messages: list) -> str | None:
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "assistant":
                text = _content(m)
                if "Observation:" in text:
                    return text.split("Observation:", 1)[1].strip()
        return None

    @staticmethod
    def _resolve_args(tool, user_query: str) -> dict | None:
        schema = getattr(tool, "args_schema", None)
        if schema is None:
            return None
        # The knowledge-base search gets the question text without record ids or the
        # memory-resolution suffix (they carry no policy meaning and dilute the embedding).
        policy_text = " ".join(RECORD_ID_RE.sub("", MEMORY_SUFFIX_RE.sub("", user_query)).split()).strip(" ?,.") + "?"
        resolvers = {"query": lambda: policy_text, "record_id": lambda: extract_record_id(user_query)}
        args = {}
        for fname, field in schema.model_fields.items():
            value = resolvers[fname]() if fname in resolvers else None
            if value is None:
                if field.is_required():
                    return None  # this tool cannot be called for this query
                continue
            args[fname] = value
        return args

    # --------------------------------------------------------------------- call
    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None, **kwargs) -> str:
        MOCK_CREW_STATS["calls"] += 1
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        prompt_text = "\n".join(_content(m) for m in messages)
        user_text = "\n".join(_content(m) for m in messages if isinstance(m, dict) and m.get("role") == "user")
        m = USER_QUERY_RE.search(user_text)
        user_query = m.group(1).strip() if m else ""

        agent_tools = list((from_task.tools if from_task is not None and from_task.tools else None)
                           or (from_agent.tools if from_agent is not None else []) or [])
        observation = self._observation_from_own_output(messages)

        if agent_tools and observation is None:
            for tool in agent_tools:
                args = self._resolve_args(tool, user_query)
                if args is not None:
                    out = (f"Thought: I should call {tool.name} with the member's input.\n"
                           f"Action: {tool.name}\nAction Input: {json.dumps(args)}")
                    budget.charge("crew:" + getattr(from_agent, "role", "?"), prompt_text, out)
                    return out
            out = ('Thought: I now know the final answer\nFinal Answer: '
                   + json.dumps({"tool": None, "note": "No tool applies to this query."}))
        elif observation is not None:
            out = f"Thought: I now know the final answer\nFinal Answer: {observation}"
        else:
            # Composer: combine prior task outputs that CrewAI placed in the task context.
            outputs = [o for o in _json_objects(user_text) if o.get("tool")]
            out = ("Thought: I now know the final answer\nFinal Answer: "
                   + compose(user_query, outputs).model_dump_json())
        budget.charge("crew:" + getattr(from_agent, "role", "?"), prompt_text, out)
        return out

    async def acall(self, *args, **kwargs) -> str:
        return self.call(*args, **kwargs)
