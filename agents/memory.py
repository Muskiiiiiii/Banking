"""Task 8 — session memory with LangChain's InMemoryChatMessageHistory +
RunnableWithMessageHistory (in-process only; does not survive a restart, by design).

RunnableWithMessageHistory emits a LangChainDeprecationWarning pointing to LangGraph
persistence. That is expected per the brief and is deliberately not silenced.

Only the MASKED user text is stored in history, so PAN/Aadhaar/account numbers never
enter memory.
"""
from __future__ import annotations

import threading

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from agents.guardrails import mask_pii
from agents.pipeline import get_agent


class ConversationManager:
    def __init__(self, agent=None):
        self.agent = agent or get_agent()
        self._store: dict[str, InMemoryChatMessageHistory] = {}
        self._lock = threading.Lock()
        self.chain = RunnableWithMessageHistory(
            RunnableLambda(self._run),
            self.get_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="output",
        )

    def get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = InMemoryChatMessageHistory()
            return self._store[session_id]

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

    def _run(self, inp: dict) -> dict:
        result = self.agent.handle(inp["raw_input"], history=inp.get("history", []),
                                   **inp.get("options", {}))
        return {"output": result["response"].answer, "result": result}

    def ask(self, session_id: str, text: str, **options) -> dict:
        masked, _ = mask_pii(text)
        out = self.chain.invoke({"input": masked, "raw_input": text, "options": options},
                                config={"configurable": {"session_id": session_id}})
        return out["result"]
