"""Task 11 — FastAPI deployment.

  GET  /health          liveness + mode info
  POST /ask             one question (optional session_id for multi-turn memory)
  POST /add-document    add a policy document to BOTH Chroma collections
  WS   /ws/chat         real-time multi-turn chat; a client disconnecting mid-conversation
                        is caught (WebSocketDisconnect) and the server keeps serving others

Run:  uvicorn api.main:app --port 8000
"""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import config  # noqa: F401  (telemetry off first)
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agents.memory import ConversationManager
from agents.schemas import CrewResponse, ReviewVerdict
from api.logging_utils import Timer, log_request, new_trace_id
from rag.index import add_document, ensure_indexes
from rag.retrieval import default_generator

conversations: ConversationManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global conversations
    await run_in_threadpool(ensure_indexes)
    conversations = ConversationManager()
    yield


app = FastAPI(title="Cred Domain Support Agent", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def log_rejected_request(request: Request, exc: RequestValidationError):
    """Requests rejected by Pydantic validation (422) never reach an endpoint, so they are
    logged here — every request gets exactly one log line. The body goes through the same
    PII masking inside log_request()."""
    trace_id = new_trace_id()
    body = exc.body if isinstance(exc.body, (str, type(None))) else json.dumps(exc.body, default=str)
    log_request(trace_id=trace_id, endpoint=f"{request.method} {request.url.path}", request_text=body,
                status="rejected_validation", duration_ms=0,
                errors=[f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()])
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors()),
                                                  "trace_id": trace_id})


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    trace_id: str
    session_id: str
    response: CrewResponse
    review: Optional[ReviewVerdict] = None
    tool_calls: list[dict] = Field(default_factory=list)
    used_tokens: int
    cost_usd: float
    latency_ms: float


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(pattern=r"^[A-Za-z0-9_\-]{3,64}$")
    title: str = Field(min_length=3, max_length=200)
    text: str = Field(min_length=20, max_length=5_000)


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    chunks_upserted: dict[str, int]


def _answer(query: str, session_id: str, trace_id: str) -> AskResponse:
    r = conversations.ask(session_id, query)
    return AskResponse(trace_id=trace_id, session_id=session_id, response=r["response"],
                       review=r["review"], tool_calls=r["tool_calls"],
                       used_tokens=r["budget"]["used_tokens"], cost_usd=r["budget"]["cost_usd"],
                       latency_ms=r["latency_ms"])


@app.get("/health")
def health():
    return {"status": "ok", "mock_llm": config.MOCK_LLM, "collection": config.DEPLOYED_COLLECTION}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    trace_id, sid = new_trace_id(), req.session_id or uuid.uuid4().hex[:12]
    with Timer() as t:
        try:
            resp = await run_in_threadpool(_answer, req.query, sid, trace_id)
        except Exception as e:  # log, then surface as 500
            log_request(trace_id=trace_id, endpoint="POST /ask", request_text=req.query,
                        status="error", duration_ms=0, session_id=sid, error=type(e).__name__)
            raise HTTPException(status_code=500, detail=f"internal error (trace {trace_id})")
    log_request(trace_id=trace_id, endpoint="POST /ask", request_text=req.query, status="ok",
                duration_ms=t.ms, session_id=sid, intent=resp.response.intent,
                flags=resp.response.guardrail_flags, tools=[c["tool"] for c in resp.tool_calls],
                used_tokens=resp.used_tokens)
    return resp


@app.post("/add-document", response_model=AddDocumentResponse)
async def add_doc(req: AddDocumentRequest):
    trace_id = new_trace_id()
    with Timer() as t:
        counts = await run_in_threadpool(add_document, req.doc_id, req.title, req.text)
        default_generator().clear_cache()  # new knowledge invalidates cached answers
    log_request(trace_id=trace_id, endpoint="POST /add-document", request_text=req.text,
                status="ok", duration_ms=t.ms, doc_id=req.doc_id, chunks=counts)
    return AddDocumentResponse(trace_id=trace_id, doc_id=req.doc_id, chunks_upserted=counts)


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    sid = websocket.query_params.get("session_id") or uuid.uuid4().hex[:12]
    conn_trace = new_trace_id()
    log_request(trace_id=conn_trace, endpoint="WS /ws/chat", request_text=None, status="connected",
                duration_ms=0, session_id=sid)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                query = json.loads(raw).get("query", "")
            except (json.JSONDecodeError, AttributeError):
                query = raw
            trace_id = new_trace_id()
            with Timer() as t:
                resp = await run_in_threadpool(_answer, query, sid, trace_id)
            log_request(trace_id=trace_id, endpoint="WS /ws/chat", request_text=query, status="ok",
                        duration_ms=t.ms, session_id=sid, intent=resp.response.intent,
                        flags=resp.response.guardrail_flags)
            await websocket.send_text(resp.model_dump_json())
    except WebSocketDisconnect as e:
        # Client went away mid-conversation: log it and end only this handler.
        log_request(trace_id=conn_trace, endpoint="WS /ws/chat", request_text=None,
                    status="client_disconnected", duration_ms=0, session_id=sid, code=e.code)
    except RuntimeError as e:  # send after close (disconnect raced with our reply)
        log_request(trace_id=conn_trace, endpoint="WS /ws/chat", request_text=None,
                    status="client_disconnected", duration_ms=0, session_id=sid, detail=str(e)[:120])
