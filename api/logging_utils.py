"""Task 12 — ELK-style structured logging: one JSON-Lines entry per request.

The logged request text goes through the SAME mask_pii() the model input goes through,
so a PAN / Aadhaar / bank-account number never reaches disk in the clear. The raw text
is never passed to this module.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone

import config
from agents.guardrails import mask_pii

_lock = threading.Lock()


def new_trace_id() -> str:
    return uuid.uuid4().hex


def log_request(*, trace_id: str, endpoint: str, request_text: str | None, status: str,
                duration_ms: float, **fields) -> dict:
    masked, pii = mask_pii(request_text or "")
    entry = {
        "@timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "trace_id": trace_id,
        "service": "cred-support-agent",
        "endpoint": endpoint,
        "status": status,
        "duration_ms": round(duration_ms, 2),
        "request_text": masked,
        "pii_masked": pii,
        **fields,
    }
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False, default=str)
    with _lock, open(config.REQUEST_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return entry


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.t0) * 1000
        return False
