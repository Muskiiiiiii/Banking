"""Central configuration for the Cred Domain Support Agent.

Importing this module FIRST (every entry point does) guarantees that:
  * CrewAI / OpenTelemetry telemetry is disabled before crewai is imported
    (otherwise crew.kickoff() attempts an outbound network call);
  * ChromaDB anonymous telemetry is disabled;
  * MOCK_LLM defaults to ON, so nothing needs an API key or network access.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- zero-network guarantees (must run before crewai / chromadb import) -----
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # chromadb
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent

# --- LLM mode ----------------------------------------------------------------
# MOCK_LLM=1 (default): deterministic, keyless, offline. All graded transcripts use this.
# MOCK_LLM=0 + OPENAI_API_KEY: optional real-LLM wiring (not used for grading).
MOCK_LLM: bool = os.getenv("MOCK_LLM", "1").strip().lower() not in {"0", "false", "no"}
REAL_LLM_MODEL: str = os.getenv("REAL_LLM_MODEL", "gpt-4o-mini")

# --- Embeddings / vector store ----------------------------------------------
# Free local SentenceTransformers model. Set EMBED_MODEL to a local folder path
# to run fully offline after the one-time model download.
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR: Path = Path(os.getenv("CHROMA_DIR", ROOT / "chroma_db"))
KB_DIR: Path = ROOT / "kb"

COLLECTION_FIXED = "cred_kb_fixed"        # fixed-size-with-overlap chunks
COLLECTION_SENTENCE = "cred_kb_sentence"  # sentence-based chunks

FIXED_CHUNK_SIZE = 200     # characters
FIXED_CHUNK_OVERLAP = 50   # characters
SENTENCES_PER_CHUNK = 2    # sentence-based strategy groups 2 sentences per chunk

TOP_K = 3

# Deployed collection = the strategy recommended by Task 5 (see README / transcripts/part1).
DEPLOYED_COLLECTION = os.getenv("DEPLOYED_COLLECTION", COLLECTION_FIXED)

# --- "I don't know" threshold (Task 4) -------------------------------------
# Calibrated empirically with scripts/calibrate_threshold.py — see README for the
# measured in-scope and out-of-scope top-1 cosine similarities. The value sits in
# the gap between the two observed clusters.
IDK_THRESHOLD: float = float(os.getenv("IDK_THRESHOLD", "0.42"))

# Claim-level support threshold used by the groundedness check and the Autogen
# reviewer: an answer sentence counts as supported when its best cosine match to
# a retrieved context sentence is >= this value AND every number it states also
# appears in the context.
CLAIM_SUPPORT_THRESHOLD: float = float(os.getenv("CLAIM_SUPPORT_THRESHOLD", "0.75"))

# --- Escalation (Task 6) -----------------------------------------------------
ESCALATION_W_FRAUD = 0.6
ESCALATION_W_STALENESS = 0.4
# The escalation threshold is DERIVED from the generated data in tools/loan_status.py:
#   threshold = W_STALENESS * (p80 of days_since_created among open applications) / 30

# --- Runtime governance (Task 15) -------------------------------------------
MAX_INPUT_TOKENS_PER_REQUEST = int(os.getenv("MAX_INPUT_TOKENS_PER_REQUEST", "400"))
MAX_TOTAL_TOKENS_PER_REQUEST = int(os.getenv("MAX_TOTAL_TOKENS_PER_REQUEST", "20000"))
# Notional price used for the cost cap (USD per 1K tokens); real pricing varies by model.
PRICE_PER_1K_TOKENS_USD = float(os.getenv("PRICE_PER_1K_TOKENS_USD", "0.0006"))
MAX_COST_PER_REQUEST_USD = float(os.getenv("MAX_COST_PER_REQUEST_USD", "0.012"))

# --- Logging -----------------------------------------------------------------
LOG_DIR: Path = Path(os.getenv("LOG_DIR", ROOT / "logs"))
REQUEST_LOG_FILE: Path = LOG_DIR / "requests.jsonl"
