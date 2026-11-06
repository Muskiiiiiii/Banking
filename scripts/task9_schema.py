"""Task 9 on its own: every crew response must conform to the Pydantic CrewResponse schema,
and is validated in code.
Run: python -m scripts.task9_schema
"""
import json

import config  # noqa: F401
from pydantic import ValidationError

from agents.memory import ConversationManager
from agents.schemas import CrewResponse
from rag.index import ensure_indexes

QUERIES = [
    "What is the late-payment fee on the credit card?",          # policy
    "What is the status of CRED-LA-0004?",                       # status
    "What is the status of CRED-LA-0001 and when will it be disbursed?",  # both
    "What is the best pizza place in Bengaluru?",                # fallback
    "What is its loan amount?",                                  # clarify (fresh session)
]


def main():
    ensure_indexes()
    print("CrewResponse JSON schema (what every answer must look like):")
    schema = CrewResponse.model_json_schema()
    for name, spec in schema["properties"].items():
        req = "required" if name in schema.get("required", []) else "optional"
        print(f"  {name:<16} {req:<9} {json.dumps({k: v for k, v in spec.items() if k != 'title'})[:90]}")

    cm = ConversationManager()
    print("\nValidating real crew responses:")
    for i, q in enumerate(QUERIES):
        r = cm.ask(f"schema-{i}", q, review=False)
        resp = r["response"]
        CrewResponse.model_validate(resp.model_dump())  # explicit re-validation
        print(f"\n  Q: {q}\n  VALID ✓ intent={resp.intent} grounded={resp.grounded} "
              f"tools={resp.tools_used}\n  {resp.model_dump_json()[:260]}...")

    print("\nNegative tests (must be rejected):")
    bad_cases = {
        "empty answer + unknown intent + non-bool grounded":
            {"answer": "", "intent": "chit-chat", "grounded": "maybe"},
        "escalation_score out of [0, 1]":
            {"answer": "x", "intent": "status", "grounded": True,
             "record_lookup": {"record_id": "CRED-LA-0001", "found": True, "escalation_score": 1.7}},
        "missing required field 'grounded'": {"answer": "x", "intent": "policy"},
    }
    for label, payload in bad_cases.items():
        try:
            CrewResponse.model_validate(payload)
            print(f"  {label}: ACCEPTED (unexpected)")
        except ValidationError as e:
            print(f"  {label}: REJECTED -> {e.error_count()} error(s): "
                  + "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()))


if __name__ == "__main__":
    main()
