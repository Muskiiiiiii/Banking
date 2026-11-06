"""Task 11 on its own: FastAPI HTTP endpoints + WebSocket chat that survives a client
disconnecting mid-conversation.

Section A uses FastAPI's in-process TestClient. Section B starts a real uvicorn server on
127.0.0.1 and drops a real WebSocket connection mid-conversation to show the server keeps
serving other clients.
Run: python -m scripts.task11_api
"""
import json
import os
import socket
import subprocess
import sys
import time

import config
from fastapi.testclient import TestClient


def hr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def short(resp_json: dict) -> str:
    r = resp_json["response"]
    return (f"intent={r['intent']} flags={r['guardrail_flags']} tools={r['tools_used']}\n"
            f"      answer: {r['answer']}")


def remove_demo_doc():
    """The demo adds '14_gold_loan' to the persistent ChromaDB. Remove it first so every run
    shows the same before/after behaviour."""
    from rag.index import STRATEGY_TO_COLLECTION, get_collection
    for cname in STRATEGY_TO_COLLECTION.values():
        col = get_collection(cname)
        ids = [i for i in col.get()["ids"] if i.startswith("14_gold_loan::")]
        if ids:
            col.delete(ids=ids)


def main():
    remove_demo_doc()
    from api.main import app

    hr("TASK 11 (A) — HTTP endpoints via TestClient")
    with TestClient(app) as client:
        print("GET /health ->", client.get("/health").json())

        r = client.post("/ask", json={"query": "What is the late-payment fee on the credit card?",
                                      "session_id": "http-1"})
        print(f"\nPOST /ask (policy) -> {r.status_code}\n   {short(r.json())}")
        r = client.post("/ask", json={"query": "Status of CRED-LA-0035 please. My account no is 987654321012.",
                                      "session_id": "http-1"})
        print(f"\nPOST /ask (status + PII) -> {r.status_code}\n   {short(r.json())}")
        r = client.post("/ask", json={"query": ""})
        print(f"\nPOST /ask with empty query -> {r.status_code} (Pydantic validation: "
              f"{r.json()['detail'][0]['msg']})")

        q = "Can I get a gold loan from Cred?"
        print(f"\nBefore adding a document: POST /ask {q!r}")
        print("   ", short(client.post("/ask", json={"query": q}).json()))
        doc = {"doc_id": "14_gold_loan", "title": "Gold loan rules",
               "text": ("Gold loans are offered against gold jewellery of 18 to 24 carat purity. "
                        "The loan amount is capped at 75% of the appraised value of the gold, as RBI requires. "
                        "Gold loans carry interest of 9.9% to 18% per annum and tenures of 3 to 24 months.")}
        r = client.post("/add-document", json=doc)
        print(f"\nPOST /add-document -> {r.status_code} {r.json()}")
        print(f"After adding: POST /ask {q!r}")
        print("   ", short(client.post("/ask", json={"query": q}).json()))

        hr("TASK 11 (A) — WebSocket /ws/chat: multi-turn, then a client disconnects mid-conversation")
        with client.websocket_connect("/ws/chat?session_id=ws-alice") as ws:
            ws.send_text(json.dumps({"query": "Please check CRED-LA-0034"}))
            print("alice turn 1 ->", short(json.loads(ws.receive_text())))
            ws.send_text(json.dumps({"query": "Is it flagged for escalation?"}))
            print("alice turn 2 (memory) ->", short(json.loads(ws.receive_text())))
            ws.send_text(json.dumps({"query": "And what fee applies if an EMI bounces?"}))
            ws.close(code=1001)  # client leaves mid-conversation, reply to turn 3 never read
            print("alice sent turn 3 and closed the socket WITHOUT waiting for the reply")
        time.sleep(1)
        with client.websocket_connect("/ws/chat?session_id=ws-bob") as ws:
            ws.send_text("What are the KYC refresh intervals?")
            print("bob (new client, same server) ->", short(json.loads(ws.receive_text())))
        print("GET /health after disconnect ->", client.get("/health").json())

    hr("TASK 11 (B) — real uvicorn server + real WebSocket disconnect")
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    env = dict(os.environ)
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1",
                             "--port", str(port), "--log-level", "warning"], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        import httpx
        from websockets.sync.client import connect
        for _ in range(240):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        print(f"uvicorn up on port {port}:", httpx.get(f"http://127.0.0.1:{port}/health").json())
        ws = connect(f"ws://127.0.0.1:{port}/ws/chat?session_id=real-1")
        ws.send(json.dumps({"query": "What is the status of CRED-LA-0010?"}))
        print("real client 1 ->", short(json.loads(ws.recv(timeout=120))))
        ws.send(json.dumps({"query": "How long do fraud disputes take?"}))
        ws.socket.close()  # hard drop: no close handshake, reply still in flight
        print("real client 1 dropped its TCP socket mid-conversation")
        time.sleep(3)
        print("server process still running:", proc.poll() is None)
        with connect(f"ws://127.0.0.1:{port}/ws/chat?session_id=real-2") as ws2:
            ws2.send(json.dumps({"query": "What is the minimum balance for a salary account?"}))
            print("real client 2 ->", short(json.loads(ws2.recv(timeout=120))))
        r = httpx.post(f"http://127.0.0.1:{port}/ask", json={"query": "What is the NRO repatriation limit?"},
                       timeout=120)
        print("POST /ask over real HTTP ->", r.status_code, short(r.json()))
    finally:
        proc.terminate(); proc.wait(timeout=20)
        remove_demo_doc()  # leave the persistent index exactly as kb/ defines it


if __name__ == "__main__":
    main()
