"""Task 8 on its own: multi-turn session memory, plus a separate fresh session where the
state is correctly absent.
(review=False here only because the Autogen review stage is added in Task 14.)
Run: python -m scripts.task8_memory
"""
import config  # noqa: F401
from agents.memory import ConversationManager
from rag.index import ensure_indexes


def turn(cm, sid, text):
    r = cm.ask(sid, text, review=False)
    print(f"\n[{sid}] USER : {text}")
    if r["resolved_record_id_from_memory"]:
        print(f"   memory   : resolved {r['resolved_record_id_from_memory']} from this session's history")
    print(f"   tools    : {[c['tool'] + str(c['args']) for c in r['tool_calls']] or 'none'}")
    print(f"   AGENT    : {r['response'].answer}")


def main():
    ensure_indexes()
    cm = ConversationManager()

    print("=" * 90 + "\nTRANSCRIPT A — one session, state carried across turns\n" + "=" * 90)
    turn(cm, "session-A", "Can you check application CRED-LA-0013 for me?")
    turn(cm, "session-A", "What is its loan amount?")
    turn(cm, "session-A", "Can I prepay it, and is there a penalty?")
    hist = cm.get_history("session-A").messages
    print(f"\nsession-A history: {len(hist)} messages -> {[type(m).__name__ for m in hist]}")

    print("\n" + "=" * 90 + "\nTRANSCRIPT B — a fresh session, state correctly absent\n" + "=" * 90)
    turn(cm, "session-B", "What is its loan amount?")
    print(f"\nsession-B history: {len(cm.get_history('session-B').messages)} messages "
          "(only its own exchange; nothing from session-A)")


if __name__ == "__main__":
    main()
