"""Regenerate every transcript in transcripts/ under MOCK_LLM, offline.
Run from the project root:  python -m scripts.run_all
Each task's output is written to its own file; a failing task is reported and the rest
still run.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "transcripts"

STEPS = [
    ("task01_dataset", ["dataset.py"]),
    ("task04a_calibrate_threshold", ["-m", "scripts.calibrate_threshold"]),
    ("task03_chunk_index", ["-m", "scripts.task3_index"]),
    ("task04b_grounded_generation", ["-m", "scripts.task4_grounded"]),
    ("task05_chunking_eval", ["-m", "scripts.task5_eval_chunking"]),
    ("task06_lookup_escalation", ["-m", "scripts.task6_lookup"]),
    ("task07_crew", ["-m", "scripts.task7_crew"]),
    ("task08_memory", ["-m", "scripts.task8_memory"]),
    ("task09_schema", ["-m", "scripts.task9_schema"]),
    ("task10_guardrails", ["-m", "scripts.task10_guardrails"]),
    ("task11_api", ["-m", "scripts.task11_api"]),
    ("task12_logging", ["-m", "scripts.task12_logging"]),
    ("task13_evaluation", ["-m", "eval.evaluate"]),
    ("task14_review", ["-m", "scripts.task14_review"]),
    ("task15_governance", ["-m", "scripts.task15_governance"]),
    ("task16_cache", ["-m", "scripts.task16_cache"]),
]


def main():
    OUT.mkdir(exist_ok=True)
    env = dict(os.environ, MOCK_LLM="1", CREWAI_DISABLE_TELEMETRY="true", OTEL_SDK_DISABLED="true",
               HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8")
    failed = []
    for name, args in STEPS:
        t0 = time.perf_counter()
        print(f"running {name:<32}", end="", flush=True)
        p = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        header = (f"# {name}\n# command: python {' '.join(args)}\n"
                  f"# env: MOCK_LLM=1 CREWAI_DISABLE_TELEMETRY=true OTEL_SDK_DISABLED=true HF_HUB_OFFLINE=1\n\n")
        (OUT / f"{name}.txt").write_text(header + p.stdout + ("\n--- stderr ---\n" + p.stderr if p.stderr else ""),
                                         encoding="utf-8")
        status = "ok" if p.returncode == 0 else f"FAILED (exit {p.returncode})"
        if p.returncode != 0:
            failed.append(name)
        print(f"{status:<18} {time.perf_counter() - t0:6.1f}s")
    print(f"\nTranscripts written to {OUT}")
    if failed:
        print(f"Failed: {failed} — open the matching transcript file to see the error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
