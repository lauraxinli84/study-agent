"""
Evaluation harness.

Metrics computed:
  1. TOOL-SELECTION ACCURACY (quality)
     Did the agent use the expected tool(s) for each scenario? For scenarios
     with multiple expected tools, we require the set of tools the agent
     actually called to equal the expected set. For scenarios where no tool
     is expected, we require the agent to call nothing.

  2. END-TO-END LATENCY (operational)
     Wall-clock time per scenario, pulled from the tracer's `runs` table.
     We report mean, median, p95, and max.

Also reported (bonus):
  - per-scenario tool-precision and tool-recall
  - total error rate
  - a CSV of all per-scenario results at eval/results.csv

Usage (from project root):
    python -m eval.run_eval

Requires OPENAI_API_KEY set.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agent.core import run_agent
from agent.vectorstore import add_document, load_store
from observability.tracer import _connect, init_db

load_dotenv()

SCENARIO_PATH = Path(__file__).parent / "scenarios.json"
SAMPLE_NOTES = Path(__file__).parent / "sample_notes.md"
RESULTS_CSV = Path(__file__).parent / "results.csv"


def _ensure_sample_indexed(store, client: OpenAI) -> None:
    """Make sure sample_notes.md is in the vector store so 'requires_uploads'
    scenarios have something to retrieve from."""
    if SAMPLE_NOTES.name in store.doc_names():
        return
    print(f"[setup] indexing {SAMPLE_NOTES.name} for eval...")
    add_document(store, SAMPLE_NOTES, client)


def _tools_used_in_run(run_id: str) -> list[str]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT name FROM events WHERE run_id=? AND kind='tool_call' ORDER BY ts",
            (run_id,),
        ).fetchall()
    return [r["name"] for r in rows]


def _latency_of_run(run_id: str) -> int | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT total_latency_ms FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
    return row["total_latency_ms"] if row else None


def evaluate(limit: int | None = None) -> dict:
    init_db()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY is required to run the eval.")
    client = OpenAI(api_key=api_key)
    store = load_store()
    _ensure_sample_indexed(store, client)

    scenarios = json.loads(SCENARIO_PATH.read_text())
    if limit:
        scenarios = scenarios[:limit]

    rows = []
    latencies: list[int] = []
    errors = 0

    for s in scenarios:
        print(f"[eval] {s['id']}: {s['question']}")
        try:
            result = run_agent(s["question"], store, client)
            tools_used = _tools_used_in_run(result.run_id)
            latency_ms = _latency_of_run(result.run_id) or 0
            expected = set(s["expected_tools"])
            actual = set(tools_used)
            correct = actual == expected
            # precision / recall against the expected set
            if expected and actual:
                prec = len(expected & actual) / len(actual)
                rec = len(expected & actual) / len(expected)
            elif not expected and not actual:
                prec = rec = 1.0
            else:
                prec = 0.0 if actual and not expected else 0.0
                rec = 0.0 if expected and not actual else 0.0
            rows.append(
                {
                    "id": s["id"],
                    "question": s["question"],
                    "expected_tools": "|".join(sorted(expected)),
                    "actual_tools": "|".join(tools_used),
                    "exact_match": correct,
                    "precision": round(prec, 3),
                    "recall": round(rec, 3),
                    "latency_ms": latency_ms,
                    "error": "",
                    "run_id": result.run_id,
                }
            )
            latencies.append(latency_ms)
        except Exception as e:
            errors += 1
            rows.append(
                {
                    "id": s["id"],
                    "question": s["question"],
                    "expected_tools": "|".join(sorted(s["expected_tools"])),
                    "actual_tools": "",
                    "exact_match": False,
                    "precision": 0.0,
                    "recall": 0.0,
                    "latency_ms": 0,
                    "error": f"{type(e).__name__}: {e}",
                    "run_id": "",
                }
            )

    # Write CSV
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Aggregate
    n = len(rows)
    correct_n = sum(1 for r in rows if r["exact_match"])
    avg_prec = statistics.mean(r["precision"] for r in rows)
    avg_rec = statistics.mean(r["recall"] for r in rows)

    latency_summary: dict[str, float] = {}
    if latencies:
        latencies_sorted = sorted(latencies)
        latency_summary = {
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": latencies_sorted[max(0, int(0.95 * len(latencies_sorted)) - 1)],
            "max_ms": max(latencies),
        }

    summary = {
        "n_scenarios": n,
        "tool_selection_accuracy": round(correct_n / n, 3) if n else 0.0,
        "avg_tool_precision": round(avg_prec, 3),
        "avg_tool_recall": round(avg_rec, 3),
        "errors": errors,
        "latency": latency_summary,
    }
    print("\n=== EVAL SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nPer-scenario results written to: {RESULTS_CSV}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run first N scenarios")
    args = parser.parse_args()
    evaluate(limit=args.limit)
