"""
Lightweight SQLite-backed tracer.

Captures one row per agent "run" (a single user message -> final answer cycle)
and one row per "event" inside that run (LLM call, tool call, error).

Chosen over Langfuse/LangSmith so the project has zero external SaaS dependency
and no account is needed to inspect logs -- the grader can just download traces.db.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any

DB_PATH = os.getenv("LOG_DB_PATH", "data/traces.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    started_at    REAL NOT NULL,
    ended_at      REAL,
    user_input    TEXT NOT NULL,
    final_output  TEXT,
    total_latency_ms INTEGER,
    num_steps     INTEGER,
    status        TEXT,            -- 'ok' or 'error'
    error         TEXT
);

CREATE TABLE IF NOT EXISTS events (
    event_id      TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    parent_id     TEXT,
    ts            REAL NOT NULL,
    kind          TEXT NOT NULL,   -- 'llm_call' | 'tool_call' | 'error'
    name          TEXT,            -- model name or tool name
    input_json    TEXT,
    output_json   TEXT,
    latency_ms    INTEGER,
    status        TEXT,            -- 'ok' | 'error'
    error         TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
"""


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        return json.dumps({"_repr": repr(obj)})


class Tracer:
    """One Tracer instance per agent run. Use as a context manager."""

    def __init__(self, user_input: str):
        self.run_id = str(uuid.uuid4())
        self.user_input = user_input
        self._started = time.time()
        self._ended: float | None = None
        self._num_steps = 0
        self._final_output: str | None = None
        self._status = "ok"
        self._error: str | None = None
        init_db()
        with _connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, started_at, user_input, status) "
                "VALUES (?, ?, ?, 'running')",
                (self.run_id, self._started, user_input),
            )

    def finalize(self, output: str | None, error: str | None = None) -> None:
        self._ended = time.time()
        self._final_output = output
        if error:
            self._status = "error"
            self._error = error
        latency_ms = int((self._ended - self._started) * 1000)
        with _connect() as conn:
            conn.execute(
                "UPDATE runs SET ended_at=?, final_output=?, total_latency_ms=?, "
                "num_steps=?, status=?, error=? WHERE run_id=?",
                (
                    self._ended,
                    output,
                    latency_ms,
                    self._num_steps,
                    self._status,
                    self._error,
                    self.run_id,
                ),
            )

    @contextmanager
    def event(self, kind: str, name: str, inputs: Any = None):
        """Context manager that records one event. Yields a dict the caller
        mutates to set `output` before exiting."""
        self._num_steps += 1
        event_id = str(uuid.uuid4())
        start = time.time()
        record: dict[str, Any] = {"output": None}
        status = "ok"
        err: str | None = None
        try:
            yield record
        except Exception as e:
            status = "error"
            err = f"{type(e).__name__}: {e}"
            raise
        finally:
            latency_ms = int((time.time() - start) * 1000)
            with _connect() as conn:
                conn.execute(
                    "INSERT INTO events (event_id, run_id, ts, kind, name, "
                    "input_json, output_json, latency_ms, status, error) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event_id,
                        self.run_id,
                        start,
                        kind,
                        name,
                        _safe_json(inputs),
                        _safe_json(record.get("output")),
                        latency_ms,
                        status,
                        err,
                    ),
                )


# Read-side helpers for the Streamlit "Traces" tab -------------------------

def recent_runs(limit: int = 50) -> list[dict]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def run_events(run_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE run_id=? ORDER BY ts ASC", (run_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def aggregate_stats() -> dict:
    init_db()
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"]
        errors = conn.execute(
            "SELECT COUNT(*) AS n FROM runs WHERE status='error'"
        ).fetchone()["n"]
        avg_latency = conn.execute(
            "SELECT AVG(total_latency_ms) AS v FROM runs WHERE status='ok'"
        ).fetchone()["v"]
        tool_counts = conn.execute(
            "SELECT name, COUNT(*) AS n FROM events "
            "WHERE kind='tool_call' GROUP BY name ORDER BY n DESC"
        ).fetchall()
    return {
        "total_runs": total,
        "error_runs": errors,
        "avg_latency_ms": int(avg_latency) if avg_latency else 0,
        "tool_counts": [dict(r) for r in tool_counts],
    }
