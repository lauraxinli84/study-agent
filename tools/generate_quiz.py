"""Tool: generate a short quiz based on a topic + retrieved material.

This tool makes its own mini-LLM call. It's a good example of a 'sub-agent'
pattern: the main agent decides to invoke quiz generation, and the tool
internally structures a JSON output for reliable rendering.
"""
from __future__ import annotations

import json
import os

from openai import OpenAI

from agent.vectorstore import VectorStore, search


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "generate_quiz",
        "description": (
            "Generate a short multiple-choice or short-answer quiz on a topic, "
            "grounded in the user's uploaded course materials. Use when the user "
            "asks to quiz them, test them, or practice a concept."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic or concept to quiz on.",
                },
                "num_questions": {
                    "type": "integer",
                    "description": "How many questions (1-5).",
                    "default": 3,
                },
            },
            "required": ["topic"],
            "additionalProperties": False,
        },
    },
}


_QUIZ_SYSTEM = (
    "You are a study-assistant quiz generator. Given a topic and supporting "
    "passages from course materials, write exam-style questions. Return ONLY "
    "a JSON object matching this schema, no prose, no markdown fence:\n"
    '{"questions": [{"q": str, "type": "mcq" | "short", '
    '"choices": [str, ...] (only for mcq), "answer": str, "explanation": str}]}'
)


def run(
    store: VectorStore,
    client: OpenAI,
    topic: str,
    num_questions: int = 3,
) -> dict:
    num_questions = max(1, min(int(num_questions), 5))
    if store.is_empty():
        return {"ok": False, "error": "No documents uploaded to quiz on."}

    passages = search(store, topic, client, k=4)
    context = "\n\n---\n\n".join(
        f"[{p['doc_name']} chunk {p['chunk_idx']}]\n{p['text']}" for p in passages
    )
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _QUIZ_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\n"
                    f"Write {num_questions} questions. Mix MCQ and short-answer. "
                    f"Ground every question in the passages below.\n\n"
                    f"PASSAGES:\n{context}"
                ),
            },
        ],
    )
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"Quiz generator returned invalid JSON: {e}"}

    return {
        "ok": True,
        "topic": topic,
        "grounded_in": sorted({p["doc_name"] for p in passages}),
        "quiz": data.get("questions", []),
    }
