"""Tool: retrieve passages from uploaded course materials."""
from __future__ import annotations

from openai import OpenAI

from agent.vectorstore import VectorStore, search


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "search_course_materials",
        "description": (
            "Search the user's uploaded course materials (lecture notes, "
            "textbook chapters, papers) for passages relevant to a question. "
            "Use this when the question is about content the student has uploaded, "
            "or when grounding an answer in their specific course material would help."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query describing what to find.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of passages to retrieve (1-6).",
                    "default": 4,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def run(store: VectorStore, client: OpenAI, query: str, k: int = 4) -> dict:
    k = max(1, min(int(k), 6))
    if store.is_empty():
        return {
            "ok": False,
            "error": "No documents have been uploaded yet.",
            "results": [],
        }
    hits = search(store, query, client, k=k)
    return {
        "ok": True,
        "num_results": len(hits),
        "results": [
            {
                "source": f"{h['doc_name']} (chunk {h['chunk_idx']})",
                "score": round(h["score"], 3),
                "text": h["text"][:1200],  # cap so we don't blow the context window
            }
            for h in hits
        ],
    }
