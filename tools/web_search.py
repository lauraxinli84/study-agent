"""Tool: DuckDuckGo web search for current / outside-of-materials info."""
from __future__ import annotations

from duckduckgo_search import DDGS


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the public web. Use this when a question requires information "
            "not in the uploaded course materials -- current events, recent research, "
            "definitions of terms not covered in the materials, etc. Do NOT use this "
            "if the question is clearly answerable from uploaded materials."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "max_results": {
                    "type": "integer",
                    "description": "How many results to return (1-5).",
                    "default": 4,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def run(query: str, max_results: int = 4) -> dict:
    max_results = max(1, min(int(max_results), 5))
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "results": []}

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", "")[:400],
        }
        for r in raw
    ]
    return {"ok": True, "num_results": len(results), "results": results}
