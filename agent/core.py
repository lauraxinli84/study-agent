"""
Agent loop using OpenAI Chat Completions + tool calling.

The LLM is given four tools and decides autonomously:
  - whether to call any tool (could answer from its own knowledge)
  - which tool(s) to call
  - what arguments to pass
  - whether to chain tool calls (e.g. search materials -> calculator)
  - when it has enough info to write a final answer
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from openai import OpenAI

from agent.vectorstore import VectorStore
from observability.tracer import Tracer
from tools import calculator, generate_quiz, search_materials, web_search

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_STEPS = int(os.getenv("MAX_AGENT_STEPS", "6"))

TOOL_SPECS = [
    search_materials.TOOL_SPEC,
    web_search.TOOL_SPEC,
    calculator.TOOL_SPEC,
    generate_quiz.TOOL_SPEC,
]

SYSTEM_PROMPT = """\
You are a study assistant for a student. You help them understand course materials, \
practice concepts, and answer questions.

You have four tools:
  - search_course_materials: retrieve passages from the student's uploaded documents.
  - web_search: look up information on the public web.
  - calculator: evaluate numeric expressions.
  - generate_quiz: produce a short practice quiz grounded in uploaded material.

Decision guidelines:
  - If the question is clearly about the student's uploaded materials (concepts, \
definitions, content from their course), call search_course_materials FIRST. Cite \
the returned sources in your answer.
  - If the question needs current or general information not plausibly in the \
uploaded materials, use web_search.
  - If the question involves a specific numeric computation, use calculator for \
the arithmetic rather than doing it in your head.
  - If the user asks to be quizzed, tested, or to practice, use generate_quiz.
  - If the question is conversational or about studying strategy generally and \
needs no external facts, answer directly without any tool.
  - You may chain tools (e.g. retrieve passages, then compute). Stop and answer \
when you have enough information.

When citing uploaded sources, use the format [doc_name chunk N] inline.
Be concise and accurate. If you're unsure, say so."""


@dataclass
class AgentResult:
    answer: str
    run_id: str
    steps: list[dict] = field(default_factory=list)


def _dispatch_tool(
    name: str, args: dict, store: VectorStore, client: OpenAI
) -> dict:
    if name == "search_course_materials":
        return search_materials.run(
            store, client, args["query"], args.get("k", 4)
        )
    if name == "web_search":
        return web_search.run(args["query"], args.get("max_results", 4))
    if name == "calculator":
        return calculator.run(args["expression"])
    if name == "generate_quiz":
        return generate_quiz.run(
            store, client, args["topic"], args.get("num_questions", 3)
        )
    return {"ok": False, "error": f"Unknown tool: {name}"}


def run_agent(
    user_message: str,
    store: VectorStore,
    client: OpenAI,
    history: list[dict] | None = None,
) -> AgentResult:
    """Execute one user turn. Returns the final answer plus a step trace."""
    tracer = Tracer(user_input=user_message)
    steps: list[dict] = []
    if store.is_empty():
        inventory = "No documents are currently indexed."
    else:
        inventory = (
            "Documents currently indexed: "
            + ", ".join(store.doc_names())
            + ". If the question is plausibly about any of them, try "
            "search_course_materials FIRST. For general conversation, "
            "stable world facts (e.g. capital cities), or topics clearly "
            "outside the uploaded material, answer directly without any tool."
        )
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + inventory}
    ]
    if history:
        # Pass prior turns (user + assistant text only; we don't replay old tool calls)
        for h in history:
            if h["role"] in ("user", "assistant") and isinstance(h.get("content"), str):
                messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_message})

    final_answer: str | None = None
    try:
        for step in range(MAX_STEPS):
            with tracer.event(
                kind="llm_call",
                name=MODEL,
                inputs={"messages": messages, "tools": [t["function"]["name"] for t in TOOL_SPECS]},
            ) as ev:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOL_SPECS,
                    tool_choice="auto",
                    temperature=0.2,
                )
                msg = resp.choices[0].message
                ev["output"] = {
                    "content": msg.content,
                    "tool_calls": [
                        {"name": tc.function.name, "args": tc.function.arguments}
                        for tc in (msg.tool_calls or [])
                    ],
                    "finish_reason": resp.choices[0].finish_reason,
                }

            # If no tool call, we have a final answer.
            if not msg.tool_calls:
                final_answer = msg.content or ""
                steps.append({"step": step, "type": "final_answer"})
                break

            # Record the assistant's tool-call message in history
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            # Execute each requested tool and append the result
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                with tracer.event(
                    kind="tool_call", name=name, inputs=args
                ) as ev:
                    result = _dispatch_tool(name, args, store, client)
                    ev["output"] = result
                steps.append(
                    {"step": step, "type": "tool_call", "name": name, "args": args}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result)[:6000],  # cap tool output size
                    }
                )
        else:
            # Loop exhausted without a final answer
            final_answer = (
                "[Agent stopped: exceeded max steps without producing a final answer.]"
            )
            steps.append({"step": MAX_STEPS, "type": "max_steps_exceeded"})

        tracer.finalize(final_answer)
        return AgentResult(
            answer=final_answer or "", run_id=tracer.run_id, steps=steps
        )

    except Exception as e:
        tracer.finalize(None, error=f"{type(e).__name__}: {e}")
        raise
