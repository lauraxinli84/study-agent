"""
Streamlit UI for the study agent.

Three tabs:
  1. Chat      - talk to the agent
  2. Documents - upload / manage course materials
  3. Traces    - inspect runs and per-event logs (the observability view)
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from agent.core import run_agent
from agent.vectorstore import add_document, load_store, remove_document, save_store
from observability.tracer import aggregate_stats, recent_runs, run_events

load_dotenv()

st.set_page_config(page_title="Study Agent", page_icon="📚", layout="wide")


# ---- API key handling -------------------------------------------------------

def get_api_key() -> str | None:
    # 1) env var (local .env or HF Space Secret)
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    # 2) Streamlit secrets (for Streamlit Cloud)
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return None


api_key = get_api_key()
if not api_key:
    st.error(
        "No OPENAI_API_KEY found. Set it as an environment variable or in "
        "Streamlit secrets."
    )
    st.stop()

client = OpenAI(api_key=api_key)

# ---- Shared state -----------------------------------------------------------

if "store" not in st.session_state:
    st.session_state.store = load_store()
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": ..., "content": ..., "steps": ...}

store = st.session_state.store

# ---- Sidebar ----------------------------------------------------------------

with st.sidebar:
    st.title("📚 Study Agent")
    st.caption("An agentic LLM study assistant")
    st.markdown("---")
    st.subheader("Indexed documents")
    if store.is_empty():
        st.info("No documents yet. Upload some in the **Documents** tab.")
    else:
        for name in store.doc_names():
            n = sum(1 for c in store.chunks if c.doc_name == name)
            st.write(f"• **{name}** — {n} chunks")
    st.markdown("---")
    st.caption(f"Model: `{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}`")
    if st.button("Reset conversation"):
        st.session_state.history = []
        st.rerun()


tab_chat, tab_docs, tab_traces = st.tabs(["💬 Chat", "📄 Documents", "🔍 Traces"])

# ---- Chat tab ---------------------------------------------------------------

with tab_chat:
    st.subheader("Ask the agent")
    st.caption(
        "The agent can search your uploaded materials, search the web, do "
        "math, or generate a quiz — it decides which based on your question."
    )

    # Render history
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("steps"):
                with st.expander(f"🔧 Agent trace ({len(turn['steps'])} steps)"):
                    for s in turn["steps"]:
                        if s["type"] == "tool_call":
                            st.code(
                                f"→ called tool `{s['name']}`\n"
                                f"  args: {s['args']}",
                                language="text",
                            )
                        elif s["type"] == "final_answer":
                            st.write("→ produced final answer")
                        elif s["type"] == "max_steps_exceeded":
                            st.warning("→ hit max-steps limit")

    prompt = st.chat_input("Ask anything about your materials…")
    if prompt:
        st.session_state.history.append(
            {"role": "user", "content": prompt, "steps": []}
        )
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result = run_agent(
                        prompt,
                        store,
                        client,
                        history=st.session_state.history[:-1],
                    )
                    st.markdown(result.answer)
                    if result.steps:
                        with st.expander(f"🔧 Agent trace ({len(result.steps)} steps)"):
                            for s in result.steps:
                                if s["type"] == "tool_call":
                                    st.code(
                                        f"→ called tool `{s['name']}`\n"
                                        f"  args: {s['args']}",
                                        language="text",
                                    )
                                elif s["type"] == "final_answer":
                                    st.write("→ produced final answer")
                    st.session_state.history.append(
                        {
                            "role": "assistant",
                            "content": result.answer,
                            "steps": result.steps,
                        }
                    )
                except Exception as e:
                    st.error(f"Agent error: {type(e).__name__}: {e}")

# ---- Documents tab ----------------------------------------------------------

with tab_docs:
    st.subheader("Upload course materials")
    st.caption("PDF, TXT, or Markdown. Each file is chunked and embedded on upload.")
    uploaded = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded:
        for f in uploaded:
            save_path = Path("data/uploads") / f.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # Skip if we've already indexed a file with this name
            if f.name in store.doc_names():
                st.info(f"{f.name} already indexed — skipping.")
                continue
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
            with st.spinner(f"Embedding {f.name}…"):
                try:
                    n = add_document(store, save_path, client)
                    st.success(f"Indexed {f.name} ({n} chunks).")
                except Exception as e:
                    st.error(f"Failed to index {f.name}: {e}")

    st.markdown("---")
    st.subheader("Currently indexed")
    if store.is_empty():
        st.info("Nothing indexed yet.")
    else:
        for name in store.doc_names():
            cols = st.columns([4, 1])
            cols[0].write(
                f"**{name}** — "
                f"{sum(1 for c in store.chunks if c.doc_name == name)} chunks"
            )
            if cols[1].button("Remove", key=f"rm_{name}"):
                remove_document(store, name)
                st.rerun()

# ---- Traces tab -------------------------------------------------------------

with tab_traces:
    st.subheader("Observability")
    st.caption(
        "Every agent run is logged to SQLite. This tab reads from that DB — "
        "the same data a grader or developer would inspect in production."
    )

    stats = aggregate_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total runs", stats["total_runs"])
    c2.metric(
        "Error rate",
        f"{(stats['error_runs'] / stats['total_runs'] * 100):.1f}%"
        if stats["total_runs"]
        else "—",
    )
    c3.metric("Avg latency (ok runs)", f"{stats['avg_latency_ms']} ms")

    if stats["tool_counts"]:
        st.write("**Tool call counts**")
        for tc in stats["tool_counts"]:
            st.write(f"• `{tc['name']}` — {tc['n']}")

    st.markdown("---")
    st.write("**Recent runs**")
    runs = recent_runs(limit=25)
    if not runs:
        st.info("No runs logged yet.")
    for r in runs:
        label = (
            f"{r['status']} · {r['total_latency_ms'] or '—'}ms · "
            f"{r['num_steps'] or 0} steps · {r['user_input'][:80]}"
        )
        with st.expander(label):
            st.caption(f"run_id: `{r['run_id']}`")
            st.write("**User input:**", r["user_input"])
            if r.get("final_output"):
                st.write("**Final output:**")
                st.markdown(r["final_output"])
            if r.get("error"):
                st.error(r["error"])
            st.write("**Events:**")
            for e in run_events(r["run_id"]):
                st.code(
                    f"[{e['kind']}] {e['name']}  ({e['latency_ms']}ms, {e['status']})\n"
                    f"input:  {e['input_json'][:400] if e['input_json'] else ''}\n"
                    f"output: {e['output_json'][:400] if e['output_json'] else ''}",
                    language="text",
                )
