---
title: Study Agent
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Agentic LLM study assistant grounded in your uploads.
---

# 📚 Study Agent

An agentic LLM-based study assistant. Upload your course materials, then ask
questions — the agent decides whether to search your documents, search the web,
do math, or generate a practice quiz.

Highlights:

- a genuine agentic tool-calling loop (LLM chooses among 4 tools, with chaining)
- lightweight custom observability (SQLite trace log)
- a labeled evaluation harness with two headline metrics
- public deployment on Hugging Face Spaces (Docker SDK)

## Live demo

https://huggingface.co/spaces/stevenlx96/study-agent

### Quick test path (≈ 2 minutes)

1. Open the **Documents** tab and upload [`eval/sample_notes.md`](eval/sample_notes.md)
   from this repo (a tiny ML-notes file).
2. Switch to **Chat** and try one prompt per tool to see the agent route correctly:
   - *"According to my notes, what is gradient descent?"* → calls
     `search_course_materials`, returns a citation `[sample_notes.md (chunk 0)]`.
   - *"What is sqrt(2) times log(1000) to 4 decimal places?"* → calls `calculator`.
   - *"Quiz me on backpropagation."* → calls `generate_quiz`.
   - *"Who won the most recent Nobel Prize in Physics?"* → calls `web_search`.
   - *"Hi, what can you help me with?"* → no tool, direct answer.
3. Open the **Traces** tab to see every run logged with per-event timing,
   inputs, and outputs.

> The free Hugging Face tier sleeps the Space after inactivity, so the
> first request may take 10–20 s to cold-start. Uploads are also wiped on
> Space rebuilds — re-upload `sample_notes.md` if the sidebar shows no
> indexed documents.

## What makes it "agentic"

On every user turn, the LLM makes real decisions:

| Decision | Options |
|---|---|
| Should I use any tool at all? | Yes / No (answer from own knowledge) |
| Which tool? | `search_course_materials`, `web_search`, `calculator`, `generate_quiz` |
| What arguments? | free-form (query, expression, topic, k, etc.) |
| Should I chain? | After a tool result, either call another tool or write the final answer |

Different inputs produce different trajectories — e.g. a math question goes
straight to `calculator`, a content question first retrieves from uploaded
materials, a "quiz me on X" call goes to `generate_quiz`, and a casual
greeting uses no tools at all. This is verified by the eval harness.

## Architecture

```
               ┌──────────────────────┐
  user ───▶   │   Streamlit UI        │
               │ (chat / docs / traces)│
               └──────────┬────────────┘
                          │
                          ▼
               ┌──────────────────────┐
               │   Agent loop          │     tools
               │  (agent/core.py)      │────┬──▶  search_course_materials  ──▶  VectorStore (pickle, OpenAI embeddings)
               │  OpenAI Chat          │    ├──▶  web_search                ──▶  DuckDuckGo
               │  Completions +        │    ├──▶  calculator                ──▶  AST-safe evaluator
               │  tool calling         │    └──▶  generate_quiz             ──▶  mini-LLM call + retrieval
               └──────────┬────────────┘
                          │
                          ▼
               ┌──────────────────────┐
               │   Tracer (SQLite)     │
               │  observability/       │
               └──────────────────────┘
```

## Project layout

```
├── app.py                      # Streamlit UI (Chat / Documents / Traces tabs)
├── agent/
│   ├── core.py                 # tool-calling loop
│   └── vectorstore.py          # in-memory vector store + chunking + embeddings
├── tools/
│   ├── search_materials.py     # RAG over uploaded docs
│   ├── web_search.py           # DuckDuckGo
│   ├── calculator.py           # AST-sandboxed math
│   └── generate_quiz.py        # sub-agent that produces structured JSON quizzes
├── observability/
│   └── tracer.py               # SQLite run + event logger
├── eval/
│   ├── scenarios.json          # labeled test cases (expected tools)
│   ├── sample_notes.md         # tiny corpus used for eval
│   ├── run_eval.py             # harness; writes results.csv
│   └── results.csv             # eval output (generated)
├── data/
│   ├── uploads/                # user-uploaded files
│   ├── vectorstore/store.pkl   # pickled embeddings
│   └── traces.db               # SQLite trace log
├── Dockerfile                  # HF Docker Space entrypoint
├── requirements.txt
├── .env.example
└── README.md
```

## Setup (local)

```bash
git clone <this repo>
cd study-agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

streamlit run app.py
```

Then open http://localhost:8501.

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | — | Your OpenAI API key. |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | Chat model used by agent + quiz tool. |
| `OPENAI_EMBED_MODEL` | no | `text-embedding-3-small` | Embedding model. |
| `MAX_AGENT_STEPS` | no | `6` | Hard cap on tool-call iterations per turn. |
| `LOG_DB_PATH` | no | `data/traces.db` | SQLite file location. |

Do not commit secrets. `.env` is gitignored.

## Deploy to Hugging Face Spaces

This project ships with a Dockerfile so it runs as a **Docker Space** on HF
(the UI dropped Streamlit from the SDK picker; Docker gives us full control
and honours the `app_port` in the README frontmatter).

1. Create a free account at https://huggingface.co.
2. Click **New Space** → SDK = **Docker** → Template = **Blank** → choose a
   name → Create.
3. In the Space's **Settings → Variables and secrets**, add a **secret**
   named `OPENAI_API_KEY` with your key as the value.
4. Push this repo to the Space's Git remote (add it alongside your GitHub
   remote):
   ```bash
   git remote add hf https://huggingface.co/spaces/<username>/<space-name>
   git push hf main
   ```
5. HF builds the image from the `Dockerfile` (~3–4 min on first push).
   When the build succeeds the app goes live at
   `https://huggingface.co/spaces/<username>/<space-name>`.

## Run the evaluation

```bash
python -m eval.run_eval
```

Output:
- printed summary with tool-selection accuracy + latency stats
- `eval/results.csv` with per-scenario rows

Metrics reported:

1. **Tool-selection accuracy** (quality): fraction of scenarios where the set of
   tools the agent called exactly matches the expected set. Also reports
   per-scenario precision and recall.
2. **End-to-end latency** (operational): mean / median / p95 / max wall-clock
   time per run, pulled from the tracer's `runs` table.

## Inspecting traces

Every run is logged to `data/traces.db` (SQLite). Three ways to inspect:

- **In-app**: open the **Traces** tab in the Streamlit UI.
- **CLI**: `sqlite3 data/traces.db "SELECT * FROM runs ORDER BY started_at DESC LIMIT 10;"`
- **Python**: `from observability.tracer import recent_runs, run_events`

## Acknowledgments

- OpenAI Python SDK and Chat Completions tool-calling interface.
- `duckduckgo_search` library for the web-search tool.
- `pypdf` for PDF text extraction.
- Streamlit for the UI.

## License

MIT.
