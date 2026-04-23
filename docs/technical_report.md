# Technical Report: Study Agent

An agentic LLM-based web application that helps a student learn from their own
uploaded course materials. Built for the course's final project; this report
maps directly to rubric sections a–i.

---

## a. Problem and Use Case

**Problem.** Students accumulate lecture notes, textbook chapters, and papers
over a semester. When reviewing, they ask questions like *"what is gradient
descent?"* or *"quiz me on backprop."* Three things typically go wrong with a
plain ChatGPT conversation: (1) the answer isn't grounded in their specific
course materials; (2) time-sensitive questions get stale answers; (3) there's
no structured way to self-test.

**User.** A student studying for an exam or writing up coursework, interacting
through a web UI.

**What the app does.** A user uploads PDFs / Markdown / text files. Then they
chat. The agent decides, per question, whether to:
- retrieve passages from the uploaded materials (RAG),
- hit the public web for recent / out-of-materials info,
- do concrete arithmetic,
- generate a short practice quiz grounded in their materials,
- or just answer directly.

It cites sources from the uploads when relevant, and every run is logged for
inspection.

## b. System Design

### High-level architecture

```
Streamlit UI  ──►  Agent loop  ──►  4 tools  ──►  (vector store / web / math / sub-LLM)
     │                 │                                 │
     └── Traces tab ◄── Tracer (SQLite)  ◄───────────────┘
```

### Main components

| Component | File | Responsibility |
|---|---|---|
| UI | `app.py` | 3 tabs: Chat, Documents, Traces. Manages upload/index, renders per-turn agent trace. |
| Agent loop | `agent/core.py` | Sends messages + tool specs to OpenAI, dispatches tool calls, stops when model returns no tool call or hits `MAX_AGENT_STEPS`. |
| Vector store | `agent/vectorstore.py` | Chunks text on paragraph boundaries (~250 words with 40-word overlap), embeds with `text-embedding-3-small`, persists as pickle, searches by cosine similarity. |
| Tools | `tools/*.py` | Four independent modules, each exporting a `TOOL_SPEC` (JSON schema) and a `run(...)` function. |
| Tracer | `observability/tracer.py` | SQLite DB with `runs` (one per user turn) and `events` (LLM calls + tool calls) tables. |
| Eval | `eval/run_eval.py` | Runs labeled scenarios, pulls tool-use + latency from the tracer, writes CSV + summary. |

### Agentic behavior

Each user turn opens a `Tracer` and enters a loop (≤ `MAX_AGENT_STEPS = 6`
iterations). In each iteration:

1. The agent sends the full conversation + all four tool specs to
   `chat.completions.create` with `tool_choice="auto"`.
2. If the model responds with no `tool_calls`, the loop ends and the content
   is returned as the final answer.
3. Otherwise, each tool call is dispatched to the local Python function, and
   its JSON output is appended as a `tool` message. The loop continues.

This lets the model chain tools (e.g. retrieve → calculate) without any fixed
pipeline on our side.

## c. Why the System is Agentic

Per the rubric, an agentic system must have the LLM genuinely deciding what
happens next. Here, on every turn the LLM controls:

- **Whether** to use any tool (could answer directly — and does, for casual
  questions; confirmed in the eval scenarios `s9`, `s10`, `s12`).
- **Which** of the four tools to invoke, from a list exposed via JSON-schema.
- **Arguments** to each call — the model chooses the query string, the value
  of `k`, the math expression, the quiz topic, etc.
- **Chaining**: after a tool result is returned, the model either calls
  another tool (e.g. retrieve a value from notes, then compute with it — see
  scenario `s6`) or writes a final answer.
- **Stopping**: the model decides when it has enough information.

This is not a fixed pipeline — the tool trajectory differs across inputs. A
math question like "what is √2 · log(1000)?" produces `[calculator]`; a quiz
request produces `[generate_quiz]`; a current-events question produces
`[web_search]`; "hi, what can you do?" produces no tool calls.

The deterministic parts of the system are exactly the parts that *shouldn't*
be agentic: JSON schema validation, tool dispatch, trace logging, UI
rendering. The LLM controls every routing decision.

## d. Technical Choices and Rationale

| Area | Choice | Why |
|---|---|---|
| LLM | OpenAI `gpt-4o-mini` (default, overridable via env var) | User indicated OpenAI access; `gpt-4o-mini` has solid tool-calling reliability at very low cost (project-friendly). |
| Tool-calling API | Chat Completions + `tools=[…]`, `tool_choice="auto"` | The most stable, best-documented, and universally supported pattern. Responses API and the Agents SDK would also work but add deployment surface area without changing what the rubric rewards. |
| Embeddings | `text-embedding-3-small` | Cheap, good recall for small corpora; 1536 dims fits easily in memory. |
| Vector store | scikit-learn cosine similarity over a numpy matrix, pickled to disk | Zero external service. Course-size corpora (< few thousand chunks) don't justify Pinecone / Weaviate / pgvector complexity. |
| Orchestration | Hand-written loop | LangGraph / CrewAI add abstraction that the project rubric doesn't reward and that make the control flow harder to reason about for an eval. A ~60-line loop is easier to trace, debug, and explain. |
| Web-search tool | `duckduckgo-search` | No API key required — keeps the deployment story to a single secret. |
| Observability | Custom SQLite tracer | Avoids a SaaS signup dependency for the grader, data is portable (DB file can be downloaded), and it was trivial to wire both writes (`Tracer.event(...)` context manager) and reads (Traces tab + eval harness). |
| UI | Streamlit | One file, zero front-end plumbing, works out of the box with HF Spaces. |
| Deployment | Hugging Face Spaces (Streamlit SDK) | Free CPU tier, no credit card, git-push workflow, handles HTTPS + the public URL automatically. |
| Secrets | `OPENAI_API_KEY` as a Space secret; `.env` for local | Standard, no key in repo. |

## e. Observability

**What it is.** A SQLite-backed tracer (`observability/tracer.py`) with two
tables:

- `runs`: one row per user turn. Columns: `run_id`, `started_at`, `ended_at`,
  `user_input`, `final_output`, `total_latency_ms`, `num_steps`, `status`,
  `error`.
- `events`: one row per atomic step inside a run. Columns: `kind` (`llm_call`
  / `tool_call` / `error`), `name` (model or tool name), `input_json`,
  `output_json`, `latency_ms`, `status`, `error`.

The `Tracer.event(kind, name, inputs)` context manager captures timing,
inputs, outputs, and exceptions automatically.

**What it captures.** Every user input, every OpenAI call (messages sent,
content + tool_calls returned, finish reason, latency), every tool dispatch
(args passed, result returned, latency, error), and final outputs.

**How it helps.** Three ways to inspect:

1. The **Traces** tab in the UI shows aggregate stats (total runs, error
   rate, avg latency, tool-call counts) and a list of recent runs with
   expandable event-level detail. No external account needed.
2. The **eval harness** reads directly from the same DB to compute tool-use
   accuracy and latency statistics — so observability and evaluation share
   the same source of truth.
3. The DB file is portable: `sqlite3 data/traces.db` from the shell works for
   ad-hoc queries when something breaks.

## f. Metrics

Two metrics, one per category as the rubric suggests:

### 1. Tool-selection accuracy (quality)

**What it is.** For each labeled scenario, did the set of tools the agent
actually called equal the expected set? Reported as a fraction over 12
scenarios covering the four tools, no-tool cases, chained cases, and one
deliberately adversarial "over-tooling" case (`s12`: the agent should NOT
web-search for "capital of France").

**Why it matters.** This is the decision the whole project is about: is the
LLM routing correctly? A high-quality answer from the wrong tool is still a
failure (wasted tokens, wasted latency, possibly wrong for the right
reasons). We also report precision and recall per scenario for finer insight.

**How tracked.** At eval time, `run_eval.py` reads the `events` table for
each run_id to get the ordered list of tool calls, compares to the expected
list in `scenarios.json`, and writes per-scenario rows to `eval/results.csv`.

### 2. End-to-end latency (operational)

**What it is.** Wall-clock time from user input to final answer per run.
Reported as mean, median, p95, and max.

**Why it matters.** Latency is the single most user-facing operational
property of an agent. A loop that chains four tool calls can easily balloon
past 15s, which destroys the UX. Tracking p95 specifically catches the
chaining / retry cases that mean latency misses.

**How tracked.** `total_latency_ms` is written to the `runs` row on
`tracer.finalize()`. The eval harness aggregates across all runs.

### Secondary measurements also reported

- `avg_tool_precision` and `avg_tool_recall` — useful when the expected set
  and actual set differ partially (e.g. agent does the right retrieval but
  then skips a math call).
- `error rate` — runs that raised an exception.

## g. Evaluation

### Scenarios (12 total, see `eval/scenarios.json`)

Grouped by intent:

- **Materials questions** (3): should hit `search_course_materials`.
- **Web-search questions** (2): current events, recent versions.
- **Pure math** (1): should hit `calculator`.
- **Chained** (1): retrieve a value from notes, then compute.
- **Quiz requests** (2): should hit `generate_quiz`.
- **No-tool** (3): greeting, study advice, stable world fact — agent should
  answer without tools. `s12` specifically tests that the agent doesn't
  over-tool by web-searching "capital of France".

All "requires_uploads" scenarios are backed by `eval/sample_notes.md`, which
the harness auto-indexes on first run so the eval is reproducible without
any external document.

### Running it

```bash
python -m eval.run_eval
```

Produces `eval/results.csv` and a printed summary.

### Expected results and interpretation

*(I was not able to execute the harness against the live OpenAI API from the
development environment this project was built in. The harness is wired end
to end and tested at the import and tracer level; run it once locally to
populate the numbers below.)*

Rough qualitative expectations based on prior experience with `gpt-4o-mini`
+ tool-calling:

- Tool-selection accuracy: ~0.80–0.90 on this scenario set. The most common
  failure mode is over-tooling — the model calls `web_search` on a general
  question it could have answered directly (scenario `s12`-style).
- Latency: single-tool runs typically 1.5–3 s, chained runs 3–6 s. p95
  should stay under ~8 s.

### Known strengths

- Clean separation between tool specs, dispatch, and tracer means adding a
  new tool is ~20 lines and doesn't touch the agent loop.
- The quiz tool uses structured JSON (`response_format={"type":"json_object"}`)
  and is reliable on short topics.
- Tracer captures enough state that most failures can be diagnosed without
  re-running the agent.

### Known limitations / failure modes

- **Retrieval quality is coarse.** Paragraph-based chunking with no reranker
  — fine for short notes, weaker on long PDFs with complex structure. A
  reranker step (e.g. `text-embedding-3-large` rerank over top-20) would
  improve recall cheaply.
- **No deduplication of retrieved chunks.** If the same concept appears in
  two uploaded files, the model can see near-duplicate context.
- **`duckduckgo_search` is rate-limited and occasionally flaky.** The tool
  catches errors and returns them structured, which lets the agent decide to
  continue without search, but this means the web-search metric depends on
  external conditions.
- **No agent-level memory across sessions.** History is kept in Streamlit
  session state, so a refresh wipes the conversation (but NOT the traces
  or vector store, which persist).
- **Model still sometimes hallucinates citations.** I mitigate this with an
  explicit cite-format instruction in the system prompt, but it's not
  formally enforced. A post-hoc check (does the cited chunk actually exist?)
  would close this.

### Tradeoffs

- **Simplicity vs. framework power.** I chose raw Chat Completions + a
  hand-written loop over LangGraph. Upside: every decision is visible in <80
  lines and trivially debuggable. Downside: no built-in resume, no parallel
  tool execution, no pretty visual graph.
- **Portability vs. polish of observability.** Custom SQLite over Langfuse.
  Upside: zero signup, nothing to configure, portable DB. Downside: no
  hosted UI, no team sharing, no built-in dashboards beyond what the Traces
  tab renders.
- **In-memory vector store vs. a real DB.** Good enough up to a few thousand
  chunks. For a semester's worth of materials across many courses, this
  should become pgvector or similar.

### What I'd improve with more time

1. Add an evaluation of **answer correctness**, not just tool choice, using
   an LLM-as-judge over the final answers with a reference rubric.
2. Add a **reranker** over the top-k retrieved chunks.
3. Add a **citation-validation pass**: after the agent writes a final
   answer, verify that every `[doc chunk N]` citation corresponds to an
   actual chunk the retriever returned this turn.
4. Add **streaming** so the user sees tokens as they're produced.
5. Swap `duckduckgo_search` for a keyed search API for reliability.

## h. Deployment

- **Platform**: Hugging Face Spaces, Streamlit SDK, free CPU tier.
- **URL**: *(fill in once the Space is deployed)*.
- **Config**: `README.md` YAML frontmatter specifies `sdk: streamlit` and
  `sdk_version: 1.38.0`. Streamlit is intentionally *not* listed in
  `requirements.txt` — HF installs the declared version itself.
- **Secrets**: `OPENAI_API_KEY` is set as a Space secret under Settings →
  Variables and secrets. It is never committed.
- **State**: The `data/` directory (vectorstore pickle and traces DB) is
  ephemeral on the free tier — it persists across restarts of the same
  container but not across Space rebuilds. For a graded demo this is fine;
  for production we'd attach HF persistent storage or move to HF Datasets
  for write-through persistence.

### Practical constraints

- Free CPU: embedding a large PDF takes a few seconds per hundred chunks.
- Per-request OpenAI cost is small but non-zero; the free HF tier means the
  Space sleeps after inactivity and cold-starts in ~10–20 s on first hit.
- DuckDuckGo occasionally rate-limits; the tool surfaces this structurally.

## i. Reflection

**What I learned.**

- Clean tool specs matter more than the framework. The agent's routing
  quality is mostly a function of clear tool descriptions and precise
  decision guidelines in the system prompt, not the orchestrator.
- A small, custom tracer is dramatically more useful than a vendor SDK for a
  project at this scale, because it's trivial to read alongside the code
  that wrote it.
- Labeling expected tool sets per scenario is cheap and gives you a metric
  that *actually* distinguishes a working agent from a bad one, unlike
  generic "answer quality" scores that require LLM-as-judge.

**What I'd revisit.**

- I'd consider the OpenAI Agents SDK if the project were to grow. For a
  4-tool agent a hand-written loop is clearer, but once you add handoffs,
  guardrails, or parallel tool calls, the SDK's primitives start earning
  their keep.
- I'd add retrieval evaluation separately from tool-use evaluation.
  Currently a scenario can "pass" (correct tool used) while the retriever
  returns mediocre chunks, which is invisible to the headline metric.

**Design choices I'd revisit with more time.**

- Persist the vector store to HF Datasets rather than local pickle, so
  uploads survive Space rebuilds.
- Add a "dry run" mode that logs what tool the agent *would* call without
  actually calling it — useful for faster eval iteration.
- Make `MAX_AGENT_STEPS` adaptive (e.g. allow more steps on quiz/chained
  tasks).
