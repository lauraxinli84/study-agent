# Runbook: Your Manual Steps

Everything below is what *you* need to do, in order. Should take ~30–45 min
end to end.

---

## Step 1 — Get an OpenAI API key

1. Go to https://platform.openai.com/api-keys.
2. Create a key. Copy it (you won't see it again).
3. Add ~$5 of credits at Settings → Billing if you haven't already.
   `gpt-4o-mini` is cheap — this project will likely cost pennies.

## Step 2 — Run it locally first (to verify everything works)

```bash
cd study-agent
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate

# Install deps + streamlit (streamlit is commented out in requirements.txt
# because HF Spaces provides it; we need it locally):
pip install -r requirements.txt streamlit==1.38.0

cp .env.example .env
# Open .env and paste your key:   OPENAI_API_KEY=sk-...

streamlit run app.py
```

Open http://localhost:8501 in your browser.

**Quick smoke test:**
1. Go to the **Documents** tab, upload `eval/sample_notes.md`.
2. Go to **Chat** and ask: *"What is gradient descent?"* → should retrieve.
3. Ask: *"What is sqrt(2) times log(1000)?"* → should call calculator.
4. Ask: *"Quiz me on backpropagation."* → should call generate_quiz.
5. Go to **Traces** tab — you should see all three runs with events.

If all four work, you're good.

## Step 3 — Run the evaluation

From the project root, with your venv active:

```bash
python -m eval.run_eval
```

This runs all 12 scenarios against the live API (~30–60 seconds, ~$0.02).
It:
- auto-indexes `eval/sample_notes.md` so retrieval scenarios have material
- prints a summary with tool-selection accuracy + latency stats
- writes `eval/results.csv` with per-scenario detail

**Take a screenshot of the summary output** — you'll want it in your
submission.

Then update the "Expected results and interpretation" section of
`docs/technical_report.md` with your actual numbers.

## Step 4 — Create the GitHub repo

```bash
cd study-agent
git init
git add .
git commit -m "Initial commit: study agent"

# Create a new public repo on github.com, then:
git remote add origin https://github.com/<you>/study-agent.git
git branch -M main
git push -u origin main
```

Do **not** commit `.env` or the `data/traces.db` — both are already in
`.gitignore`. Double-check with `git status` before pushing.

You can commit `eval/results.csv` after running the eval so the grader sees
your numbers.

## Step 5 — Deploy to Hugging Face Spaces (Docker SDK)

> HF recently dropped Streamlit from the new-Space SDK picker. We use the
> **Docker** SDK instead — the repo ships with a `Dockerfile` that runs
> Streamlit on port 7860 (HF's default Docker port).

1. Sign up at https://huggingface.co (free).
2. Click your avatar → **New Space**. Name: `study-agent` (or whatever).
   SDK: **Docker**. Template: **Blank**. License: MIT. Hardware: free CPU.
   Visibility: Public.
3. In the new Space: **Settings → Variables and secrets → New secret**.
   Name: `OPENAI_API_KEY`. Value: your key. Save.
4. From *this* repo (your GitHub clone), add the Space as a second remote
   and push:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-hf-username>/study-agent
   git push hf main
   ```
   HF will prompt for your username + a write token
   (https://huggingface.co/settings/tokens → New token, type: write).
   Cache credentials or use the token as the password.
5. Watch the **Logs** tab on the Space page. First Docker build takes ~3–5
   minutes. When it flips to the **App** tab showing your UI, you're live.

Your public URL:
`https://huggingface.co/spaces/<your-hf-username>/study-agent`

## Step 6 — Fill in the report placeholders

Open `docs/technical_report.md` and fill in:
- your live HF Space URL (section h)
- your actual eval numbers (section g — "Expected results and interpretation")
- any specific failure modes you observed that aren't already listed

Open `README.md` and fill in the live demo URL near the top.

## Step 7 — Submit

For Brightspace you need:
1. **Deployed URL**: your HF Space URL.
2. **GitHub repo URL**: your GitHub repo URL.
3. **Technical report**: `docs/technical_report.md` — either submit as .md
   or convert to PDF (`pandoc docs/technical_report.md -o report.pdf` if you
   have pandoc; otherwise paste into Google Docs and export).

---

## Troubleshooting

**"OPENAI_API_KEY not found" in Streamlit.**
Local: you didn't copy `.env.example` to `.env`, or you didn't restart
`streamlit run` after editing. HF: the secret is named wrong or you haven't
saved it. Restart the Space from Settings.

**HF Space build fails.**
Check the **Logs** tab. Most common causes: a `duckduckgo_search` version
conflict (bump the pin in `requirements.txt` and re-push) or a Docker
build error — read the log, fix the `Dockerfile`, commit, `git push hf main`
again.

**The Traces tab is empty on HF.**
The free tier's container restarts wipe `data/` unless persistent storage is
attached. Just send a couple of chat messages and they'll appear again.

**Eval tool-selection accuracy is low (< 0.7).**
Likely the model is over-tooling (web-searching for things it knows). You
can tighten the system prompt in `agent/core.py` — specifically the
"If the question is conversational…" bullet. Rerun the eval.

**DuckDuckGo returns no results / rate-limits.**
Known, flaky. The tool returns a structured error; the agent usually copes.
If it's persistent, swap to a keyed search API (Tavily, Brave) — the change
is ~10 lines in `tools/web_search.py`.
