# Competitor Research Agent

A multi-agent system that finds a company's top 3 direct competitors, researches
the company and each competitor in parallel, verifies every claim against its
source, and writes a cited brief comparing the competitors against the company.
A person reviews the work at four checkpoints.

Built with LangGraph, Tavily (search and page extraction), Nebius Token Factory
(LLMs), and Streamlit.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
Copy-Item .env.example .env        # then add NEBIUS_API_KEY and TAVILY_API_KEY
uv run python -W ignore -m competitor_research_agent.startup   # every line should say [ok]
uv run streamlit run app.py        # opens http://localhost:8501
```

In the app, enter a company name and optionally its website, market, and region,
then follow the review steps. A run takes about 2–4 minutes and costs roughly
35–50 Tavily credits and $0.03–0.05 of Nebius usage. A repeat run for the same
company reuses the competitors you approved and skips Discovery.

There is also a command-line runner:

```powershell
uv run python -W ignore scripts/run_full.py "Linear" --website linear.app --market "issue tracking software"
uv run python -W ignore scripts/run_full.py --resume <run id>    # continue a paused or stopped run
```

## How it works

```text
Company name (+ optional website, market, region)
  │  input checks: junk is rejected before anything is spent
  ▼
SCOPE      scope_search (A) → resolve_scope (R) → confirm_company (H, only if unclear or not found)
  ▼
DISCOVERY  recall_competitors (A: reuse the competitors approved for this company before)
           or plan_discovery (R) → discovery_search (A) → rank_candidates (R)
           → approve_competitors (H)
  ▼  fan_out: one parallel branch for the company (the baseline) and one per competitor
RESEARCH   plan_queries (R) → run_searches (A) → map_pricing_pages (A) → extract_pages (A)
BRANCH     → extract_claims (R) → check_quotes (A) → assess_evidence (R)
           → retry the gaps (max 2, budget permitting) or finish
  ▼  issue_gate: waits for all branches
REVIEW     review_issues (H, only if something is flagged: rerun, exclude claims, or continue)
  ▼
OUTPUT     synthesize (R) → write_brief (R) → approve_brief (H, feedback rewrites) → export_brief (A)
```

R = reasoning node (LLM), A = action node (tools or code), H = human checkpoint.

| Agent | Role |
|---|---|
| Orchestrator | The LangGraph graph itself: fixed routing in code, with the LLM making bounded judgements at specific nodes |
| Scope Resolver | Identifies which company the user means; says "not found" rather than inventing one |
| Agent 1: Discovery | Finds candidates with Tavily and ranks direct competitors (same category, same buyer, realistic substitute, sells in the region); reuses an earlier approval for the same company, market, and region |
| Agent 2: Research | Researches the company (as the baseline) and each competitor: plans searches, searches first-party and third-party sources, maps the site with Tavily Map to find pricing pages search misses, and extracts full pages |
| Agent 3: Extraction | Turns evidence into structured claims (pricing per plan, features, positioning, news), each with an exact quote and source |
| Verifier | Code checks (quote is in the source, source names the competitor, news is recent) plus an LLM review (conflicts, wrong-company claims, gaps) |
| Synthesizer / Brief Writer | Compare competitors and write the brief, using only verified claims and citing each one |

Reasoning nodes never call tools, and action nodes never call the LLM. Web content
is treated as data: it can't trigger an action, and every paid call goes through
one place that checks the budget.

### Human checkpoints

| Checkpoint | When | Choices |
|---|---|---|
| Confirm company | Only if the name is ambiguous or no company was found | Use one, search again with a corrected name or website, or cancel |
| Approve competitors | Always | Select up to 3, fix a website, add your own, find competitors again (when they were remembered), or cancel |
| Review issues | Only if a branch failed, a required field is missing, or claims conflict | Rerun competitors, exclude claims, continue, or cancel |
| Approve brief | Always | Approve, or describe changes for a rewrite |

### State and storage

| Layer | Holds | Where |
|---|---|---|
| Run state | Company, competitors, evidence, claims, decisions, brief | LangGraph state (`state.py`) |
| Branch state | One competitor's queries, evidence, claims, retries, gaps | Private to each parallel branch |
| Checkpoints | State after every step, so runs pause and resume | `data/checkpoints.sqlite` |
| Search cache | Tavily results with expiry (news 24 h, pages 7 days) | `data/app.sqlite` |
| Budget ledger | Every paid call's reserved and actual cost | `data/app.sqlite` |
| Competitor memory | Competitors approved per company, market, and region (kept 90 days) | `data/app.sqlite` |

Parallel branches merge through reducers keyed by claim id, so they can't
overwrite each other, and a rerun replaces old claims instead of mixing with them.
The cache and ledger are written immediately, so a crash never loses spending
records or repeats a paid search. The checkpointer only reloads this project's
own data types.

### Failure handling

The order of fallbacks is: retry → change the query → try another source → ask a
person → deliver with the gap marked. Nothing is invented to fill a gap.

| Failure | Response |
|---|---|
| Tavily rate limit, timeout, or server error | Retry with backoff; failed calls aren't charged or cached |
| Bad key or plan limit | Stop immediately with a clear error |
| LLM returns invalid JSON | JSON mode → validation → one repair attempt → backup model |
| Wrong competitor website | Searches fall back to the whole web; the run notes the domain may be wrong; fix it at approval |
| Pricing page missing from search | Maps the site for pricing-like paths with Tavily Map, and tries `<domain>/pricing` |
| Unsupported or wrong-company claim | Excluded with a reason, visible in the app |
| Stale news or old third-party prices | Excluded (news window: 90 days) |
| Gaps after research | Targeted retries, at most 2, while budget remains |
| One competitor's research fails | Other branches continue; the reviewer can rerun it |
| Crash or restart | Resume from the last checkpoint without repeating searches |
| Budget limit reached | Stops spending and delivers what was verified |

## Models

Models were chosen by `evals/model_selection.py`, which tests candidates on
extraction, evidence review, and competitor ranking:

| Role | Model | Used for |
|---|---|---|
| Fast | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Planning searches, identifying the company |
| Strong | `Qwen/Qwen3-235B-A22B-Instruct-2507` | Ranking, extraction, verification, synthesis, writing |
| Backup | `deepseek-ai/DeepSeek-V4.1-Flash` | Any role, when the primary model fails |

Native JSON-schema output isn't used: on Nebius, Qwen models gave every item the
same enum value in that mode, while JSON mode labelled them correctly.

## Evaluation

`uv run python -W ignore evals/run_evals.py` runs the full pipeline on the 8
reference companies in `evals/companies.yaml` (Linear, Notion, Figma, HubSpot,
Zoom, Mailchimp, Shopify, and Razorpay in India), approving each checkpoint
automatically, and writes a report to `evals/results/`.

Results from 2026-11-08 ([full report](evals/results/eval_20261108-153140.md)):

| Metric | Result |
|---|---|
| Runs completed with a brief | 8/8 |
| Proposed competitors in the reference lists | 24/24 (100%) |
| Competitors with pricing, features, and positioning | 22/24 |
| Found claims that passed verification | 261/278 (94%) |
| Runs that needed issue review | 3/8 |
| Mean time / Tavily credits / Nebius cost per run | 96 s / 30 / $0.029 |

Excluded claims were mostly news older than 90 days (9) and quotes not found in
their source (7); one claim was about a different company. The report lists every
verified price with its source so pricing accuracy can be checked by hand.

After this run, old first-party prices (like a 2022 press release) are flagged as
low confidence instead of passing as high confidence. The run also predates the
company baseline and Tavily Map; rerun it for current figures.

## Project layout

```text
app.py                          Streamlit interface
src/competitor_research_agent/
  agents/                       scope, discovery, research, extraction, verifier,
                                synthesizer, brief_writer
  graph.py                      graph wiring, fan-out, checkpointer
  runner.py                     start / answer / resume / list runs
  state.py, schemas.py          state and data models
  llm.py                        Nebius calls with validation and fallbacks
  tools/                        Tavily client and cache
  budget.py, inputs.py, config.py, startup.py, export.py
scripts/                        command-line runners
evals/                          model selection and pipeline evaluation
tests/                          offline tests (no API calls)
```

## Tests

```powershell
uv run pytest
```

The tests use fake LLM and search clients, so they cost nothing. They cover the
budget ledger, cache, retries and fallbacks, every human checkpoint, parallel
branches, crash recovery, and the Verifier's checks.

## Known limitations

- Pages that render prices with JavaScript widgets can extract badly; such prices
  are kept at low confidence and marked in the brief.
- Discovery can propose a wrong website for a competitor; correct it at approval.
- With no region set, a competitor's site may still serve local prices (e.g. Zoho
  CRM in rupees); set a region to make the currency explicit.
- Large vendors (e.g. Microsoft) can yield many pricing claims for related
  products; the brief summarises them, and the Claims tab lists them all.
- Remembered competitors are reused only for the same website (or name), market,
  and region; change any of them, or click "Find competitors again", for fresh ones.
- Single user, local SQLite storage.
