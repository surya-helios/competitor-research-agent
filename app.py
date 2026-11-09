"""Streamlit interface for the competitor research pipeline.

    uv run streamlit run app.py
"""

import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

from competitor_research_agent.agents.scope import parse_alternative
from competitor_research_agent.deps import build_deps
from competitor_research_agent.inputs import InvalidInput
from competitor_research_agent.runner import ResearchRunner, RunStatus
from competitor_research_agent.schemas import ResearchConstraints

# LangChain and the OpenAI SDK emit harmless Pydantic serializer warnings.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

st.set_page_config(page_title="Competitor Research", page_icon="🔎", layout="wide")

STEP_LABELS = {
    "scope_search": "Searched for the company",
    "resolve_scope": "Identified the company",
    "confirm_company": "Company confirmed",
    "plan_discovery": "Planned competitor searches",
    "discovery_search": "Searched for competitors",
    "rank_candidates": "Ranked candidate competitors",
    "approve_competitors": "Competitors approved",
    "research_branch": "Finished researching a company",
    "issue_gate": "Checked the research for issues",
    "review_issues": "Issues reviewed",
    "synthesize": "Compared the competitors",
    "write_brief": "Wrote the brief",
    "approve_brief": "Brief reviewed",
    "export_brief": "Saved the brief",
}


@st.cache_resource
def get_app():
    deps = build_deps()
    return ResearchRunner(deps), deps


runner, deps = get_app()


def md(text: str) -> str:
    """Escape '$' so Streamlit doesn't render prices as LaTeX."""
    return text.replace("$", r"\$")


def run_action(action, label: str, note: str | None = None) -> RunStatus | None:
    """Call the runner with a live progress log. Returns None on failure."""
    with st.status(label, expanded=True) as box:
        if note:
            box.write(note)
        box.caption("Please wait here: clicking elsewhere in the app stops this step (you can resume it later).")
        try:
            status = action(lambda node: box.write(STEP_LABELS.get(node, node)))
        except InvalidInput as exc:
            box.update(label="Input not accepted", state="error")
            st.error(str(exc))
            return None
        except Exception as exc:  # The run is checkpointed; it can be resumed.
            box.update(label="The run stopped", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
            return None
        box.update(label="Done", state="complete", expanded=False)
    return status


def advance(action, label: str, note: str | None = None) -> None:
    status = run_action(action, label, note)
    if status is not None:
        st.session_state.run_id = status.run_id
        st.rerun()


# --- Sidebar: new run, previous runs, spend ----------------------------------

with st.sidebar:
    st.header("New research")
    # Enter must not submit: each submission reloads the page, which stops a run
    # that is still in progress (it can only be resumed later).
    with st.form("new_run", enter_to_submit=False):
        company = st.text_input("Company name *", placeholder="e.g. Linear")
        website = st.text_input("Website (optional)", placeholder="e.g. linear.app",
                                help="Helps when the name is common or shared by several companies.")
        market = st.text_input("Market / industry (optional)", placeholder="e.g. issue tracking software")
        region = st.text_input("Region (optional)", placeholder="e.g. India; leave empty for global")
        submitted = st.form_submit_button("Start research", type="primary", width="stretch")
    if submitted:
        # Started from the main area, so progress shows there.
        st.session_state.start_request = {
            "company": company, "website": website, "market": market, "region": region,
        }

    runs = runner.list_runs()
    if runs:
        st.header("Previous runs")
        labels = {f"{name} ({run_id})": run_id for run_id, name in runs}
        current = st.session_state.get("run_id")
        options = list(labels)
        index = next((i for i, key in enumerate(options) if labels[key] == current), 0)
        chosen = st.selectbox("Open a run", options, index=index, label_visibility="collapsed")
        if st.button("Open", width="stretch"):
            st.session_state.run_id = labels[chosen]
            st.rerun()

    if st.session_state.get("run_id"):
        spend = deps.ledger.summary(st.session_state.run_id)
        if spend:
            credits = sum(r.amount for r in spend if r.provider == "tavily")
            usd = sum(r.amount for r in spend if r.provider == "nebius")
            st.header("This run's spend")
            st.write(f"Tavily: **{credits:g}** / {deps.settings.tavily_credit_limit:g} credits")
            st.write(f"Nebius: **${usd:.4f}** / ${deps.settings.nebius_usd_limit:.2f}")


# --- Shared views -------------------------------------------------------------

def role(values: dict, name: str) -> str:
    company = values.get("company")
    return "your company" if company and name == company.name else "competitor"


def by_role(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """Sort with the company being analysed first."""
    return frame.sort_values(["Role", *columns], ascending=[False, *[True] * len(columns)])


def claims_frame(values: dict, include_excluded: bool) -> pd.DataFrame:
    evidence = values.get("evidence", {})
    rows = []
    for c in values.get("claims", {}).values():
        if c.superseded or (c.excluded_reason and not include_excluded):
            continue
        source = next((evidence[e] for e in c.evidence_ids if e in evidence), None)
        rows.append({
            "Role": role(values, c.competitor),
            "Company": c.competitor, "Field": c.field, "Item": c.item or "", "Value": c.value or "",
            "Status": c.status, "Confidence": c.confidence, "Checks": ", ".join(c.checks),
            "Excluded because": c.excluded_reason or "", "Source": source.url if source else "",
            "Claim id": c.id,
        })
    return by_role(pd.DataFrame(rows), "Company", "Field") if rows else pd.DataFrame()


def show_claims(values: dict, key: str) -> None:
    include = st.toggle("Show excluded claims", key=f"excluded-{key}")
    frame = claims_frame(values, include)
    if frame.empty:
        st.write("No claims yet.")
    else:
        st.dataframe(frame, hide_index=True, width="stretch",
                     column_config={"Source": st.column_config.LinkColumn()})


def show_evidence(values: dict) -> None:
    rows = [
        {"Role": role(values, e.competitor), "Company": e.competitor,
         "Type": e.source_type.replace("_", " "), "Title": e.title or "",
         "URL": e.url, "Published": e.published_at, "Retrieved": e.retrieved_at.date()}
        for e in values.get("evidence", {}).values()
    ]
    if rows:
        st.dataframe(by_role(pd.DataFrame(rows), "Company"), hide_index=True, width="stretch",
                     column_config={"URL": st.column_config.LinkColumn()})
    else:
        st.write("No evidence yet.")


def show_details(status: RunStatus) -> None:
    values = status.values
    company, constraints = values.get("company"), values.get("constraints")
    if company:
        st.subheader("Company")
        st.write(f"**{company.name}** ({company.domain or 'website unknown'}): {company.description or ''}")
    if constraints:
        st.write(f"Market: {constraints.market or 'not specified'} · Region: {constraints.region or 'global'}")
    if values.get("approved_competitors"):
        st.subheader("Approved competitors")
        st.write(", ".join(f"{c.name} ({c.domain or '?'})" for c in values["approved_competitors"]))
    if values.get("human_decisions"):
        st.subheader("Review decisions")
        for d in values["human_decisions"]:
            st.write(f"- {d.point.replace('_', ' ')} at {d.at:%H:%M}: `{d.payload}`")
    if values.get("errors"):
        st.subheader("Notes and errors")
        for e in values["errors"]:
            st.write(f"- **{e.node}**{f' ({e.competitor})' if e.competitor else ''}: {md(e.message)}")
    spend = deps.ledger.summary(status.run_id)
    if spend:
        st.subheader("Spend by step")
        st.dataframe(
            pd.DataFrame([{"Provider": r.provider, "Step": r.node, "Model": r.model or "", "Calls": r.calls,
                           "Amount": f"{r.amount:g} credits" if r.provider == "tavily" else f"${r.amount:.4f}"}
                          for r in spend]),
            hide_index=True, width="stretch",
        )


# --- Human checkpoints ---------------------------------------------------------

def confirm_company_view(run_id: str, pending: dict) -> None:
    company = pending["company"]
    if pending["reason"] == "not_found":
        st.warning(f"Couldn't identify a company called **{company['name']}**. "
                   "Check the spelling, or add its website.")
        with st.form("search_again", enter_to_submit=False):
            name = st.text_input("Company name", value=company["name"])
            website = st.text_input("Website (optional)", value=company.get("domain") or "")
            again = st.form_submit_button("Search again", type="primary")
        if again:
            advance(lambda cb: runner.answer(run_id, {"action": "search_again", "name": name, "domain": website}, cb),
                    "Searching again…")
    else:
        st.info(f"**{company['name']}** could mean several companies. Which one do you mean?")
        choice = st.radio("Company", pending["alternatives"], label_visibility="collapsed")
        if st.button("Use this company", type="primary"):
            advance(lambda cb: runner.answer(run_id, {"action": "use", **parse_alternative(choice)}, cb),
                    "Finding competitors…")
    if st.button("Cancel run"):
        advance(lambda cb: runner.answer(run_id, {"action": "cancel"}, cb), "Cancelling…")


def approve_competitors_view(run_id: str, pending: dict) -> None:
    limit = deps.settings.max_competitors
    st.subheader("Approve competitors to research")
    company = runner.status(run_id).values["company"]
    st.write(f"Select up to **{limit}**. You can correct a website before research starts. "
             f"**{company.name}** is researched too, as the baseline the competitors are compared against.")
    remembered = pending.get("remembered")
    if remembered:
        when = remembered["approved_at"][:10]
        note, button = st.columns([4, 1])
        note.info(f"These are the competitors you approved on **{when}** (run {remembered['run_id']}), "
                  "so Discovery was skipped. Keep them for a like-for-like comparison, or search again.")
        if button.button("Find competitors again"):
            advance(lambda cb: runner.answer(run_id, {"rediscover": True}, cb), "Finding competitors…",
                    "Searching for competitors again. This takes about a minute.")
    elif pending["shortfall"] > 0:
        st.warning(f"Only {len(pending['proposed'])} direct competitors were found. You can add more below.")
    candidates = pending["candidates"]
    frame = pd.DataFrame([{
        "Research": c["name"] in pending["proposed"], "Name": c["name"], "Website": c.get("domain") or "",
        "Type": "direct" if c["is_direct"] else "indirect", "Overlap": c["overlap_score"],
        "Why": c["rationale"], "Sources": len(c.get("source_urls", [])),
    } for c in candidates])
    edited = st.data_editor(
        frame, hide_index=True, width="stretch",
        disabled=["Name", "Type", "Overlap", "Why", "Sources"],
        column_config={"Overlap": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")},
    ) if candidates else frame
    added = st.text_input("Add competitors (optional)", placeholder="Name|website, Name|website",
                          help="Comma-separated. The website after | is optional.")

    approved = [{"name": r["Name"], "domain": r["Website"] or None}
                for r in edited.to_dict("records") if r.get("Research")] if candidates else []
    for entry in (e.strip() for e in added.split(",") if e.strip()):
        name, _, site = entry.partition("|")
        approved.append({"name": name.strip(), "domain": site.strip() or None})

    left, right = st.columns([1, 4])
    if left.button("Research selected", type="primary", disabled=not approved):
        if len(approved) > limit:
            st.error(f"Select at most {limit} competitors.")
        else:
            names = ", ".join(a["name"] for a in approved)
            advance(lambda cb: runner.answer(run_id, {"approved": approved}, cb),
                    f"Researching {names}…",
                    "Your company and each competitor are researched in parallel, verified, "
                    "and retried for gaps. This usually takes 1–3 minutes.")
    if right.button("Cancel run"):
        advance(lambda cb: runner.answer(run_id, {"cancel": True}, cb), "Cancelling…")


def review_issues_view(run_id: str, pending: dict, values: dict) -> None:
    st.subheader("Review issues found after research")
    st.dataframe(pd.DataFrame(pending["issues"]).rename(columns=str.title), hide_index=True,
                 width="stretch")
    subjects = [values["company"].name] + [c.name for c in values.get("approved_competitors", [])]
    conflicts = {f"{i['competitor']}: {i['detail']}": i["claim_id"] for i in pending["issues"] if i.get("claim_id")}
    with st.form("review", enter_to_submit=False):
        rerun = st.multiselect("Research these companies again", subjects)
        exclude = st.multiselect("Leave these claims out of the brief", list(conflicts))
        go = st.form_submit_button("Continue", type="primary")
    if go:
        answer = {"rerun": rerun, "exclude_claims": [conflicts[c] for c in exclude]}
        label = f"Researching {', '.join(rerun)} again…" if rerun else "Writing the brief…"
        advance(lambda cb: runner.answer(run_id, answer, cb), label)
    if st.button("Cancel run"):
        advance(lambda cb: runner.answer(run_id, {"cancel": True}, cb), "Cancelling…")
    with st.expander("Claims so far"):
        show_claims(values, "review")


def approve_brief_view(run_id: str, pending: dict) -> None:
    st.subheader("Review the draft brief")
    with st.container(border=True):
        st.markdown(md(pending["markdown"]))
    with st.form("brief"):
        feedback = st.text_area("Changes you want (only for 'Request changes')",
                                placeholder="e.g. Shorten the implications and add a pricing summary table.")
        left, right = st.columns([1, 4])
        approve = left.form_submit_button("Approve brief", type="primary")
        revise = right.form_submit_button("Request changes")
    if approve:
        advance(lambda cb: runner.answer(run_id, {}, cb), "Saving the brief…")
    elif revise:
        if not feedback.strip():
            st.error("Describe the changes you want first.")
        else:
            advance(lambda cb: runner.answer(run_id, {"feedback": feedback}, cb), "Rewriting the brief…")


# --- Main ---------------------------------------------------------------------

request = st.session_state.pop("start_request", None)
if request:
    constraints = ResearchConstraints(
        market=request["market"], region=request["region"],
        news_days=deps.settings.news_days, max_competitors=deps.settings.max_competitors,
    )
    advance(
        lambda cb: runner.start(request["company"], constraints, website=request["website"], on_step=cb),
        f"Finding competitors for {request['company']}…",
        "Identifying the company and searching for competitors. This takes about a minute.",
    )

run_id = st.session_state.get("run_id")
if not run_id:
    st.title("Competitor Research")
    st.write(
        "Enter a company in the sidebar. The system identifies it, finds its top direct "
        "competitors, researches each one in parallel with Tavily, verifies every claim "
        "against its source, and writes a cited competitor brief."
    )
    st.write("You review four points: an unclear company, the competitor list, any issues "
             "the verifier flags, and the final brief.")
    st.stop()

status = runner.status(run_id)
values = status.values
company = values.get("company")
st.title(f"Competitors of {company.name}" if company else "Competitor Research")
st.caption(f"Run {run_id}")

if status.pending:
    point = status.pending["point"]
    if point == "confirm_company":
        confirm_company_view(run_id, status.pending)
    elif point == "approve_competitors":
        approve_competitors_view(run_id, status.pending)
    elif point == "review_issues":
        review_issues_view(run_id, status.pending, values)
    elif point == "approve_brief":
        approve_brief_view(run_id, status.pending)
elif status.stopped:
    st.warning("This run stopped before finishing, for example after an error or restart. "
               "Completed steps are saved.")
    if st.button("Resume", type="primary"):
        advance(lambda cb: runner.continue_run(run_id, cb), "Resuming…")
elif status.cancelled_reason:
    st.info(f"Run cancelled: {status.cancelled_reason}")
else:
    brief_tab, claims_tab, sources_tab, details_tab = st.tabs(["Brief", "Claims", "Sources", "Run details"])
    with brief_tab:
        path = values.get("brief_path")
        if path and Path(path).exists():
            text = Path(path).read_text(encoding="utf-8")
            st.download_button("Download Markdown", text, file_name=Path(path).name, mime="text/markdown")
            st.markdown(md(text))
        else:
            st.warning("No brief was produced. See Run details for errors.")
    with claims_tab:
        show_claims(values, "done")
    with sources_tab:
        show_evidence(values)
    with details_tab:
        show_details(status)

if status.pending or status.stopped:
    with st.expander("Run details"):
        show_details(status)
