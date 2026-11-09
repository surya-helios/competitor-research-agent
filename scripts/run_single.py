"""Milestone 4 check: research one competitor and write a brief.

    uv run python -W ignore scripts/run_single.py "Notion" "Coda" coda.io
"""

import argparse
import logging
import uuid

from competitor_research_agent.agents.brief_writer import write_brief
from competitor_research_agent.agents.synthesizer import synthesize
from competitor_research_agent.deps import build_deps
from competitor_research_agent.export import export_markdown
from competitor_research_agent.graph import build_branch_graph
from competitor_research_agent.schemas import Candidate, CompanyProfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("company")
    parser.add_argument("competitor")
    parser.add_argument("domain", nargs="?")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    deps = build_deps()
    run_id = f"single-{uuid.uuid4().hex[:8]}"
    competitor = Candidate(
        name=args.competitor, domain=args.domain, rank=1, is_direct=True,
        overlap_score=1.0, rationale="Chosen manually for the milestone 4 check.",
    )

    branch = build_branch_graph(deps).invoke({
        "run_id": run_id,
        "competitor": competitor,
        "news_days": deps.settings.news_days,
        "retry_count": 0,
    })

    state = {
        "run_id": run_id,
        "company": CompanyProfile(name=args.company),
        "evidence": branch.get("evidence", {}),
        "claims": branch.get("claims", {}),
    }
    errors = list(branch.get("errors", []))
    for node in (synthesize, write_brief):
        update = node(state, deps)
        errors += update.pop("errors", [])
        state.update(update)

    print(f"run {run_id}: {len(state['evidence'])} evidence items, {len(state['claims'])} claims")
    for claim in sorted(state["claims"].values(), key=lambda c: (c.field, c.id)):
        print(f"  {claim.field:11} {claim.status:10} {claim.value or ''}"[:140])
    for error in errors:
        print(f"  error in {error.node}: {error.message}"[:200])

    if state.get("brief"):
        path = export_markdown(
            state["brief"], state["claims"], state["evidence"],
            deps.settings.data_dir / "briefs", run_id,
        )
        print(f"brief: {path}")
    for row in deps.ledger.summary(run_id):
        unit = "credits" if row.provider == "tavily" else "USD"
        print(f"  spend {row.provider:7} {row.node:15} {row.calls} calls {row.amount:.4f} {unit}")


if __name__ == "__main__":
    main()
