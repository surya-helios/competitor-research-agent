"""Run the full pipeline from the command line, answering human checkpoints
in the terminal (or automatically with --yes).

    uv run python -W ignore scripts/run_full.py "Linear" --website linear.app
    uv run python -W ignore scripts/run_full.py --resume run-1a2b3c4d
"""

import argparse
import logging

from competitor_research_agent.agents.scope import parse_alternative
from competitor_research_agent.deps import build_deps
from competitor_research_agent.inputs import InvalidInput
from competitor_research_agent.runner import ResearchRunner, RunStatus
from competitor_research_agent.schemas import ResearchConstraints


def ask_company(pending: dict, auto: bool) -> dict:
    if pending["reason"] == "not_found":
        print(f"\nCouldn't identify a company called '{pending['company']['name']}'.")
        typed = "" if auto else input("Corrected name (or name|website), or Enter to cancel: ").strip()
        if not typed:
            return {"action": "cancel"}
        name, _, domain = typed.partition("|")
        return {"action": "search_again", "name": name.strip(), "domain": domain.strip() or None}

    alternatives = pending["alternatives"]
    print("\nThe company name is ambiguous:")
    for i, option in enumerate(alternatives, start=1):
        print(f"  {i}. {option}")
    choice = 1 if auto else int(input("Which one? (0 to cancel) ") or 1)
    if choice == 0:
        return {"action": "cancel"}
    return {"action": "use", **parse_alternative(alternatives[choice - 1])}


def ask_competitors(pending: dict, auto: bool) -> dict:
    print("\nCandidate competitors:")
    for c in pending["candidates"]:
        kind = "direct" if c["is_direct"] else "indirect"
        print(f"  {c['rank']}. {c['name']} ({c['domain']}) {kind}, overlap {c['overlap_score']:.2f}: {c['rationale']}")
    print(f"Proposed: {', '.join(pending['proposed']) or 'none'}")
    if pending.get("remembered"):
        print(f"(Approved in run {pending['remembered']['run_id']}; type 'rediscover' to search again.)")
    elif pending["shortfall"] > 0:
        print(f"Only {len(pending['proposed'])} direct competitors found; you can add more.")
    by_name = {c["name"].casefold(): c for c in pending["candidates"]}
    names = pending["proposed"]
    if not auto:
        typed = input(
            "Enter to approve, 'cancel' to stop, or comma-separated names (name|website to add or fix a site): "
        ).strip()
        if typed.lower() == "cancel":
            return {"cancel": True}
        if typed.lower() == "rediscover":
            return {"rediscover": True}
        if typed:
            names = [n.strip() for n in typed.split(",") if n.strip()]
    approved = []
    for entry in names:
        name, _, domain = entry.partition("|")
        known = by_name.get(name.strip().casefold())
        approved.append({"name": known["name"] if known else name.strip(), "domain": domain.strip() or None})
    return {"approved": approved}


def ask_issues(pending: dict, auto: bool) -> dict:
    print("\nIssues found after research:")
    for issue in pending["issues"]:
        claim = f" [{issue['claim_id']}]" if issue.get("claim_id") else ""
        print(f"  {issue['competitor']}: {issue['kind']}: {issue['detail']}{claim}")
    if auto:
        return {}
    typed = input("Enter to continue, 'rerun A,B', 'exclude <claim id>,...', or 'cancel': ").strip()
    command, _, rest = typed.partition(" ")
    items = [x.strip() for x in rest.split(",") if x.strip()]
    if command == "cancel":
        return {"cancel": True}
    if command == "rerun":
        return {"rerun": items}
    if command == "exclude":
        return {"exclude_claims": items}
    return {}


def ask_brief(pending: dict, auto: bool) -> dict:
    print(f"\nDraft brief: {pending['title']}\n")
    print(pending["markdown"][:1500] + ("\n..." if len(pending["markdown"]) > 1500 else ""))
    if auto:
        return {}
    feedback = input("\nEnter to approve, or type changes you want: ").strip()
    return {"feedback": feedback} if feedback else {}


def report(status: RunStatus, deps) -> None:
    values = status.values
    outcome = f"cancelled: {status.cancelled_reason}" if status.cancelled_reason else (
        "done" if status.done else "not finished")
    print(f"\nRun {status.run_id}: {outcome}")
    for name, branch in values.get("branch_status", {}).items():
        claims = [c for c in values.get("claims", {}).values() if c.competitor == name and not c.superseded]
        found = [c for c in claims if c.status == "found" and not c.excluded_reason]
        excluded = [c for c in claims if c.excluded_reason]
        print(f"  {name}: {branch}, {len(found)} verified claims, {len(excluded)} excluded")
        for claim in excluded:
            print(f"    excluded {claim.field} '{(claim.value or '')[:60]}': {claim.excluded_reason}")
    for error in values.get("errors", []):
        print(f"  error in {error.node} ({error.competitor or '-'}): {error.message}"[:200])
    if values.get("brief_path"):
        print(f"Brief: {values['brief_path']}")
    for row in deps.ledger.summary(status.run_id):
        unit = "credits" if row.provider == "tavily" else "USD"
        print(f"  spend {row.provider:7} {row.node:18} {row.calls} calls {row.amount:.4f} {unit}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("company", nargs="?")
    parser.add_argument("--website", help="The company's website, e.g. linear.app (optional).")
    parser.add_argument("--market", help="Market or industry, e.g. 'issue tracking software' (optional).")
    parser.add_argument("--region", help="Region, e.g. 'India' or 'North America' (optional; default global).")
    parser.add_argument("--yes", action="store_true", help="Approve proposals automatically.")
    parser.add_argument("--resume", metavar="RUN_ID", help="Continue a paused or stopped run.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    deps = build_deps()
    runner = ResearchRunner(deps)
    if args.resume:
        status = runner.status(args.resume)
        if status.stopped:
            status = runner.continue_run(args.resume)
    else:
        constraints = ResearchConstraints(
            market=args.market, region=args.region,
            news_days=deps.settings.news_days, max_competitors=deps.settings.max_competitors,
        )
        try:
            status = runner.start(args.company, constraints, website=args.website)
        except InvalidInput as exc:
            raise SystemExit(f"Not started: {exc}")
        print(f"Started {status.run_id}")

    while status.pending:
        point = status.pending["point"]
        if point == "confirm_company":
            answer = ask_company(status.pending, args.yes)
        elif point == "approve_competitors":
            answer = ask_competitors(status.pending, args.yes)
        elif point == "review_issues":
            answer = ask_issues(status.pending, args.yes)
        elif point == "approve_brief":
            answer = ask_brief(status.pending, args.yes)
        else:
            raise SystemExit(f"No terminal handler for {point}")
        try:
            status = runner.answer(status.run_id, answer)
        except InvalidInput as exc:
            print(f"Not accepted: {exc}")  # The run stays paused; ask again.

    report(status, deps)


if __name__ == "__main__":
    main()
