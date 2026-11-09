"""Run the full pipeline on the reference companies and report quality metrics.

Human checkpoints are answered automatically: the proposed competitors are
approved, flagged issues are passed through, and the brief is approved.

    uv run python -W ignore evals/run_evals.py            # all companies
    uv run python -W ignore evals/run_evals.py Notion Zoom

Writes evals/results/eval_<timestamp>.json and .md. Pricing accuracy still
needs a person: the report lists every verified price with its source to check.
"""

import json
import sys
import time
from pathlib import Path
from statistics import mean

import yaml

from competitor_research_agent.agents.scope import parse_alternative
from competitor_research_agent.deps import build_deps
from competitor_research_agent.runner import ResearchRunner
from competitor_research_agent.schemas import ResearchConstraints
from competitor_research_agent.state import active_claims

HERE = Path(__file__).parent
REQUIRED_FIELDS = ("pricing", "feature", "positioning")
AUTO_ANSWERS = {
    # Every case gives a website, so this only happens if identification fails.
    "confirm_company": lambda p: (
        {"action": "use", **parse_alternative(p["alternatives"][0])} if p["alternatives"] else {"action": "cancel"}
    ),
    "approve_competitors": lambda p: {"approved": [{"name": n} for n in p["proposed"]]},
    "review_issues": lambda p: {},
    "approve_brief": lambda p: {},
}


def matches(name: str, expected: list[str]) -> bool:
    name = name.casefold()
    return any(e.casefold() in name or name in e.casefold() for e in expected)


def evaluate(case: dict, runner: ResearchRunner, deps) -> dict:
    started = time.perf_counter()
    constraints = ResearchConstraints(
        market=case.get("market"), region=case.get("region"),
        news_days=deps.settings.news_days, max_competitors=deps.settings.max_competitors,
    )
    status = runner.start(case["company"], constraints, website=case.get("website"))
    checkpoints, proposed, issues = [], [], []
    while status.pending:
        point = status.pending["point"]
        checkpoints.append(point)
        if point == "approve_competitors":
            proposed = status.pending["proposed"]
        if point == "review_issues":
            issues = status.pending["issues"]
        status = runner.answer(status.run_id, AUTO_ANSWERS[point](status.pending))
    seconds = time.perf_counter() - started

    values = status.values
    all_claims = [c for c in values.get("claims", {}).values() if not c.superseded]
    found = [c for c in all_claims if c.status == "found"]
    active = active_claims(values.get("claims", {}))
    # The company itself is researched too, as the baseline.
    subjects = [values["company"].name] + [c.name for c in values.get("approved_competitors", [])]
    coverage = {
        name: [f for f in REQUIRED_FIELDS
               if any(c.competitor == name and c.field == f and c.status in ("found", "not_public") for c in active)]
        for name in subjects
    }
    spend = deps.ledger.summary(status.run_id)
    evidence = values.get("evidence", {})
    return {
        "company": case["company"],
        "run_id": status.run_id,
        "completed": status.done and bool(values.get("brief_path")),
        "seconds": round(seconds),
        "proposed": proposed,
        "discovery_hits": [n for n in proposed if matches(n, case["expected"])],
        "checkpoints": checkpoints,
        "issues": issues,
        "found_claims": len(found),
        "verified_claims": sum(1 for c in found if not c.excluded_reason),
        "high_confidence": sum(1 for c in found if not c.excluded_reason and c.confidence == "high"),
        "excluded": {r: sum(1 for c in found if c.excluded_reason == r)
                     for r in sorted({c.excluded_reason for c in found if c.excluded_reason})},
        "field_coverage": coverage,
        "branch_status": values.get("branch_status", {}),
        "tavily_credits": sum(r.amount for r in spend if r.provider == "tavily"),
        "nebius_usd": round(sum(r.amount for r in spend if r.provider == "nebius"), 4),
        "retries": sum(r.calls for r in spend if r.node == "assess_evidence") - len(subjects),
        "prices_to_check": [
            {"competitor": c.competitor, "plan": c.item, "value": c.value, "confidence": c.confidence,
             "source": next((evidence[e].url for e in c.evidence_ids if e in evidence), None)}
            for c in active if c.field == "pricing" and c.status == "found"
        ],
        "brief_path": values.get("brief_path"),
    }


def summarise(results: list[dict]) -> str:
    done = [r for r in results if r["completed"]]
    hits = sum(len(r["discovery_hits"]) for r in results)
    proposed = sum(len(r["proposed"]) for r in results)
    cells = [(name, fields) for r in results for name, fields in r["field_coverage"].items()]
    found = sum(r["found_claims"] for r in results)
    verified = sum(r["verified_claims"] for r in results)
    lines = [
        "# Evaluation report",
        "",
        f"Runs: {len(results)}, completed with a brief: {len(done)}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Discovery: proposed competitors in the reference list | {hits}/{proposed} ({hits / max(proposed, 1):.0%}) |",
        f"| Companies (baseline + competitors) with pricing, features, and positioning | "
        f"{sum(len(f) == 3 for _, f in cells)}/{len(cells)} |",
        f"| Field coverage (pricing / feature / positioning) | "
        + " / ".join(f"{sum(f in fields for _, fields in cells)}/{len(cells)}" for f in REQUIRED_FIELDS) + " |",
        f"| Found claims that passed verification | {verified}/{found} ({verified / max(found, 1):.0%}) |",
        f"| Runs that needed issue review | {sum('review_issues' in r['checkpoints'] for r in results)}/{len(results)} |",
        f"| Mean time per run | {mean(r['seconds'] for r in results):.0f} s |",
        f"| Mean Tavily credits per run | {mean(r['tavily_credits'] for r in results):.1f} |",
        f"| Mean Nebius cost per run | ${mean(r['nebius_usd'] for r in results):.4f} |",
        "",
        "## Per company",
        "",
        "| Company | Proposed | In reference list | Verified / found claims | Credits | Time | Retries |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['company']} | {', '.join(r['proposed'])} | {len(r['discovery_hits'])}/{len(r['proposed'])} "
            f"| {r['verified_claims']}/{r['found_claims']} | {r['tavily_credits']:g} | {r['seconds']} s "
            f"| {r['retries']} |"
        )
    reasons: dict[str, int] = {}
    for r in results:
        for reason, count in r["excluded"].items():
            key = reason.split(" (")[0]
            reasons[key] = reasons.get(key, 0) + count
    lines += ["", "## Why claims were excluded", ""]
    lines += [f"- {reason}: {count}" for reason, count in sorted(reasons.items(), key=lambda x: -x[1])] or ["- none"]
    lines += ["", "## Prices to check by hand", "", "| Competitor | Plan | Price | Confidence | Source |",
              "|---|---|---|---|---|"]
    for r in results:
        for p in r["prices_to_check"]:
            lines.append(f"| {p['competitor']} | {p['plan'] or ''} | {p['value']} | {p['confidence']} | {p['source']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    cases = yaml.safe_load((HERE / "companies.yaml").read_text(encoding="utf-8"))
    if sys.argv[1:]:
        wanted = {name.casefold() for name in sys.argv[1:]}
        cases = [c for c in cases if c["company"].casefold() in wanted]
    deps = build_deps()
    runner = ResearchRunner(deps)
    results = []
    for case in cases:
        print(f"Running {case['company']}...", flush=True)
        try:
            result = evaluate(case, runner, deps)
        except Exception as exc:  # Keep going; record the failure.
            result = {"company": case["company"], "completed": False, "error": f"{type(exc).__name__}: {exc}",
                      "seconds": 0, "proposed": [], "discovery_hits": [], "checkpoints": [], "found_claims": 0,
                      "verified_claims": 0, "excluded": {}, "field_coverage": {}, "tavily_credits": 0,
                      "nebius_usd": 0, "retries": 0, "prices_to_check": []}
        results.append(result)
        print(f"  {result.get('run_id', '-')}: completed={result['completed']} "
              f"hits={len(result['discovery_hits'])}/{len(result['proposed'])} "
              f"verified={result['verified_claims']}/{result['found_claims']} "
              f"credits={result['tavily_credits']:g} {result.get('error', '')}", flush=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"eval_{stamp}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (out / f"eval_{stamp}.md").write_text(summarise(results), encoding="utf-8")
    print(f"\nReport: {out / f'eval_{stamp}.md'}")


if __name__ == "__main__":
    main()
