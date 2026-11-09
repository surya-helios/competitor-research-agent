"""Render a brief as Markdown, with a source list for every citation."""

from pathlib import Path

from .schemas import Brief, Claim, Evidence


def _source_line(label: str, claim: Claim | None, evidence: dict[str, Evidence]) -> str:
    if claim is None:
        return f"- **[{label}]** source unavailable"
    value = (claim.value or claim.status.replace("_", " ")).rstrip(". ")
    field = f"{claim.field} ({claim.item})" if claim.item else claim.field
    summary = f"{claim.competitor}, {field}: {value} ({claim.confidence} confidence)"
    sources = [evidence[eid] for eid in claim.evidence_ids if eid in evidence]
    if not sources:
        return f"- **[{label}]** {summary}. No source."
    links = "; ".join(
        f"[{s.title or s.url}]({s.url}) ({s.source_type.replace('_', ' ')}, "
        + (f"published {s.published_at}, " if s.published_at else "")
        + f"retrieved {s.retrieved_at.date()})"
        for s in sources
    )
    return f"- **[{label}]** {summary}. {links}"


def render_markdown(brief: Brief, claims: dict[str, Claim], evidence: dict[str, Evidence]) -> str:
    parts = [
        f"# {brief.title}",
        f"*Generated {brief.generated_at:%Y-%m-%d %H:%M} UTC*",
        "## Executive summary",
        brief.executive_summary,
        brief.body_markdown,
    ]
    if brief.gaps:
        parts += ["## Gaps", "\n".join(f"- {gap}" for gap in brief.gaps)]
    ordered = sorted(brief.citations.items(), key=lambda item: int(item[0][1:]))
    parts += [
        "## Sources",
        "\n".join(_source_line(label, claims.get(cid), evidence) for label, cid in ordered),
    ]
    return "\n\n".join(parts) + "\n"


def export_markdown(
    brief: Brief,
    claims: dict[str, Claim],
    evidence: dict[str, Evidence],
    out_dir: Path,
    run_id: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.md"
    path.write_text(render_markdown(brief, claims, evidence), encoding="utf-8")
    return path
