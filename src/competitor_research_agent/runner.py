"""Start, pause, resume, and inspect research runs. Used by the CLI script and
the Streamlit app; each run is a LangGraph thread identified by its run id."""

import uuid
from dataclasses import dataclass
from typing import Any, Callable

from langgraph.types import Command

from .deps import Deps
from .graph import build_research_graph, open_checkpointer
from .inputs import clean_company_name, clean_domain, clean_market, clean_region
from .schemas import CompanyProfile, ResearchConstraints


def _clean_answer(answer: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(answer)
    if cleaned.get("name"):
        cleaned["name"] = clean_company_name(cleaned["name"])
    if "domain" in cleaned:
        cleaned["domain"] = clean_domain(cleaned["domain"])
    if "approved" in cleaned:
        cleaned["approved"] = [
            {**choice, "name": clean_company_name(choice["name"]), "domain": clean_domain(choice.get("domain"))}
            for choice in cleaned["approved"]
        ]
    return cleaned


@dataclass
class RunStatus:
    run_id: str
    values: dict[str, Any]
    pending: dict[str, Any] | None  # The human checkpoint waiting for an answer
    next_nodes: tuple[str, ...]

    @property
    def done(self) -> bool:
        return not self.next_nodes and self.pending is None

    @property
    def cancelled_reason(self) -> str | None:
        return self.values.get("cancelled_reason")

    @property
    def stopped(self) -> bool:
        """Stopped mid-run without a pending question, e.g. after a crash."""
        return bool(self.next_nodes) and self.pending is None


OnStep = Callable[[str], None]


class ResearchRunner:
    def __init__(self, deps: Deps, checkpointer=None):
        self._deps = deps
        self.checkpointer = checkpointer or open_checkpointer(deps.settings.checkpoint_db)
        self.graph = build_research_graph(deps, self.checkpointer)

    def _config(self, run_id: str) -> dict:
        return {
            "configurable": {"thread_id": run_id},
            "max_concurrency": self._deps.settings.max_concurrency,
        }

    def _drive(self, graph_input: Any, run_id: str, on_step: OnStep | None) -> RunStatus:
        """Run until the next human checkpoint or the end, reporting each
        finished step (e.g. to show progress in the UI)."""
        for update in self.graph.stream(graph_input, self._config(run_id), stream_mode="updates"):
            for node in update:
                if on_step and not node.startswith("__"):
                    on_step(node)
        return self.status(run_id)

    def start(
        self,
        company: str,
        constraints: ResearchConstraints | None = None,
        website: str | None = None,
        run_id: str | None = None,
        on_step: OnStep | None = None,
    ) -> RunStatus:
        """Raises InvalidInput, before anything is spent, for junk input."""
        profile = CompanyProfile(name=clean_company_name(company), domain=clean_domain(website))
        run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        constraints = constraints or ResearchConstraints(
            news_days=self._deps.settings.news_days,
            max_competitors=self._deps.settings.max_competitors,
        )
        constraints = constraints.model_copy(update={
            "market": clean_market(constraints.market),
            "region": clean_region(constraints.region),
        })
        return self._drive(
            {"run_id": run_id, "company": profile, "constraints": constraints}, run_id, on_step
        )

    def answer(self, run_id: str, answer: dict[str, Any], on_step: OnStep | None = None) -> RunStatus:
        """Resume a run paused at a human checkpoint. Names and websites in the
        answer are validated first; InvalidInput leaves the run paused."""
        interrupts = self.graph.get_state(self._config(run_id)).interrupts
        if not interrupts:
            raise ValueError(f"Run {run_id} isn't waiting for an answer.")
        # Address the answer by interrupt id: LangGraph reads a bare dict as an
        # id-to-answer mapping, so an answer like {} would otherwise resume nothing.
        command = Command(resume={interrupts[0].id: _clean_answer(answer)})
        return self._drive(command, run_id, on_step)

    def continue_run(self, run_id: str, on_step: OnStep | None = None) -> RunStatus:
        """Resume a run that stopped mid-way, from its last checkpoint."""
        return self._drive(None, run_id, on_step)

    def list_runs(self, limit: int = 30) -> list[tuple[str, str]]:
        """Recent runs, newest first, as (run id, company name)."""
        rows = self.checkpointer.conn.execute(
            "SELECT thread_id FROM checkpoints WHERE checkpoint_ns = ''"
            " GROUP BY thread_id ORDER BY MAX(checkpoint_id) DESC LIMIT ?",
            (limit,),
        ).fetchall()
        runs = []
        for (run_id,) in rows:
            company = self.status(run_id).values.get("company")
            runs.append((run_id, company.name if company else "?"))
        return runs

    def status(self, run_id: str) -> RunStatus:
        snapshot = self.graph.get_state(self._config(run_id))
        pending = snapshot.interrupts[0].value if snapshot.interrupts else None
        return RunStatus(run_id, dict(snapshot.values), pending, tuple(snapshot.next))
