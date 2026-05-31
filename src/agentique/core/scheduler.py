"""The Scheduler: the core hub that registers agents and dispatches between them.

The :class:`~agentique.core.runtime.Engine` drives *one* agent; the Scheduler is the
substrate that coordinates *many*. It registers agents by id (addressing-by-id is
load-bearing — it is what lets a run be resumed in a fresh process by re-registering
the same agents), dispatches request/response between them over an internal bus,
tracks a lightweight :class:`Run` per dispatch (state + Paused snapshot + parent_id),
records run-lineage (who spawned whom), owns the single event stream, and routes
resume-by-id to the right run. It uses an Engine to advance any individual agent.

The bus is **synchronous and deterministic**: ``dispatch`` is a direct ``await`` on
the same task — no queue, no background task, no ``gather`` — so a StubModel-driven
run, including nested dispatch, is fully reproducible. The public surface is kept
tiny (``register``/``dispatch``/``resume``); pub/sub, mailboxes, concurrent fan-out,
and cross-process actors are headroom the bus can grow into without changing it.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from functools import partial
from typing import Literal

from pydantic.dataclasses import dataclass

from agentique.core.agent import Agent
from agentique.core.events import Dispatch, EventSink, NullSink
from agentique.core.middleware import TracingMiddleware
from agentique.core.result import Blocked, Completed, NeedsHuman, Paused, Result
from agentique.core.run_context import RunContext
from agentique.core.runtime import Engine

type RunState = Literal["running", "paused", "done", "blocked"]
"""A run's lifecycle: driving, parked for human input, finished, or stopped."""


@dataclass(frozen=True, slots=True)
class Run:
    """The Scheduler's lightweight record of one dispatch.

    A serializable value (frozen Pydantic): ``id`` and ``agent_id`` address it,
    ``state`` is its lifecycle, ``paused`` is the resumable snapshot when parked,
    and ``parent_id`` is the run that spawned it (``None`` for a top-level dispatch)
    — the edge that makes the run graph.
    """

    id: str
    agent_id: str
    state: RunState
    paused: Paused | None = None
    parent_id: str | None = None


class Scheduler:
    """Registers agents by id and dispatches runs between them over a sync bus."""

    def __init__(
        self, *, engine: Engine | None = None, sink: EventSink | None = None
    ) -> None:
        self._sink: EventSink = sink if sink is not None else NullSink()
        base = engine if engine is not None else Engine()
        # The Scheduler owns the event stream: wrap the engine with tracing bound to
        # this sink so model/tool/turn events join the dispatch events on one tap.
        self._engine = dc_replace(
            base, middleware=(TracingMiddleware(self._sink), *base.middleware)
        )
        self._agents: dict[str, Agent] = {}
        self._runs: dict[str, Run] = {}
        self._run_seq = 0

    @property
    def sink(self) -> EventSink:
        return self._sink

    def register(self, agent_id: str, agent: Agent) -> None:
        """Register ``agent`` under ``agent_id`` so runs can address it (including
        across a process restart, by re-registering under the same id)."""
        self._agents[agent_id] = agent

    def restore(self, run: Run) -> None:
        """Re-seat a paused ``run`` loaded from durable storage into a fresh
        Scheduler, so ``resume(run.id, …)`` can continue it. The matching agent must
        be (re-)registered under ``run.agent_id`` first. Keeps the run-id counter
        ahead of restored ids so a new dispatch cannot collide with one."""
        self._runs[run.id] = run
        if run.id.startswith("r") and run.id[1:].isdigit():
            self._run_seq = max(self._run_seq, int(run.id[1:]))

    def run(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def runs(self) -> tuple[Run, ...]:
        return tuple(self._runs.values())

    async def dispatch(self, agent_id: str, prompt: str) -> Result:
        """Dispatch a top-level run of ``agent_id`` on ``prompt`` to a Result."""
        return await self._dispatch(agent_id, prompt, parent_id=None)

    async def resume(self, run_id: str, answer: str) -> Result:
        """Resume the paused run ``run_id``, folding the human's ``answer`` in."""
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(f"no run {run_id!r}")
        if run.state != "paused" or run.paused is None:
            raise ValueError(f"run {run_id!r} is not paused")
        agent = self._agents.get(run.agent_id)
        if agent is None:
            raise KeyError(f"no agent {run.agent_id!r}")
        result = await self._engine.resume(
            agent, run.paused, answer, ctx=self._context_for(run_id)
        )
        self._record(run_id, run.agent_id, run.parent_id, result)
        return result

    async def _dispatch(
        self, agent_id: str, prompt: str, *, parent_id: str | None
    ) -> Result:
        agent = self._agents.get(agent_id)
        if agent is None:
            raise KeyError(f"no agent {agent_id!r}")
        self._run_seq += 1
        run_id = f"r{self._run_seq}"
        self._sink.emit(Dispatch(run_id=run_id, parent_id=parent_id, agent_id=agent_id))
        # Record the run as running before driving it, so lineage exists even while
        # a child is dispatched mid-run.
        self._runs[run_id] = Run(
            id=run_id, agent_id=agent_id, state="running", parent_id=parent_id
        )
        result = await self._engine.run(agent, prompt, ctx=self._context_for(run_id))
        self._record(run_id, agent_id, parent_id, result)
        return result

    def _context_for(self, run_id: str) -> RunContext:
        # The dispatch handle is bound to this run, so a child it spawns records
        # this run as its parent.
        return RunContext(
            run_id=run_id,
            emit=self._sink,
            dispatch=partial(self._dispatch, parent_id=run_id),
        )

    def _record(
        self, run_id: str, agent_id: str, parent_id: str | None, result: Result
    ) -> Run:
        match result:
            case Completed():
                run = Run(
                    id=run_id, agent_id=agent_id, state="done", parent_id=parent_id
                )
            case NeedsHuman(paused=paused):
                run = Run(
                    id=run_id,
                    agent_id=agent_id,
                    state="paused",
                    paused=paused,
                    parent_id=parent_id,
                )
            case Blocked():
                run = Run(
                    id=run_id,
                    agent_id=agent_id,
                    state="blocked",
                    parent_id=parent_id,
                )
        self._runs[run_id] = run
        return run
