"""The fleet: the console's concrete specialist roles.

Each specialist is a :class:`~agentique.role.Role` — a declarative Agent
(prompt, model, tools) plus the kind of artifact its runs produce. The *meaning*
of those kinds — ``plan``, ``findings``, ``change``, ``review`` — is a domain
concept, which is why the fleet lives in the console (application) layer, built on
the generic harness.

All file-touching tools are confined to a shared
:class:`~agentique.tools.workspace.Workspace`, so the fleet reads and writes
against one authorized root with workspace-relative paths. The Builder is the only
do-er that changes the world (write/edit/run); the others are read-only and run
freely (low consequence). Every specialist is told to **act, not ask** — it
produces its artifact directly rather than pausing, so a dispatch resolves in one
shot.
"""

from __future__ import annotations

from agentique import Role
from agentique.core import Agent, Model
from agentique.tools import (
    EditFile,
    ListDir,
    ReadFile,
    RunCommand,
    Workspace,
    WriteFile,
)

PLANNER = "planner"
EXPLORER = "explorer"
BUILDER = "builder"
REVIEWER = "reviewer"

_PLANNER_INSTRUCTIONS = (
    "You are a planning specialist. Given the operator's intent, produce a concise, "
    "actionable, numbered plan for accomplishing it. You may read existing files "
    "(read_file) and list the workspace (list_dir) for context. Do not ask "
    "questions and do not write any files — produce the plan directly as your final "
    "message."
)

_EXPLORER_INSTRUCTIONS = (
    "You are an exploration specialist. Investigate the workspace and the files you "
    "are pointed at (read_file, list_dir) and report concise findings relevant to "
    "the task. You are read-only: never modify anything. Do not ask questions — "
    "report your findings directly as your final message."
)

_BUILDER_INSTRUCTIONS = (
    "You are a builder — the do-er of the fleet. You WRITE the code that fulfills "
    "the task, directly, into the workspace. Act first; never ask questions.\n"
    "- Use write_file and edit_file to create and modify files. Paths are relative "
    "to the workspace root.\n"
    "- Use read_file and list_dir to inspect your work.\n"
    "- Use run_command to self-verify (e.g. confirm a file exists and looks right, "
    "or run a quick check). If verification reveals a problem, fix it and re-check.\n"
    "When the work is done and verified, reply with a short summary of what you "
    "built and the files you created — this becomes the change artifact."
)

_REVIEWER_INSTRUCTIONS = (
    "You are a code reviewer, independent of the builder. Read the files in the "
    "workspace that the builder produced (list_dir, read_file) and review them for "
    "correctness, completeness, and obvious bugs against the task. You are "
    "read-only: never modify anything. Do not ask questions — produce a concise "
    "review (what is good, what is wrong, and whether it meets the task) directly."
)


def build_fleet(model: Model, workspace: Workspace) -> tuple[Role, ...]:
    """Build the four specialist roles over a shared model and workspace.

    Read-only specialists get ``read_file``/``list_dir`` confined to the workspace;
    the Builder additionally gets the world-changing ``write_file``/``edit_file``/
    ``run_command`` (all confined). Returns roles ready to register on a Coordinator.
    """
    read_only = (ReadFile(workspace), ListDir(workspace))
    builder_tools = (
        ReadFile(workspace),
        ListDir(workspace),
        WriteFile(workspace),
        EditFile(workspace),
        RunCommand(workspace),
    )
    return (
        Role(
            name=PLANNER,
            agent=Agent(
                name=PLANNER,
                instructions=_PLANNER_INSTRUCTIONS,
                model=model,
                tools=read_only,
            ),
            kind="plan",
        ),
        Role(
            name=EXPLORER,
            agent=Agent(
                name=EXPLORER,
                instructions=_EXPLORER_INSTRUCTIONS,
                model=model,
                tools=read_only,
            ),
            kind="findings",
        ),
        Role(
            name=BUILDER,
            agent=Agent(
                name=BUILDER,
                instructions=_BUILDER_INSTRUCTIONS,
                model=model,
                tools=builder_tools,
            ),
            kind="change",
        ),
        Role(
            name=REVIEWER,
            agent=Agent(
                name=REVIEWER,
                instructions=_REVIEWER_INSTRUCTIONS,
                model=model,
                tools=read_only,
            ),
            kind="review",
        ),
    )
