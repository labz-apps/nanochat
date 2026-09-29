# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated AI Scientist runs: start, watch, cancel, retry.

The launcher keeps its own fail-closed gates, and this route refuses a request that
could not satisfy one rather than spawning a child that would die on the check:

* provider calls are opt-in (``allowProviderCalls``), exactly as in the launcher;
* ``--max-nodes`` is pinned at 1 and is not a client field;
* the lineage handoff arguments are all-or-nothing, as the launcher requires.

Everything a run needs from the machine -- the checkout, the interpreter, the shared
cache, the experiment directory -- is read from the server's configuration, so a
request cannot point a run at another Python, another checkout, or another path.
"""

from __future__ import annotations

from utils.account_context import current_account, run_as
import asyncio
import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from auth.authentication import get_current_subject
from core.scientist_runs import (
    MAX_TIMEOUT_SECONDS,
    MIN_TIMEOUT_SECONDS,
    build_argv,
    environment,
)
from storage import scientist_runs_db as db

router = APIRouter()
# The launcher's own default, restated so the UI and the validator agree on it.
DEFAULT_IDEAS = "ai_scientist/ideas/nanochat_pretraining.json"
DEFAULT_CONFIG = "bfts_config.yaml"
# A relative POSIX path with no traversal and no absolute prefix. Deliberately strict:
# these name files inside the checkout, and the launcher resolves them against its own
# root, so there is no legitimate value outside that shape.
_SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$")
# Dedicated to the blocking event wait so open streams cannot exhaust the default executor.
_EVENT_WAIT_EXECUTOR = ThreadPoolExecutor(max_workers = 32, thread_name_prefix = "scientist-events")
# Events that carry a log chunk: the run snapshot is re-sent with the rest, and a
# multi-megabyte log chunk has no business being repeated in every frame.
_DELTA_ONLY_EVENTS = {"run.log", "run.progress", "run.spawned", "run.note"}


class CreateScientistRun(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    model: str = Field(min_length = 1, max_length = 200)
    ideaIdx: int = Field(default = 0, ge = 0)
    attemptId: int = Field(default = 0, ge = 0)
    loadCode: bool = True
    loadIdeas: str = Field(default = DEFAULT_IDEAS, max_length = 400)
    config: str = Field(default = DEFAULT_CONFIG, max_length = 400)
    allowProviderCalls: bool = False
    timeoutSeconds: int | None = Field(default = None, ge = 1)
    parentJournal: str | None = Field(default = None, max_length = 400)
    parentNodeId: str | None = Field(default = None, max_length = 200)
    parentStage: int | None = Field(default = None, ge = 1, le = 2)
    lineageId: str | None = Field(default = None, max_length = 200)

    @field_validator("model", "loadIdeas", "config", mode = "before")
    @classmethod
    def _reject_non_strings(cls, value: Any) -> Any:
        # bool is not a str, but a dict or a list reaching a str-typed field is a
        # 500 from pydantic's own coercion, not the 400 the rest of this route speaks.
        if value is not None and not isinstance(value, str):
            raise ValueError("must be a string")
        return value

    @field_validator("model")
    @classmethod
    def _model_is_an_identifier(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        # It becomes one argv element, so anything that could be read as a flag or a
        # path is refused rather than quoted around.
        if any(character.isspace() for character in stripped) or stripped.startswith("-"):
            raise ValueError("must be a provider model id")
        return stripped


def _require_run(run_id: str) -> dict:
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code = 404, detail = "AI Scientist run not found")
    return run


def _checkout_file(env: dict[str, Any], value: str, label: str) -> str:
    """A client-named file, resolved inside the checkout and required to exist.

    The launcher resolves a relative path against its own root, so a value that
    escapes it would be read somewhere else entirely. The check here is the boundary.
    """
    candidate = value.strip()
    if not _SAFE_RELATIVE.match(candidate) or ".." in candidate.split("/"):
        raise HTTPException(
            status_code = 400, detail = f"{label} must be a relative path inside the project"
        )
    root = Path(env["projectRoot"]).resolve()
    target = (root / candidate).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(status_code = 400, detail = f"{label} was not found in the project")
    return candidate


def _sanitize_config(payload: CreateScientistRun, env: dict[str, Any]) -> dict[str, Any]:
    # The launcher refuses to make a single provider call without this flag, and refuses
    # anything but --max-nodes 1. Both are gates, not preferences, so a request that
    # would not carry them is rejected here instead of failing inside the child.
    if not payload.allowProviderCalls:
        raise HTTPException(
            status_code = 400,
            detail = "An AI Scientist run makes live provider calls; confirm to allow them",
        )
    handoff = (payload.parentJournal, payload.parentNodeId, payload.parentStage, payload.lineageId)
    if any(value is not None for value in handoff) and not all(
        value is not None for value in handoff
    ):
        raise HTTPException(
            status_code = 400, detail = "Lineage handoff fields must be provided together"
        )
    timeout = payload.timeoutSeconds
    if timeout is not None and not MIN_TIMEOUT_SECONDS <= timeout <= MAX_TIMEOUT_SECONDS:
        raise HTTPException(
            status_code = 400,
            detail = f"timeoutSeconds must be between {MIN_TIMEOUT_SECONDS} and {MAX_TIMEOUT_SECONDS}",
        )
    return {
        "model": payload.model,
        "ideaIdx": payload.ideaIdx,
        "attemptId": payload.attemptId,
        "loadCode": payload.loadCode,
        "loadIdeas": _checkout_file(env, payload.loadIdeas, "loadIdeas"),
        "config": _checkout_file(env, payload.config, "config"),
        "allowProviderCalls": True,
        "timeoutSeconds": timeout,
        "parentJournal": payload.parentJournal,
        "parentNodeId": payload.parentNodeId,
        "parentStage": payload.parentStage,
        "lineageId": payload.lineageId,
    }


@router.get("/environment")
def scientist_environment(current_subject: str = Depends(get_current_subject)):
    """What this installation can run, so the page can explain itself before a run."""
    return environment()


@router.post("", status_code = 202)
def create_scientist_run(
    payload: CreateScientistRun,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    env = environment()
    if not env["available"]:
        raise HTTPException(status_code = 503, detail = env["error"])
    config = _sanitize_config(payload, env)
    try:
        run = db.create_run(
            run_id = uuid.uuid4().hex,
            owner_subject = current_subject,
            config = config,
            argv = build_argv(config, env),
        )
    except db.ScientistConflictError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    supervisor = getattr(request.app.state, "scientist_supervisor", None)
    if supervisor is not None:
        supervisor.wake()
    return run


@router.get("")
def list_scientist_runs(
    limit: int = Query(50, ge = 1, le = 200),
    offset: int = Query(0, ge = 0),
    status: str | None = Query(None),
    current_subject: str = Depends(get_current_subject),
):
    if status is not None and status not in db.ALL_STATUSES:
        raise HTTPException(status_code = 400, detail = f"Unknown status: {status}")
    return {
        "runs": db.list_runs(current_subject, limit = limit, offset = offset, status = status),
        "total": db.count_runs(current_subject),
    }


@router.get("/active")
def active_scientist_runs(current_subject: str = Depends(get_current_subject)):
    return {"runs": db.list_active(current_subject)}


@router.get("/{run_id}")
def get_scientist_run(run_id: str, current_subject: str = Depends(get_current_subject)):
    return _require_run(run_id)


@router.get("/{run_id}/logs")
def scientist_run_logs(
    run_id: str,
    after: int = Query(0, ge = 0),
    current_subject: str = Depends(get_current_subject),
):
    run = _require_run(run_id)
    return {
        "runId": run["id"],
        "text": db.read_log(run["id"], after),
        "truncated": run["logTruncated"],
        "lastEventSeq": run["lastEventSeq"],
    }


@router.post("/{run_id}/cancel")
def cancel_scientist_run(
    run_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    run = _require_run(run_id)
    status = db.request_cancel(run_id)
    supervisor = getattr(request.app.state, "scientist_supervisor", None)
    if supervisor is not None and status == "cancelling":
        supervisor.cancel(run_id)
    return _require_run(run_id)


@router.post("/{run_id}/retry")
def retry_scientist_run(
    run_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    _require_run(run_id)
    try:
        db.retry(run_id)
    except (db.ScientistConflictError, KeyError) as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    supervisor = getattr(request.app.state, "scientist_supervisor", None)
    if supervisor is not None:
        supervisor.wake()
    return _require_run(run_id)


# POST too: proxies that stream /v1/chat/completions still buffer a streamed GET until it closes.
@router.post("/{run_id}/events")
# Separate registration, out of the schema: one api_route would give both verbs one operationId.
@router.get("/{run_id}/events", include_in_schema = False)
async def scientist_events(
    run_id: str,
    request: Request,
    after: int | None = Query(None, ge = 0),
    last_event_id: str | None = Header(None, alias = "Last-Event-ID"),
    current_subject: str = Depends(get_current_subject),
):
    _require_run(run_id)
    header_after = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0
    cursor = max(after or 0, header_after)

    async def stream():
        nonlocal cursor
        loop = asyncio.get_running_loop()
        while True:
            # off the default executor: parked followers there starved the run's own db writes.
            events = await loop.run_in_executor(
                _EVENT_WAIT_EXECUTOR,
                run_as,
                current_account(),
                db.wait_for_events,
                run_id,
                cursor,
                15,
            )
            # Not the wait executor: this read is short, and queueing it behind parked waits
            # would delay every follower once the pool is full.
            snapshot = await asyncio.to_thread(db.get_run, run_id)
            if snapshot is None:
                return
            for event in events:
                cursor = int(event["seq"])
                event_data = dict(event["data"])
                event_data["createdAt"] = event["createdAt"]
                if event["type"] not in _DELTA_ONLY_EVENTS:
                    event_data["run"] = snapshot
                data = json.dumps(event_data, separators = (",", ":"), ensure_ascii = False)
                yield f"id: {cursor}\nevent: {event['type']}\ndata: {data}\n\n"
            if snapshot["status"] in db.TERMINAL_STATUSES and cursor >= int(
                snapshot["lastEventSeq"]
            ):
                return
            if await request.is_disconnected():
                return
            if not events:
                yield ": keep-alive\n\n"

    return StreamingResponse(
        stream(),
        media_type = "text/event-stream",
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
