# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated Make runs: pretrain a nanochat model from the app.

The request chooses a preset and overrides. Everything that decides *where* the run
happens -- the checkout, the interpreter, the cache, the checkpoint directory -- comes
from the server's configuration, so a request cannot aim a trainer at another Python,
another checkout, or a path outside the project's own base directory.
"""

from __future__ import annotations

from utils.account_context import current_account, run_as
import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from auth.authentication import get_current_subject
from core.make_runs import (
    MAX_DEPTH,
    MAX_DEVICE_BATCH,
    MAX_ITERATIONS,
    MAX_SEQ_LEN,
    MAX_TOTAL_BATCH,
    MIN_DEPTH,
    MIN_DEVICE_BATCH,
    MIN_ITERATIONS,
    MIN_SEQ_LEN,
    MIN_TOTAL_BATCH,
    PRESETS,
    build_argv,
    environment,
    normalize_config,
)
from storage import make_runs_db as db

router = APIRouter()
# Dedicated to the blocking event wait so open streams cannot exhaust the default executor.
_EVENT_WAIT_EXECUTOR = ThreadPoolExecutor(max_workers = 32, thread_name_prefix = "make-events")
# A pretrain emits a line per step, so the run snapshot must not repeat a log chunk.
_DELTA_ONLY_EVENTS = {"run.log", "run.progress", "run.spawned", "run.note"}


class CreateMakeRun(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    preset: str = Field(default = "small", max_length = 32)
    depth: int | None = Field(default = None, ge = MIN_DEPTH, le = MAX_DEPTH)
    iterations: int | None = Field(default = None, ge = MIN_ITERATIONS, le = MAX_ITERATIONS)
    deviceBatchSize: int | None = Field(default = None, ge = MIN_DEVICE_BATCH, le = MAX_DEVICE_BATCH)
    totalBatchSize: int | None = Field(default = None, ge = MIN_TOTAL_BATCH, le = MAX_TOTAL_BATCH)
    aspectRatio: int | None = Field(default = None, ge = 8, le = 256)
    headDim: int | None = Field(default = None, ge = 32, le = 256)
    maxSeqLen: int | None = Field(default = None, ge = MIN_SEQ_LEN, le = MAX_SEQ_LEN)
    runTag: str | None = Field(default = None, max_length = 64)
    timeoutSeconds: int | None = Field(default = None, ge = 60)
    evaluate: bool = True


def _require_run(run_id: str) -> dict:
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code = 404, detail = "Make run not found")
    return run


def _sanitize_config(payload: CreateMakeRun) -> dict[str, Any]:
    # Only the fields the client actually sent are passed on, so normalize_config sees a
    # preset plus overrides rather than a full config of mostly-nulls.
    request = payload.model_dump(exclude_none = True, exclude = {"preset"})
    request["preset"] = payload.preset
    try:
        return normalize_config(request)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc


@router.get("/environment")
def make_environment(current_subject: str = Depends(get_current_subject)):
    """What this installation can pretrain with, so the page can say why it cannot."""
    return environment()


@router.get("/presets")
def make_presets(current_subject: str = Depends(get_current_subject)):
    return {"presets": [{"id": key, **value} for key, value in PRESETS.items()]}


@router.post("", status_code = 202)
def create_make_run(
    payload: CreateMakeRun,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    env = environment()
    if not env["available"]:
        raise HTTPException(status_code = 503, detail = env["error"])
    config = _sanitize_config(payload)
    try:
        run = db.create_run(
            run_id = uuid.uuid4().hex,
            owner_subject = current_subject,
            preset = config["preset"],
            config = config,
            argv = build_argv(config),
        )
    except db.MakeConflictError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    supervisor = getattr(request.app.state, "make_supervisor", None)
    if supervisor is not None:
        supervisor.wake()
    return run


@router.get("")
def list_make_runs(
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
def active_make_runs(current_subject: str = Depends(get_current_subject)):
    return {"runs": db.list_active(current_subject)}


@router.get("/{run_id}")
def get_make_run(run_id: str, current_subject: str = Depends(get_current_subject)):
    return _require_run(run_id)


@router.get("/{run_id}/logs")
def make_run_logs(
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
def cancel_make_run(
    run_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    _require_run(run_id)
    status = db.request_cancel(run_id)
    supervisor = getattr(request.app.state, "make_supervisor", None)
    if supervisor is not None and status == "cancelling":
        supervisor.cancel(run_id)
    return _require_run(run_id)


@router.post("/{run_id}/retry")
def retry_make_run(
    run_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    _require_run(run_id)
    try:
        db.retry(run_id)
    except (db.MakeConflictError, KeyError) as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    supervisor = getattr(request.app.state, "make_supervisor", None)
    if supervisor is not None:
        supervisor.wake()
    return _require_run(run_id)


# POST too: proxies that stream a GET hold it until the response closes.
@router.post("/{run_id}/events")
# Separate registration, out of the schema: one api_route would give both verbs one operationId.
@router.get("/{run_id}/events", include_in_schema = False)
async def make_events(
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
