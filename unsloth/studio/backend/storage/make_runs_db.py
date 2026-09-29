# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Transactional durable state for Make runs (a nanochat pretrain).

Deliberately parallel to ``scientist_runs_db`` rather than shared with it: Make is a
single process with one log and one output directory, where a Scientist run is a tree
search with lineage, curves and a per-attempt artifact contract. The lease, event and
cancel machinery is the same shape, and if a third job type appears it should be
factored out of both rather than copied a third time.
"""

from __future__ import annotations

from core.training.account_jobs import account_is_retired
import json
import sqlite3
import threading
import time
from typing import Any

from storage.studio_db import get_connection as _studio_connection

ACTIVE_STATUSES = frozenset({"queued", "running", "cancelling"})
TERMINAL_STATUSES = frozenset({"cancelled", "completed", "failed"})
ALL_STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES
_CLAIMABLE = ("queued", "running", "cancelling")
_EVENTS_CHANGED = threading.Condition()

# A pretrain prints a line per step, so the log is where the run is watched. Capped so
# an overnight training run cannot grow the studio database without limit.
MAX_LOG_BYTES = 8 * 1024 * 1024
LOG_FLUSH_LINES = 40
LOG_FLUSH_SECONDS = 1.0
# One training run owns the GPU outright; a second would OOM the first or quietly
# oversubscribe the device, so a new run is refused rather than queued behind it.
MAX_ACTIVE_PER_OWNER = 1
MAX_RETRIES = 3


def get_connection():
    if account_is_retired():
        raise RuntimeError("Account is retired")
    return _studio_connection()


class MakeConflictError(RuntimeError):
    pass


def now_ms() -> int:
    return int(time.time() * 1000)


def _loads(value: str | None, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _event_locked(conn: sqlite3.Connection, run_id: str, event_type: str, data: dict) -> int:
    row = conn.execute(
        "SELECT next_event_seq, retry_count FROM make_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise KeyError(run_id)
    seq = int(row["next_event_seq"])
    created = now_ms()
    event_data = dict(data)
    event_data.setdefault("attempt", int(row["retry_count"]))
    conn.execute(
        "INSERT INTO make_events (run_id, seq, event_type, data_json, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, seq, event_type, json.dumps(event_data, ensure_ascii = False), created),
    )
    conn.execute(
        "UPDATE make_runs SET next_event_seq = ?, updated_at = ? WHERE id = ?",
        (seq + 1, created, run_id),
    )
    return seq


def _commit_event(conn: sqlite3.Connection) -> None:
    conn.commit()
    with _EVENTS_CHANGED:
        _EVENTS_CHANGED.notify_all()


def _worker_can_write_locked(
    conn: sqlite3.Connection, run_id: str, worker_id: str, statuses: set[str]
) -> bool:
    row = conn.execute(
        "SELECT status, lease_owner, lease_expires_at, cancel_requested "
        "FROM make_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    return bool(
        row is not None
        and row["lease_owner"] == worker_id
        and row["status"] in statuses
        and not bool(row["cancel_requested"])
        and row["lease_expires_at"] is not None
        and int(row["lease_expires_at"]) >= now_ms()
    )


def append_event(run_id: str, event_type: str, data: dict[str, Any]) -> int:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        seq = _event_locked(conn, run_id, event_type, data)
        _commit_event(conn)
        return seq
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def append_worker_event(
    run_id: str, worker_id: str, event_type: str, data: dict[str, Any]
) -> int | None:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        if not _worker_can_write_locked(conn, run_id, worker_id, {"running"}):
            conn.commit()
            return None
        seq = _event_locked(conn, run_id, event_type, data)
        _commit_event(conn)
        return seq
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def append_log(run_id: str, worker_id: str, text: str) -> int | None:
    """Append a chunk of the trainer's stdout, honouring the per-run log cap.

    The cap is enforced with a read-then-write inside the same IMMEDIATE transaction,
    so two flushes cannot both see room left and jointly overflow it.
    """
    if not text:
        return None
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        if not _worker_can_write_locked(conn, run_id, worker_id, {"running"}):
            conn.commit()
            return None
        used = int(
            conn.execute("SELECT log_bytes FROM make_runs WHERE id = ?", (run_id,)).fetchone()[
                "log_bytes"
            ]
            or 0
        )
        if used >= MAX_LOG_BYTES:
            conn.commit()
            return None
        room = MAX_LOG_BYTES - used
        encoded = len(text.encode("utf-8"))
        truncated = encoded > room
        if truncated:
            text = text.encode("utf-8")[:room].decode("utf-8", errors = "ignore")
        seq = _event_locked(
            conn,
            run_id,
            "run.log",
            {"text": text, "truncated": truncated, "logBytes": used + encoded},
        )
        conn.execute(
            "UPDATE make_runs SET log_bytes = ? WHERE id = ?",
            (min(MAX_LOG_BYTES, used + encoded), run_id),
        )
        _commit_event(conn)
        return seq
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_run(
    *,
    run_id: str,
    owner_subject: str,
    preset: str,
    config: dict[str, Any],
    argv: list[str],
    created_at: int | None = None,
) -> dict:
    created = created_at or now_ms()
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        active = conn.execute(
            "SELECT COUNT(*) FROM make_runs WHERE owner_subject=? AND status IN (?,?,?)",
            (owner_subject, *_CLAIMABLE),
        ).fetchone()[0]
        if int(active) >= MAX_ACTIVE_PER_OWNER:
            raise MakeConflictError(
                "A training run is already in progress; wait for it to finish"
            )
        conn.execute(
            "INSERT INTO make_runs "
            "(id, owner_subject, status, preset, config_json, argv_json, created_at, updated_at) "
            "VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)",
            (
                run_id,
                owner_subject,
                preset,
                json.dumps(config, sort_keys = True, ensure_ascii = False),
                json.dumps(argv, ensure_ascii = False),
                created,
                created,
            ),
        )
        _event_locked(conn, run_id, "run.queued", {"status": "queued"})
        _commit_event(conn)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_run(run_id)


def _row_to_run(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    return {
        "id": data["id"],
        "ownerSubject": data["owner_subject"],
        "status": data["status"],
        "preset": data["preset"],
        "config": _loads(data["config_json"], {}),
        "argv": _loads(data["argv_json"], []),
        "checkpointDir": data["checkpoint_dir"],
        "logBytes": int(data["log_bytes"] or 0),
        "logTruncated": int(data["log_bytes"] or 0) >= MAX_LOG_BYTES,
        "cancelRequested": bool(data["cancel_requested"]),
        "retryCount": data["retry_count"],
        "error": data["error_message"],
        "metrics": _loads(data["metrics_json"], None),
        "createdAt": data["created_at"],
        "updatedAt": data["updated_at"],
        "startedAt": data["started_at"],
        "completedAt": data["completed_at"],
        "heartbeatAt": data["heartbeat_at"],
        "lastEventSeq": int(data["next_event_seq"]) - 1,
    }


def get_run(run_id: str, owner_subject: str | None = None) -> dict | None:
    conn = get_connection()
    try:
        sql = "SELECT * FROM make_runs WHERE id = ?"
        args: tuple = (run_id,)
        if owner_subject is not None:
            sql += " AND owner_subject = ?"
            args += (owner_subject,)
        row = conn.execute(sql, args).fetchone()
        return None if row is None else _row_to_run(row)
    finally:
        conn.close()


def list_runs(
    owner_subject: str | None = None,
    *,
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
) -> list[dict]:
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    sql = "SELECT * FROM make_runs WHERE 1=1"
    args: list[Any] = []
    if owner_subject is not None:
        sql += " AND owner_subject = ?"
        args.append(owner_subject)
    if status is not None:
        if status not in ALL_STATUSES:
            raise ValueError(status)
        sql += " AND status = ?"
        args.append(status)
    sql += " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
    args += [limit, offset]
    conn = get_connection()
    try:
        return [_row_to_run(row) for row in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()


def count_runs(owner_subject: str | None = None) -> int:
    sql = "SELECT COUNT(*) FROM make_runs"
    args: tuple = ()
    if owner_subject is not None:
        sql += " WHERE owner_subject = ?"
        args = (owner_subject,)
    conn = get_connection()
    try:
        return int(conn.execute(sql, args).fetchone()[0])
    finally:
        conn.close()


def list_active(owner_subject: str | None = None) -> list[dict]:
    sql = "SELECT id FROM make_runs WHERE status IN ({})".format(
        ",".join("?" for _ in ACTIVE_STATUSES)
    )
    args: tuple = (*sorted(ACTIVE_STATUSES),)
    if owner_subject is not None:
        sql += " AND owner_subject = ?"
        args += (owner_subject,)
    sql += " ORDER BY created_at"
    conn = get_connection()
    try:
        rows = conn.execute(sql, args).fetchall()
    finally:
        conn.close()
    return [run for row in rows if (run := get_run(row["id"])) is not None]


def set_checkpoint_dir(run_id: str, worker_id: str, checkpoint_dir: str) -> bool:
    conn = get_connection()
    try:
        cur = conn.execute(
            "UPDATE make_runs SET checkpoint_dir=?, updated_at=? WHERE id=? AND lease_owner=?",
            (checkpoint_dir, now_ms(), run_id, worker_id),
        )
        conn.commit()
        return cur.rowcount == 1
    finally:
        conn.close()


def request_cancel(run_id: str) -> str:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status FROM make_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        status = row["status"]
        if status in TERMINAL_STATUSES or status == "cancelling":
            conn.commit()
            return status
        # A queued run has no child yet, so it goes straight to cancelled; a running one
        # has to be reaped by the worker first.
        new_status = "cancelled" if status == "queued" else "cancelling"
        completed = now_ms() if new_status == "cancelled" else None
        conn.execute(
            "UPDATE make_runs SET cancel_requested = 1, status = ?, completed_at = ?, "
            "updated_at = ? WHERE id = ?",
            (new_status, completed, now_ms(), run_id),
        )
        _event_locked(conn, run_id, f"run.{new_status}", {"status": new_status})
        _commit_event(conn)
        return new_status
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def retry(run_id: str, max_retries: int = MAX_RETRIES) -> str:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status FROM make_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        if row["status"] not in {"failed", "cancelled"}:
            raise MakeConflictError("Only failed or cancelled runs can be retried")
        # Counted against the event log rather than the row: retry_count also ticks when
        # a lease expires and the run is re-claimed, so a run that took Studio down three
        # times would otherwise spend its own budget.
        spent = conn.execute(
            "SELECT COUNT(*) FROM make_events WHERE run_id=? AND event_type='run.retried'",
            (run_id,),
        ).fetchone()[0]
        if int(spent) >= max_retries:
            raise MakeConflictError("Retry budget exhausted")
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        active = conn.execute(
            f"SELECT id FROM make_runs WHERE id<>? AND status IN ({placeholders}) LIMIT 1",
            (run_id, *sorted(ACTIVE_STATUSES)),
        ).fetchone()
        if active is not None:
            raise MakeConflictError("Another training run is already in progress")
        now = now_ms()
        conn.execute(
            "UPDATE make_runs SET status='queued', cancel_requested=0, "
            "retry_count = retry_count + 1, error_message=NULL, metrics_json=NULL, "
            "checkpoint_dir=NULL, completed_at=NULL, lease_owner=NULL, "
            "lease_expires_at=NULL, updated_at=? WHERE id=?",
            (now, run_id),
        )
        _event_locked(conn, run_id, "run.retried", {"status": "queued"})
        _commit_event(conn)
        return "queued"
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


_CLAIMABLE_SQL = """SELECT id FROM make_runs
                    WHERE status IN ('queued','running','cancelling')
                      AND (lease_owner IS NULL OR lease_expires_at < ?)
                    ORDER BY created_at LIMIT 1"""


def _has_claimable(now: int) -> bool:
    """Read-only probe, taking no write lock. The supervisor polls twice a second and
    almost every poll finds nothing, so opening BEGIN IMMEDIATE first would hold the
    writer lock 2x/second while idle.
    """
    conn = get_connection()
    try:
        return conn.execute(_CLAIMABLE_SQL, (now,)).fetchone() is not None
    finally:
        conn.close()


def claim_next(worker_id: str, lease_ms: int = 120_000) -> dict | None:
    if not _has_claimable(now_ms()):
        return None
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        now = now_ms()
        row = conn.execute(
            "SELECT * FROM make_runs "
            "WHERE status IN ('queued','running','cancelling') "
            "AND (lease_owner IS NULL OR lease_expires_at < ?) "
            "ORDER BY created_at LIMIT 1",
            (now,),
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        status = row["status"]
        next_status = "cancelling" if status == "cancelling" else "running"
        # A re-claim of a running row means the previous worker died. Its child is gone,
        # so its log and its partial checkpoint belong to an attempt nobody is watching.
        resumed = status == "running"
        if resumed:
            conn.execute(
                "DELETE FROM make_events WHERE run_id=? AND event_type='run.log'",
                (row["id"],),
            )
            conn.execute(
                "UPDATE make_runs SET log_bytes=0, checkpoint_dir=NULL WHERE id=?",
                (row["id"],),
            )
        conn.execute(
            "UPDATE make_runs SET status=?, lease_owner=?, lease_expires_at=?, "
            "heartbeat_at=?, started_at=COALESCE(started_at, ?), updated_at=? WHERE id=?",
            (next_status, worker_id, now + lease_ms, now, now, now, row["id"]),
        )
        _event_locked(
            conn, row["id"], "run.started", {"status": next_status, "resumed": resumed}
        )
        _commit_event(conn)
        claimed = get_run(row["id"])
        if claimed is not None:
            claimed["claimedFromStatus"] = status
        return claimed
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def heartbeat(run_id: str, worker_id: str, lease_ms: int = 120_000) -> bool:
    conn = get_connection()
    try:
        now = now_ms()
        cur = conn.execute(
            "UPDATE make_runs SET heartbeat_at=?, lease_expires_at=? "
            "WHERE id=? AND lease_owner=? AND lease_expires_at>=?",
            (now, now + lease_ms, run_id, worker_id, now),
        )
        conn.commit()
        return cur.rowcount == 1
    finally:
        conn.close()


def is_cancel_requested(run_id: str) -> bool:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT cancel_requested FROM make_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return row is None or bool(row[0])
    finally:
        conn.close()


def finish(
    run_id: str,
    worker_id: str,
    status: str,
    error: str | None = None,
    metrics: dict[str, Any] | None = None,
    allow_expired: bool = False,
) -> str | None:
    if status not in TERMINAL_STATUSES:
        raise ValueError(status)
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        now = now_ms()
        row = conn.execute(
            "SELECT status, cancel_requested, lease_expires_at FROM make_runs "
            "WHERE id=? AND lease_owner=?",
            (run_id, worker_id),
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        if (
            not allow_expired
            and not bool(row["cancel_requested"])
            and (row["lease_expires_at"] is None or int(row["lease_expires_at"]) < now)
        ):
            conn.commit()
            return None
        actual = (
            "cancelled" if bool(row["cancel_requested"]) or row["status"] == "cancelling" else status
        )
        if actual == "cancelled":
            # A stop is the user asking for nothing back: no error, and no half-trained
            # checkpoint presented as if it were a finished model.
            metrics = None
            error = None
        payload: dict[str, Any] = {"status": actual}
        if actual != "cancelled":
            payload["error"] = error
            if metrics is not None:
                payload["metrics"] = metrics
        conn.execute(
            "UPDATE make_runs SET status=?, error_message=?, metrics_json=?, "
            "completed_at=?, updated_at=?, lease_owner=NULL, lease_expires_at=NULL WHERE id=?",
            (
                actual,
                None if actual == "cancelled" else error,
                json.dumps(metrics, allow_nan = False) if metrics is not None else None,
                now,
                now,
                run_id,
            ),
        )
        _event_locked(conn, run_id, f"run.{actual}", payload)
        _commit_event(conn)
        return actual
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_events(run_id: str, after: int = 0, limit: int = 1000) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT seq, event_type, data_json, created_at FROM make_events "
            "WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?",
            (run_id, after, limit),
        ).fetchall()
        return [
            {
                "seq": r["seq"],
                "type": r["event_type"],
                "data": _loads(r["data_json"], {}),
                "createdAt": r["created_at"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def read_log(run_id: str, after: int = 0, limit: int = 5000) -> str:
    """The run's streamed stdout, reassembled from its ``run.log`` events."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT data_json FROM make_events "
            "WHERE run_id=? AND seq>? AND event_type='run.log' ORDER BY seq LIMIT ?",
            (run_id, after, limit),
        ).fetchall()
    finally:
        conn.close()
    return "".join(_loads(r["data_json"], {}).get("text", "") for r in rows)


def wait_for_events(run_id: str, after: int = 0, timeout: float = 15) -> list[dict]:
    """Block until committed events are available or the keep-alive timeout expires."""
    events = list_events(run_id, after)
    if events:
        return events
    with _EVENTS_CHANGED:
        # Recheck under the condition lock so a commit cannot be missed between the
        # first read and the wait.
        events = list_events(run_id, after)
        if events:
            return events
        _EVENTS_CHANGED.wait(timeout)
    return list_events(run_id, after)


def recover_expired(now: int | None = None) -> int:
    conn = get_connection()
    try:
        now = now or now_ms()
        cur = conn.execute(
            "UPDATE make_runs SET lease_owner=NULL, lease_expires_at=NULL, updated_at=? "
            "WHERE status IN ('queued','running','cancelling') "
            "AND lease_owner IS NOT NULL AND lease_expires_at < ?",
            (now, now),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def owns_lease(run_id: str, worker_id: str) -> bool:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM make_runs WHERE id=? AND lease_owner=? AND lease_expires_at>=?",
            (run_id, worker_id, now_ms()),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def release_worker_leases(worker_id: str) -> int:
    conn = get_connection()
    try:
        cur = conn.execute(
            "UPDATE make_runs SET lease_owner=NULL, lease_expires_at=NULL, updated_at=? "
            "WHERE lease_owner=? AND status IN ('queued','running','cancelling')",
            (now_ms(), worker_id),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()
