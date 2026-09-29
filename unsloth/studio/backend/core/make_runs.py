# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Make: pretrain a nanochat model from the app, as a shortened speedrun.

``runs/speedrun.sh`` is the reference pipeline -- download shards, train the tokenizer,
pretrain, evaluate -- but it is written for a blank 8xH100 node and cannot be run as-is
on the machine that has Studio installed. Two things make it unusable verbatim here, and
both are corrected below rather than left for the user to discover as a crash:

* ``--fp8`` needs H100 tensor cores. On a consumer card it aborts the run.
* ``torchrun --nproc_per_node=8`` and its device-batch size assume eight devices.

So this builds the same *steps* as the speedrun, in the same order, with single-device
settings and an iteration budget the user picks. It is the pipeline, not the script: a
shell script cannot be driven, watched and cancelled from a UI, and its ``uv sync`` step
would rewrite the interpreter out from under a run already in flight.

Every knob is a request; the interpreter, the checkout, the cache and the checkpoint
directory are resolved from the server's configuration, so a run cannot be pointed at
another Python, another checkout or another path.
"""

from __future__ import annotations

from core.training.account_jobs import account_key, job_accounts
from loggers import get_logger
from storage.studio_db import is_sqlite_busy_error
from utils.account_context import arun_as, run_as
import asyncio
import json
import os
import queue
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import weakref
from pathlib import Path
from typing import Any

from storage import make_runs_db as db

logger = get_logger(__name__)
_supervisors = weakref.WeakSet()

LAUNCHER_MODULE = "scripts.base_train"
EVAL_MODULE = "scripts.base_eval"
RESULTS_FILENAME = "results.json"
# What nanochat's own tokenizer_hash() requires. Without these the results contract's
# provenance is silently empty, so an empty cache is not a usable cache.
_TOKENIZER_FILES = ("tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt")
_REQUIRED_CACHE_DIRS = ("base_data_climbmix", "eval_bundle")
_REQUIRED_CURRICULUM = "curriculum_data/karpathy--climbmix-400b-shuffle/default"

# The presets. `iterations` is the "shorter" knob the speedrun does not expose: with
# --target-param-data-ratio the iteration count is derived from a FLOPs budget, and on
# one consumer GPU that budget is hours. Pinning it is what makes this a Make run.
#
# deviceBatchSize is the per-step micro-batch, and totalBatchSize must stay a multiple
# of it. These are the values runs/curriculum_4060ti.sh uses for a 16GB 4060 Ti, which
# is the card this ships against.
PRESETS: dict[str, dict[str, Any]] = {
    "tiny": {
        "label": "Tiny",
        "depth": 4,
        "iterations": 120,
        "deviceBatchSize": 8,
        "totalBatchSize": 32768,
        "aspectRatio": 64,
        "headDim": 64,
        "maxSeqLen": 1024,
    },
    "small": {
        "label": "Small",
        "depth": 6,
        "iterations": 300,
        "deviceBatchSize": 16,
        "totalBatchSize": 65536,
        "aspectRatio": 64,
        "headDim": 128,
        "maxSeqLen": 2048,
    },
    "medium": {
        "label": "Medium",
        "depth": 8,
        "iterations": 800,
        "deviceBatchSize": 16,
        "totalBatchSize": 65536,
        "aspectRatio": 64,
        "headDim": 128,
        "maxSeqLen": 2048,
    },
}
DEFAULT_PRESET = "small"

# Bounds. A depth past ~12 stops fitting a 16GB card at 2K context, and an iteration
# count past a few thousand is not a "make something" run any more.
MIN_DEPTH, MAX_DEPTH = 2, 12
MIN_ITERATIONS, MAX_ITERATIONS = 10, 20_000
# Two different quantities, so two different ranges. deviceBatchSize is sequences per
# micro-step; totalBatchSize is TOKENS per optimiser step and is orders of magnitude
# larger, so sharing one bound would cap a real run at 256 tokens.
MIN_DEVICE_BATCH, MAX_DEVICE_BATCH = 1, 256
MIN_TOTAL_BATCH, MAX_TOTAL_BATCH = 1024, 4 * 1024 * 1024
MIN_SEQ_LEN, MAX_SEQ_LEN = 256, 8192
# A wall-clock cap. Pretraining that outlives it has stopped being a Make run, and the
# run is stopped rather than left occupying the GPU overnight.
MIN_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 24 * 3600

TERMINATE_GRACE_SECONDS = 15.0
TICK_SECONDS = 0.25
CHECK_INTERVAL_SECONDS = 2.0
LEASE_MS = 120_000

_BASE_ENV = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "HOME",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "TEMP",
    "TMP",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "TZ",
)
_OPTIONAL_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTHONHASHSEED",
)
_PROJECT_ROOT_ENV = "UNSLOTH_SCIENTIST_PROJECT_ROOT"
_PYTHON_ENV = "UNSLOTH_SCIENTIST_PYTHON"
_SHARED_CACHE_ENV = "UNSLOTH_SCIENTIST_SHARED_CACHE"
_EXPERIMENT_ROOT_ENV = "UNSLOTH_SCIENTIST_EXPERIMENT_ROOT"
_REPO_ROOT_PARENTS = 4

# What the run needs before it can be trusted to produce a real model. Kept separate from
# the Scientist's list: this pipeline does not use the curriculum corpus.
_CACHE_PROBLEMS = (
    (("tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt"), "tokenizer"),
    (("base_data_climbmix",), "ClimbMix shards"),
    (("eval_bundle",), "the evaluation bundle"),
)


class RunCancelled(Exception):
    pass


class LeaseLost(Exception):
    pass


def _candidate_project_roots() -> list[Path]:
    roots: list[Path] = []
    configured = os.environ.get(_PROJECT_ROOT_ENV, "").strip()
    if configured:
        roots.append(Path(configured).expanduser())
    here = Path(__file__).resolve()
    if len(here.parents) > _REPO_ROOT_PARENTS:
        roots.append(here.parents[_REPO_ROOT_PARENTS])
    roots.extend(here.parents)
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def find_project_root() -> Path | None:
    for root in _candidate_project_roots():
        if (root / "launch_scientist_bfts.py").is_file() and (root / "nanochat").is_dir():
            return root
    return None


def _resolve_python(root: Path) -> str | None:
    """The interpreter that can import nanochat.

    The checkout's own virtualenv first: this pipeline is nanochat's, and Studio's
    interpreter has torch but not nanochat.
    """
    configured = os.environ.get(_PYTHON_ENV, "").strip()
    if configured:
        return configured
    for candidate in (root / ".venv" / "bin" / "python", root / ".venv" / "Scripts" / "python.exe"):
        if candidate.is_file():
            return str(candidate)
    return None


def _resolve_shared_cache(root: Path) -> Path | None:
    for name in (_SHARED_CACHE_ENV, "NANOCHAT_SHARED_CACHE", "NANOCHAT_BASE_DIR"):
        configured = os.environ.get(name, "").strip()
        if configured:
            candidate = Path(configured).expanduser()
            if candidate.is_dir():
                return candidate
    for candidate in (Path.home() / ".cache" / "nanochat", root / "data", root / "cache"):
        if candidate.is_dir():
            return candidate
    return None


def _resolve_checkpoint_root() -> Path | None:
    """Where base_train writes its checkpoints: <base_dir>/base_checkpoints."""
    for name in (_SHARED_CACHE_ENV, "NANOCHAT_SHARED_CACHE", "NANOCHAT_BASE_DIR"):
        configured = os.environ.get(name, "").strip()
        if configured:
            return Path(configured).expanduser() / "base_checkpoints"
    return Path.home() / ".cache" / "nanochat" / "base_checkpoints"


def interpreter_can_import(root: Path, python: str) -> bool:
    import subprocess

    try:
        completed = subprocess.run(
            [python, "-c", "import nanochat.common, scripts.base_train"],
            cwd = str(root),
            capture_output = True,
            timeout = 180,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def cache_missing_assets(cache: Path) -> list[str]:
    """Which of the pipeline's inputs are absent, in the order they are needed."""
    missing: list[str] = []
    for files, label in _CACHE_PROBLEMS:
        if all((cache / name).is_file() for name in files):
            continue
        if all((cache / name).is_dir() for name in files):
            continue
        missing.append(label)
    return missing


def _problem(code: str, message: str, fix: str) -> dict[str, str]:
    """One thing standing between this install and a run that can start."""
    return {"code": code, "message": message, "fix": fix}


def environment() -> dict[str, Any]:
    """What this installation can actually pretrain with.

    Pessimistic on purpose: every check here is one whose failure would otherwise surface
    as a trainer crash minutes into a run.
    """
    root = find_project_root()
    if root is None:
        return {
            "available": False,
            "error": "Could not find a nanochat checkout.",
            "problems": [
                _problem(
                    "no-project-root",
                    "Could not find a nanochat checkout.",
                    f"Set {_PROJECT_ROOT_ENV} to the nanochat checkout.",
                )
            ],
            "projectRoot": None,
            "python": None,
            "sharedCache": None,
            "checkpointRoot": None,
            "presets": [{"id": key, **value} for key, value in PRESETS.items()],
        }
    python = _resolve_python(root)
    cache = _resolve_shared_cache(root)
    problems: list[dict[str, str]] = []
    if python is None:
        problems.append(
            _problem(
                "missing-python",
                "No nanochat interpreter found.",
                f"Run `uv sync --extra gpu --group dev` in {root}, or set {_PYTHON_ENV}.",
            )
        )
    elif not interpreter_can_import(root, python):
        problems.append(
            _problem(
                "python-cannot-import",
                f"{python} cannot import nanochat from {root}.",
                "Install the project dependencies into that interpreter, or point "
                f"{_PYTHON_ENV} at one that has them.",
            )
        )
    if cache is None:
        problems.append(
            _problem(
                "no-shared-cache",
                "No nanochat shared cache found.",
                "Run `python -m nanochat.dataset -n 8` and `python -m scripts.tok_train` "
                "in the checkout, or set "
                f"{_SHARED_CACHE_ENV} to a populated cache.",
            )
        )
    else:
        missing = cache_missing_assets(cache)
        if missing:
            problems.append(
                _problem(
                    "incomplete-shared-cache",
                    f"{cache} is missing: {', '.join(missing)}.",
                    "Populate the shared cache. The tokenizer needs "
                    "`python -m scripts.tok_train`; the shards need "
                    "`python -m nanochat.dataset -n 8`.",
                )
            )
    return {
        "available": not problems,
        "error": "; ".join(problem["message"] for problem in problems) or None,
        "problems": problems,
        "projectRoot": str(root),
        "python": python,
        "sharedCache": str(cache) if cache is not None else None,
        "checkpointRoot": str(_resolve_checkpoint_root()),
        "presets": [{"id": key, **value} for key, value in PRESETS.items()],
    }


def normalize_config(payload: dict[str, Any]) -> dict[str, Any]:
    """A preset with the request's overrides folded in, and everything range-checked.

    Refuses rather than clamps. A silently clamped batch size is a run that trains
    something other than what the dialog said, and the only sign is a loss curve that
    does not match the last one.
    """
    preset_id = str(payload.get("preset") or DEFAULT_PRESET)
    if preset_id not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_id}")
    preset = PRESETS[preset_id]

    def pick(key: str, minimum: int, maximum: int) -> int:
        raw = payload.get(key, preset[key])
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{key} must be a number")
        value = int(raw)
        if not minimum <= value <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}")
        return value

    device_batch = pick("deviceBatchSize", MIN_DEVICE_BATCH, MAX_DEVICE_BATCH)
    total_batch = pick("totalBatchSize", MIN_TOTAL_BATCH, MAX_TOTAL_BATCH)
    if total_batch % device_batch:
        # nanochat's grad accumulation is total/device, and a non-integer factor either
        # errors inside the trainer or silently changes the effective batch.
        raise ValueError("totalBatchSize must be a multiple of deviceBatchSize")
    head_dim = pick("headDim", 32, 256)
    seq_len = pick("maxSeqLen", MIN_SEQ_LEN, MAX_SEQ_LEN)
    if seq_len % 128:
        raise ValueError("maxSeqLen must be a multiple of 128")
    timeout = payload.get("timeoutSeconds")
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, int):
            raise ValueError("timeoutSeconds must be an integer")
        if not MIN_TIMEOUT_SECONDS <= timeout <= MAX_TIMEOUT_SECONDS:
            raise ValueError(
                f"timeoutSeconds must be between {MIN_TIMEOUT_SECONDS} and {MAX_TIMEOUT_SECONDS}"
            )
    return {
        "preset": preset_id,
        "depth": pick("depth", MIN_DEPTH, MAX_DEPTH),
        "iterations": pick("iterations", MIN_ITERATIONS, MAX_ITERATIONS),
        "deviceBatchSize": device_batch,
        "totalBatchSize": total_batch,
        "aspectRatio": pick("aspectRatio", 8, 256),
        "headDim": head_dim,
        "maxSeqLen": seq_len,
        "runTag": _run_tag(payload.get("runTag")),
        "timeoutSeconds": timeout,
        "evaluate": bool(payload.get("evaluate", True)),
    }


def _run_tag(raw: Any) -> str:
    """A checkpoint directory name. Restricted to what a path segment can hold."""
    value = "make" if raw is None else str(raw).strip()
    if not value:
        return "make"
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    cleaned = cleaned.strip("-") or "make"
    return cleaned[:64]


def build_argv(config: dict[str, Any]) -> list[str]:
    """The exact pretraining command, stored on the run and shown in the UI.

    Two things are deliberately absent, and both are what make speedrun.sh unusable on
    this machine: ``--fp8`` (H100 tensor cores only) and ``torchrun --nproc_per_node=8``
    (a single device needs neither). ``--num-iterations`` is pinned, which is the whole
    point: left at -1 the iteration count is derived from a FLOPs budget meant for eight
    H100s.
    """
    argv = [
        "-m",
        LAUNCHER_MODULE,
        "--",
        f"--depth={config['depth']}",
        f"--aspect-ratio={config['aspectRatio']}",
        f"--head-dim={config['headDim']}",
        f"--max-seq-len={config['maxSeqLen']}",
        f"--num-iterations={config['iterations']}",
        f"--device-batch-size={config['deviceBatchSize']}",
        f"--total-batch-size={config['totalBatchSize']}",
        f"--model-tag={config['runTag']}",
        "--run=dummy",
    ]
    # Evaluation is a separate process in the speedrun. It is optional here because it
    # is a meaningful share of a short run's wall clock and the user may only want weights.
    if config.get("evaluate"):
        argv.append("--results-json=results.json")
    return argv


def build_child_env(env: dict[str, Any]) -> dict[str, str]:
    """A fresh environment for the trainer: an allowlist, never a copy of ours.

    No credential crosses this boundary. Pretraining does not call a provider, so a key
    that reached the child would be a key the run had no reason to hold.
    """
    child: dict[str, str] = {}
    for name in _BASE_ENV:
        value = os.environ.get(name)
        if value:
            child[name] = value
    for name in _OPTIONAL_ENV:
        value = os.environ.get(name)
        if value:
            child[name] = value
    child.update(
        PYTHONUNBUFFERED = "1",
        PYTHONIOENCODING = "utf-8",
        # nanochat's own training log and checkpoints live under the base dir; without
        # this the trainer would resolve a different cache than the one we validated.
        NANOCHAT_BASE_DIR = str(env["sharedCache"]),
        NANOCHAT_SHARED_CACHE = str(env["sharedCache"]),
    )
    return child


def _read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding = "utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def collect_results(checkpoint_dir: Path | None) -> dict[str, Any] | None:
    """The metrics the trainer wrote, or None if it wrote none.

    base_train --results-json emits the same canonical contract the AI Scientist reads,
    so a Make run and a Scientist run are directly comparable.
    """
    if checkpoint_dir is None:
        return None
    for candidate in (checkpoint_dir / RESULTS_FILENAME, checkpoint_dir.parent / RESULTS_FILENAME):
        payload = _read_json(candidate)
        if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
            metrics = {
                key: value
                for key, value in payload["metrics"].items()
                if isinstance(value, (int, float, str, bool)) or value is None
            }
            # NaN is a float and JSON.parse rejects it, so a terminal write must not
            # store one.
            return {key: value for key, value in metrics.items() if not isinstance(value, float) or value == value}
    return None


def _progress_from_log(lines: list[str]) -> dict[str, Any] | None:
    """The step and loss from the trainer's own progress line, when it has printed one.

    Parsed rather than asked for: the trainer's stdout is the only progress channel that
    survives a Studio restart, and a run watched through a reconnect still needs to show
    where it got to.
    """
    for line in reversed(lines[-40:]):
        stripped = line.strip()
        if not stripped.startswith("step "):
            continue
        progress: dict[str, Any] = {}
        # Every token, not every token but the last: the loss is written last, and
        # skipping it drops exactly the number anyone is watching.
        for token in stripped.split()[1:]:
            if "=" not in token:
                continue
            key, _, value = token.partition("=")
            try:
                progress[key] = int(value) if value.isdigit() else float(value)
            except ValueError:
                progress[key] = value
        if progress:
            return progress
    return None


def _spawn(python: str, argv: list[str], cwd: str, env: dict[str, str]) -> subprocess.Popen:
    from utils.process_lifetime import adopt_pid, child_popen_kwargs, spawn_on_lifetime_thread

    def _popen() -> subprocess.Popen:
        return subprocess.Popen(
            [python, *argv],
            cwd = cwd,
            env = env,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            stdin = subprocess.DEVNULL,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            bufsize = 1,
            **child_popen_kwargs(),
        )

    process = spawn_on_lifetime_thread(_popen)
    adopt_pid(process.pid)
    return process


def _terminate(process: subprocess.Popen) -> None:
    """Stop the trainer and the dataloader workers under it, then let go of the record."""
    from utils.process_lifetime import terminate_pid

    pid = process.pid
    try:
        process.terminate()
    except Exception:  # noqa: BLE001 - already gone, or the escalation will settle it
        pass

    def escalate() -> None:
        time.sleep(TERMINATE_GRACE_SECONDS)
        try:
            if process.poll() is None:
                process.kill()
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target = escalate, daemon = True).start()
    # The trainer is the run, but it owns dataloader workers, so the leader alone is not
    # the whole thing. terminate_pid takes the validated tree, then drops the record.
    threading.Thread(
        target = terminate_pid, args = (pid,), kwargs = {"owner_verified": True}, daemon = True
    ).start()


class MakeSupervisor:
    def __init__(self, app: Any, poll_seconds: float = 0.5) -> None:
        _supervisors.add(self)
        self.job_account = None
        self.app = app
        self.poll_seconds = poll_seconds
        self.worker_id = uuid.uuid4().hex
        self._stopping = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._cancel_events: dict[str, threading.Event] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._lost_leases: set[str] = set()
        self._last_claim_account: str | None = None

    def start(self) -> None:
        for account in job_accounts():
            run_as(account, db.recover_expired)
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name = "make-supervisor")

    async def stop(self) -> None:
        self._stopping.set()
        try:
            for cancel_event in self._cancel_events.values():
                cancel_event.set()
            for process in list(self._processes.values()):
                _terminate(process)
            if self._task is not None:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    # Polling is intentionally sufficient for one local process; requests
                    # never own tasks.
                    pass
        finally:
            for account in job_accounts():
                await asyncio.to_thread(run_as, account, db.release_worker_leases, self.worker_id)

    def wake(self) -> None:
        # Polling is the admission path, as for the other supervisors: one long-lived
        # task per process, and a wake flag would be a second thing to keep correct.
        pass

    def cancel(self, run_id: str) -> None:
        self._cancel_events.setdefault(account_key(run_id), threading.Event()).set()
        process = self._processes.get(account_key(run_id))
        if process is not None:
            _terminate(process)

    def _cancel_event(self, run_id: str) -> threading.Event:
        return self._cancel_events.setdefault(account_key(run_id), threading.Event())

    async def _check_active(self, run_id: str) -> None:
        if account_key(run_id) in self._lost_leases:
            raise LeaseLost()
        cancelled, owns_lease = await asyncio.gather(
            asyncio.to_thread(db.is_cancel_requested, run_id),
            asyncio.to_thread(db.owns_lease, run_id, self.worker_id),
        )
        if cancelled:
            self.cancel(run_id)
            raise RunCancelled()
        if not owns_lease:
            raise LeaseLost()
        if self._cancel_event(run_id).is_set():
            self.cancel(run_id)
            raise RunCancelled()

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                account, run = await asyncio.to_thread(self._claim_account_run)
                if run is None:
                    await asyncio.sleep(self.poll_seconds)
                    continue
                self.job_account = account
                try:
                    await arun_as(account, self._process(run))
                finally:
                    self.job_account = None
            except asyncio.CancelledError:
                raise
            except sqlite3.OperationalError as exc:
                # Losing the writer lock is normal for polling, not a fault; neither
                # branch may re-raise, since that escapes the loop and stops the
                # supervisor for the life of the process.
                if is_sqlite_busy_error(exc):
                    logger.warning("make.supervisor_db_busy: %s", exc)
                else:
                    logger.exception("make.supervisor_iteration_failed")
                await asyncio.sleep(1)
            except Exception:
                logger.exception("make.supervisor_iteration_failed")
                await asyncio.sleep(1)

    def _claim_account_run(self):
        accounts = job_accounts()
        start = 0
        for index, account in enumerate(accounts):
            if account.account_id == self._last_claim_account:
                start = index + 1
                break
        for offset in range(len(accounts)):
            account = accounts[(start + offset) % len(accounts)]
            try:
                run = run_as(account, db.claim_next, self.worker_id)
            except Exception:
                logger.exception("make.claim_failed account=%s", account.account_id)
                continue
            if run is not None:
                self._last_claim_account = account.account_id
                return account, run
        return None, None

    async def _heartbeat(self, run_id: str) -> None:
        while True:
            await asyncio.sleep(LEASE_MS / 1000 / 3)
            if not await asyncio.to_thread(db.heartbeat, run_id, self.worker_id, LEASE_MS):
                self._lost_leases.add(account_key(run_id))
                return

    async def _process(self, run: dict) -> None:
        run_id = run["id"]
        heartbeat = asyncio.create_task(self._heartbeat(run_id))
        try:
            await self._execute(run)
        except RunCancelled:
            await asyncio.to_thread(db.finish, run_id, self.worker_id, "cancelled")
        except LeaseLost:
            # Another worker owns this run now. Standing in a lease-expired row would
            # overwrite its result, so the terminal write is left to its owner.
            pass
        except Exception as exc:  # noqa: BLE001 - the run must end as failed, not vanish
            logger.exception("make.run_failed run_id=%s", run_id)
            await asyncio.to_thread(
                db.finish,
                run_id,
                self.worker_id,
                "failed",
                f"{type(exc).__name__}: {exc}"[:2000],
            )
        finally:
            heartbeat.cancel()
            self._processes.pop(account_key(run_id), None)
            self._cancel_events.pop(account_key(run_id), None)
            self._lost_leases.discard(account_key(run_id))

    async def _execute(self, run: dict) -> None:
        run_id = run["id"]
        config = run.get("config") or {}
        await self._check_active(run_id)
        env = environment()
        if not env["available"]:
            await asyncio.to_thread(db.finish, run_id, self.worker_id, "failed", env["error"])
            return
        argv = build_argv(config)
        # The checkpoint directory is derived, never taken from the request, so a run
        # cannot write outside the project's own base dir.
        checkpoint_dir = Path(env["checkpointRoot"]) / config["runTag"]
        child_env = build_child_env(env)
        await asyncio.to_thread(
            db.append_worker_event,
            run_id,
            self.worker_id,
            "run.spawned",
            {"argv": argv, "checkpointDir": str(checkpoint_dir), "python": env["python"]},
        )
        try:
            process = await asyncio.to_thread(
                _spawn, str(env["python"]), argv, env["projectRoot"], child_env
            )
        except Exception as exc:  # noqa: BLE001 - reported as a failed run
            await asyncio.to_thread(
                db.finish,
                run_id,
                self.worker_id,
                "failed",
                f"Could not start the trainer: {type(exc).__name__}: {exc}",
            )
            return
        self._processes[account_key(run_id)] = process
        lines: queue.Queue = queue.Queue()
        reader = threading.Thread(
            target = _read_lines,
            args = (process, lines),
            name = f"make-log-{run_id[:8]}",
            daemon = True,
        )
        reader.start()
        timeout = _timeout_for(config)
        deadline = time.monotonic() + timeout
        pending: list[str] = []
        last_flush = time.monotonic()
        last_check = time.monotonic()
        last_progress: str | None = None
        try:
            while True:
                pending, last_flush = await self._flush_log(run_id, lines, pending, last_flush)
                now = time.monotonic()
                if now >= deadline:
                    await self._note(
                        run_id, f"Run exceeded its {timeout}s limit and is being stopped."
                    )
                    _terminate(process)
                    await self._drain(run_id, process, lines, reader)
                    await asyncio.to_thread(
                        db.finish, run_id, self.worker_id, "failed", f"Timed out after {timeout}s"
                    )
                    return
                if now - last_check >= CHECK_INTERVAL_SECONDS:
                    last_check = now
                    await self._check_active(run_id)
                progress = _progress_from_log(pending)
                if progress is not None:
                    key = json.dumps(progress, sort_keys = True, default = str)
                    if key != last_progress:
                        last_progress = key
                        await asyncio.to_thread(
                            db.append_worker_event,
                            run_id,
                            self.worker_id,
                            "run.progress",
                            progress,
                        )
                if process.poll() is not None and lines.empty():
                    break
                await asyncio.sleep(TICK_SECONDS)
            if checkpoint_dir.is_dir():
                await asyncio.to_thread(
                    db.set_checkpoint_dir, run_id, self.worker_id, str(checkpoint_dir)
                )
                await asyncio.to_thread(
                    db.append_worker_event,
                    run_id,
                    self.worker_id,
                    "run.artifacts",
                    {"checkpointDir": str(checkpoint_dir)},
                )
            metrics = await asyncio.to_thread(collect_results, checkpoint_dir)
            returncode = process.returncode
            # _check_active raises on a cancel, so reaching here means the run was never
            # asked to stop. The flag is re-read anyway: a cancel that landed in the last
            # tick's window is the difference between cancelled and completed.
            if await asyncio.to_thread(db.is_cancel_requested, run_id):
                await asyncio.to_thread(db.finish, run_id, self.worker_id, "cancelled")
                return
            if returncode == 0 and checkpoint_dir.is_dir():
                await asyncio.to_thread(
                    db.finish, run_id, self.worker_id, "completed", None, metrics
                )
                return
            detail = f"The trainer exited with code {returncode}"
            if not checkpoint_dir.is_dir():
                # The most common cause by far, and worth saying plainly rather than
                # leaving a bare exit code to be interpreted.
                detail += "; it wrote no checkpoint directory, so it stopped before training"
            await asyncio.to_thread(db.finish, run_id, self.worker_id, "failed", detail)
        finally:
            self._processes.pop(account_key(run_id), None)
            await self._drain(run_id, process, lines, reader)

    async def _note(self, run_id: str, message: str) -> None:
        await asyncio.to_thread(
            db.append_worker_event, run_id, self.worker_id, "run.note", {"message": message}
        )

    async def _flush_log(
        self, run_id: str, lines: queue.Queue, pending: list[str], last_flush: float
    ) -> tuple[list[str], float]:
        while True:
            try:
                item = lines.get_nowait()
            except queue.Empty:
                break
            if item is None:
                break
            pending.append(item)
        now = time.monotonic()
        if not pending or (len(pending) < db.LOG_FLUSH_LINES and now - last_flush < db.LOG_FLUSH_SECONDS):
            return pending, last_flush
        text = "".join(pending)
        pending = []
        last_flush = now
        await asyncio.to_thread(db.append_log, run_id, self.worker_id, text)
        return pending, last_flush

    async def _drain(
        self, run_id: str, process: subprocess.Popen, lines: queue.Queue, reader: threading.Thread
    ) -> None:
        """Wait for the child, then flush whatever it printed on its way out."""
        try:
            await asyncio.to_thread(process.wait, TERMINATE_GRACE_SECONDS + 5)
        except Exception:  # noqa: BLE001 - the reader below is bounded anyway
            pass
        await asyncio.to_thread(lambda: reader.join(5.0))
        pending, _ = await self._flush_log(run_id, lines, [], time.monotonic())
        if pending:
            await asyncio.to_thread(db.append_log, run_id, self.worker_id, "".join(pending))


def _timeout_for(config: dict[str, Any]) -> int:
    raw = config.get("timeoutSeconds")
    if raw is None:
        # Generous by default: a Medium preset on a slow card is hours, and the point of
        # a Make run is that it finishes unattended.
        return MAX_TIMEOUT_SECONDS
    return max(MIN_TIMEOUT_SECONDS, min(int(raw), MAX_TIMEOUT_SECONDS))


def _read_lines(process: subprocess.Popen, lines: queue.Queue) -> None:
    stream = process.stdout
    try:
        if stream is not None:
            for line in stream:
                lines.put(line if line.endswith("\n") else line + "\n")
    except Exception:  # noqa: BLE001 - the reader is a log tap, not the run
        pass
    finally:
        lines.put(None)


def retire_account_make(account) -> None:
    """Stop a retiring account's run: the child goes down before its records are swept."""
    for supervisor in list(_supervisors):
        if getattr(supervisor, "job_account", None) is None:
            continue
        if supervisor.job_account.account_id != account.account_id:
            continue
        for cancel_event in supervisor._cancel_events.values():
            cancel_event.set()
        for process in list(supervisor._processes.values()):
            _terminate(process)
