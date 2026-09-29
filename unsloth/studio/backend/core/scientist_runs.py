# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Durable AI Scientist runs: spawn nanochat's BFTS launcher, stream it, record it.

The launcher (``launch_scientist_bfts.py``) owns the science: the idea, the candidate
code, the training, the artifacts. This module owns the process and the Studio-visible
record of it, so a run can be started from the app, watched over SSE, cancelled, and
picked back up after a restart.

Two boundaries matter and are both closed here:

* **argv and paths are the controller's.** The client picks an idea, an attempt, a
  model and a lineage handoff; everything else (interpreter, project root, shared
  cache, experiment root) is resolved from the server's own configuration. A run
  cannot be pointed at an arbitrary interpreter or a directory outside the project's
  artifact root.
* **the launcher keeps its own gates.** ``--max-nodes 1`` and ``--allow-provider-calls``
  are the launcher's fail-closed preconditions, so they are always passed; the route
  refuses a request that would not produce them rather than spawning a child that
  would die on the check.
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
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import weakref
from pathlib import Path
from typing import Any

from storage import scientist_runs_db as db

_supervisors = weakref.WeakSet()

# The launcher names this file; the whole integration hangs off its presence.
LAUNCHER_FILENAME = "launch_scientist_bfts.py"
# What the launcher writes as it advances, and the per-stage tree-search summary.
RUN_STATUS_FILENAME = "run_status.json"
STAGE_PROGRESS_GLOB = "**/stage_progress.json"
# The training result contract written by the nanochat adapter for the best node.
RESULTS_FILENAME = "results.json"
# Curves are plotted into PNGs, so the series themselves are small; the cap only
# stops a pathological artifact from landing in the studio database.
MAX_CURVE_POINTS = 2000
MAX_CURVE_SERIES = 24
# A cancelled launcher gets this long to exit on SIGTERM before it is killed. The
# launcher spawns a training child, so the whole tree has to come down, not just it.
TERMINATE_GRACE_SECONDS = 15.0
# Polling cadence for the child's stdout, its artifacts, the lease and the deadline.
TICK_SECONDS = 0.25
# How often a streaming run re-reads its cancel flag and lease. Long enough that a
# half-hour run costs a few hundred reads, short enough that a cancel lands at once:
# the route also signals the supervisor directly, so this is the belt to that braces.
_CHECK_INTERVAL_SECONDS = 2.0
LEASE_MS = 120_000
# The launcher's own guardrail is max_training_seconds=1800; the ceiling here is
# generous enough to cover a slow preflight plus ideation on top of the node itself.
MIN_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 6 * 3600

# Interpreter-independent essentials. Anything a child needs to start at all, and
# nothing that could carry a credential the run did not ask for.
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
# Pass-through settings a user legitimately pins for their box.
_OPTIONAL_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTHONUTF8",
    "PYTHONHASHSEED",
)
# The only credentials a run can reach, and only from the server's own environment.
# They are never written to the run row, the config, or an event.
_PROVIDER_ENV = (
    "OPENCODE_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENROUTER_APP_TITLE",
    "OPENROUTER_HTTP_REFERER",
    "AI_SCIENTIST_API_MODE",
    "AI_SCIENTIST_MAX_API_CALLS",
    "AI_SCIENTIST_MAX_INPUT_TOKENS",
    "AI_SCIENTIST_MAX_OUTPUT_TOKENS",
    "AI_SCIENTIST_ALLOW_MISSING_USAGE",
    "AI_SCIENTIST_BUDGET_FILE",
)
# Server-side configuration. Read from Studio's own environment, never from a request,
# so a run cannot be redirected to another checkout or another device.
_PROJECT_ROOT_ENV = "UNSLOTH_SCIENTIST_PROJECT_ROOT"
_PYTHON_ENV = "UNSLOTH_SCIENTIST_PYTHON"
_SHARED_CACHE_ENV = "UNSLOTH_SCIENTIST_SHARED_CACHE"
_EXPERIMENT_ROOT_ENV = "UNSLOTH_SCIENTIST_EXPERIMENT_ROOT"

# Where this file sits relative to a checkout that keeps unsloth/ inside it:
# <root>/unsloth/studio/backend/core/scientist_runs.py
_REPO_ROOT_PARENTS = 4


class RunCancelled(Exception):
    pass


class LeaseLost(Exception):
    pass


logger = get_logger(__name__)


def _candidate_project_roots() -> list[Path]:
    """Every place a nanochat checkout could be, most specific first."""
    roots: list[Path] = []
    configured = os.environ.get(_PROJECT_ROOT_ENV, "").strip()
    if configured:
        roots.append(Path(configured).expanduser())
    here = Path(__file__).resolve()
    # core/scientist_runs.py -> backend -> studio -> unsloth -> checkout root.
    if len(here.parents) > _REPO_ROOT_PARENTS:
        roots.append(here.parents[_REPO_ROOT_PARENTS])
    # And every ancestor of it, so the folder can sit anywhere in the tree.
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
        if (root / LAUNCHER_FILENAME).is_file():
            return root
    return None


def _resolve_python(root: Path) -> str | None:
    """The interpreter that can import nanochat and the AI Scientist package.

    Prefers an explicit setting, then the checkout's own virtualenv, then this
    process's interpreter. The last is the fallback, not the default: the scientist
    env is a separate Python 3.11 venv with torch, and Studio's own interpreter
    usually cannot import it.
    """
    configured = os.environ.get(_PYTHON_ENV, "").strip()
    if configured:
        return configured
    for candidate in (root / ".venv" / "bin" / "python", root / ".venv" / "Scripts" / "python.exe"):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _resolve_shared_cache(root: Path) -> Path | None:
    """The nanochat cache the launcher will read, or None if there is not one.

    A configured value that is not a directory is skipped rather than trusted: a
    stale ``NANOCHAT_SHARED_CACHE`` left over from a deleted mount would otherwise be
    handed to the launcher, which would create a fresh empty tree under it and then
    fail deep inside preflight with a missing tokenizer.
    """
    for name in (_SHARED_CACHE_ENV, "NANOCHAT_SHARED_CACHE"):
        configured = os.environ.get(name, "").strip()
        if not configured:
            continue
        candidate = Path(configured).expanduser()
        if candidate.is_dir():
            return candidate
    for candidate in (root / "data", root / "cache"):
        if candidate.is_dir():
            return candidate
    return None


# What nanochat's tokenizer_hash() requires before it will produce a digest. Without
# these the results contract's provenance is silently empty, so an empty cache is not a
# usable cache even though the directory exists.
_TOKENIZER_FILES = ("tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt")


def cache_is_usable(cache: Path) -> bool:
    return all((cache / name).is_file() for name in _TOKENIZER_FILES)


def interpreter_can_import(root: Path, python: str) -> bool:
    """Whether ``python`` can import the two packages the launcher needs.

    Checked by running the interpreter, not by inspecting the path: a fallback to this
    process's own interpreter looks fine on paper and dies on the first import. The
    probe is a one-off on the environment page, not on the run path, so a subprocess
    here costs nothing that matters.
    """
    import subprocess

    try:
        completed = subprocess.run(
            [
                python,
                "-c",
                "import ai_scientist.providers, nanochat.research_results",
            ],
            cwd = str(root),
            capture_output = True,
            timeout = 120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _resolve_experiment_root(root: Path) -> Path:
    configured = os.environ.get(_EXPERIMENT_ROOT_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return root / "experiments"


def _problem(code: str, message: str, fix: str) -> dict[str, str]:
    """One thing standing between this install and a run that can start.

    ``fix`` is the literal remedy, because the alternative is a page that says
    "not available" and leaves the reader to work out which of four environment
    variables they are missing.
    """
    return {"code": code, "message": message, "fix": fix}


def environment() -> dict[str, Any]:
    """What this installation can actually run, for the UI to show before a run starts.

    Deliberately pessimistic: every check here is one whose failure would otherwise
    surface as a launcher crash minutes into a run. A missing piece is reported, not
    raised, because the page needs to explain the problem and the create route
    re-checks the same facts and refuses.
    """
    root = find_project_root()
    if root is None:
        return {
            "available": False,
            "error": (
                f"Could not find {LAUNCHER_FILENAME}. Set {_PROJECT_ROOT_ENV} to the "
                "nanochat checkout that contains it."
            ),
            "problems": [
                _problem(
                    "no-project-root",
                    f"Could not find {LAUNCHER_FILENAME}.",
                    f"Set {_PROJECT_ROOT_ENV} to the nanochat checkout that contains it.",
                )
            ],
            "projectRoot": None,
            "python": None,
            "pythonIsFallback": False,
            "sharedCache": None,
            "experimentRoot": None,
            "providerKeys": [],
        }
    python = _resolve_python(root)
    cache = _resolve_shared_cache(root)
    experiments = _resolve_experiment_root(root)
    configured_python = os.environ.get(_PYTHON_ENV, "").strip()
    is_fallback = not configured_python and python == sys.executable
    provider_keys = sorted(
        name for name in _PROVIDER_ENV if name.endswith("API_KEY") and os.environ.get(name)
    )
    problems: list[dict[str, str]] = []
    if python is None:
        problems.append(
            _problem(
                "missing-python",
                f"{_PYTHON_ENV} points at a missing interpreter.",
                f"Unset {_PYTHON_ENV}, or point it at an interpreter that exists.",
            )
        )
    elif is_fallback:
        # Studio's own interpreter is the last resort, not a working default: it has
        # torch, but not the nanochat packages, so the launcher would fail on its
        # first import. Say that instead of presenting it as configured.
        problems.append(
            _problem(
                "python-fallback",
                f"No nanochat interpreter configured; falling back to {python}, "
                "which cannot import nanochat.",
                f"Set {_PYTHON_ENV} to the nanochat virtualenv's interpreter "
                f"(for example {root / '.venv' / 'Scripts' / 'python.exe'}), or run "
                "`uv sync` in the checkout.",
            )
        )
    elif not interpreter_can_import(root, python):
        problems.append(
            _problem(
                "python-cannot-import",
                f"{python} cannot import ai_scientist and nanochat from {root}.",
                f"Install the scientist dependencies into that interpreter, or point "
                f"{_PYTHON_ENV} at one that has them.",
            )
        )
    if cache is None:
        problems.append(
            _problem(
                "no-shared-cache",
                "No nanochat shared cache found; the launcher requires it.",
                f"Set {_SHARED_CACHE_ENV} (or NANOCHAT_SHARED_CACHE) to the cache "
                "directory, and populate it with the tokenizer and datasets.",
            )
        )
    elif not cache_is_usable(cache):
        problems.append(
            _problem(
                "empty-shared-cache",
                f"{cache} has no tokenizer, so a run cannot produce a valid result.",
                "Populate the shared cache (the tokenizer and the ClimbMix/eval "
                "bundles) before starting a run.",
            )
        )
    if not provider_keys:
        problems.append(
            _problem(
                "no-provider-key",
                "No provider API key is set, so preflight will fail closed.",
                "Set one of "
                + ", ".join(name for name in _PROVIDER_ENV if name.endswith("API_KEY"))
                + " in Studio's environment before starting a run.",
            )
        )
    # Only the experiment root is created: a missing shared cache is the user's to fix,
    # but the artifact directory is ours to own and the launcher creates it per run.
    try:
        experiments.mkdir(parents = True, exist_ok = True)
    except OSError as exc:
        problems.append(
            _problem(
                "experiment-root-unwritable",
                f"Cannot write the experiment directory: {exc}",
                f"Make {experiments} writable, or set {_EXPERIMENT_ROOT_ENV}.",
            )
        )
    return {
        "available": not problems,
        "error": "; ".join(problem["message"] for problem in problems) or None,
        "problems": problems,
        "projectRoot": str(root),
        "python": python,
        "pythonIsFallback": is_fallback,
        "sharedCache": str(cache) if cache is not None else None,
        "experimentRoot": str(experiments),
        "providerKeys": provider_keys,
    }


def build_argv(config: dict[str, Any], env: dict[str, Any]) -> list[str]:
    """The exact command line, built here so it is stored on the run and displayed.

    The two gates the launcher enforces are always present: a run that could not pass
    them is rejected at the route, not spawned to fail.
    """
    argv = [str(env["python"]), str(Path(env["projectRoot"]) / LAUNCHER_FILENAME)]
    argv += [
        "--load-ideas",
        str(config.get("loadIdeas") or "ai_scientist/ideas/nanochat_pretraining.json"),
    ]
    if config.get("loadCode"):
        argv.append("--load-code")
    argv += [
        "--idea-idx",
        str(int(config.get("ideaIdx", 0))),
        "--attempt-id",
        str(int(config.get("attemptId", 0))),
        "--model",
        str(config["model"]),
        "--config",
        str(config.get("config") or "bfts_config.yaml"),
        "--max-nodes",
        "1",
        "--allow-provider-calls",
    ]
    for flag, key in (
        ("--parent-journal", "parentJournal"),
        ("--parent-node-id", "parentNodeId"),
        ("--lineage-id", "lineageId"),
    ):
        value = config.get(key)
        if value:
            argv += [flag, str(value)]
    stage = config.get("parentStage")
    if stage:
        argv += ["--parent-stage", str(int(stage))]
    return argv


def build_child_env(env: dict[str, Any], run_id: str, staging: Path) -> dict[str, str]:
    """A fresh environment for the launcher: an allowlist, never a copy of ours.

    The credentials are the only reason the child sees anything of our environment,
    and they go through the environment rather than argv so they stay out of ``ps``
    and out of the run row.
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
    for name in _PROVIDER_ENV:
        value = os.environ.get(name)
        if value:
            child[name] = value
    child.update(
        PYTHONUNBUFFERED = "1",
        PYTHONIOENCODING = "utf-8",
        NANOCHAT_SHARED_CACHE = str(env["sharedCache"]),
        AI_SCIENTIST_EXPERIMENT_DIR = str(staging),
        AI_SCIENTIST_RUN_ID = run_id,
    )
    return child


def _read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding = "utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _newest(root: Path, name: str) -> Path | None:
    best: tuple[float, Path] | None = None
    for candidate in root.rglob(name):
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, candidate)
    return None if best is None else best[1]


def discover_idea_dir(staging: Path) -> Path | None:
    """The launcher's per-run idea directory, once it has created it.

    The launcher names it after the wall clock, so the only stable handle is the
    staging directory Studio gave it: exactly one idea directory is created there.
    """
    try:
        entries = [entry for entry in staging.iterdir() if entry.is_dir()]
    except OSError:
        return None
    return entries[0] if len(entries) == 1 else None


def collect_results(staging: Path) -> tuple[dict[str, Any] | None, Any | None]:
    """The best node's metrics and curves, or (None, None) if the run wrote none.

    A run that fails before training legitimately has no result artifact, so absence
    is not an error here; the run's own status is what says whether it mattered.
    """
    path = _newest(staging, RESULTS_FILENAME)
    if path is None:
        return None, None
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return None, None
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        metrics = None
    else:
        # Keep the numbers JSON-safe: NaN and Infinity are not JSON, and the studio
        # database is read by a browser's JSON.parse.
        metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float, str, bool)) or value is None
        }
        metrics = {
            key: value
            for key, value in metrics.items()
            if not isinstance(value, float) or value == value
        }
    primary = payload.get("primary_metric")
    if metrics is not None and isinstance(primary, dict) and "name" in primary:
        metrics = {"primary_metric": primary, **metrics}
    curves = payload.get("curves")
    if not isinstance(curves, dict):
        return metrics, None
    trimmed: dict[str, list[float]] = {}
    for name, series in list(curves.items())[:MAX_CURVE_SERIES]:
        if isinstance(series, list):
            trimmed[str(name)] = [
                float(point)
                for point in series[:MAX_CURVE_POINTS]
                if isinstance(point, (int, float))
            ]
    return metrics, trimmed or None


def _stage_progress(staging: Path) -> dict[str, Any] | None:
    path = _newest(staging, "stage_progress.json")
    if path is None:
        return None
    payload = _read_json(path)
    return payload if isinstance(payload, dict) else None


def _spawn(argv: list[str], cwd: str, env: dict[str, str]) -> subprocess.Popen:
    from utils.process_lifetime import adopt_pid, child_popen_kwargs, spawn_on_lifetime_thread

    def _popen() -> subprocess.Popen:
        return subprocess.Popen(
            argv,
            cwd = cwd,
            env = env,
            # stderr folded into stdout: the launcher's tracebacks are the run's log.
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
    """Stop the launcher and the training node under it, then let go of the record."""
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
    # The launcher spawns the trainer, so the leader alone is not the run. terminate_pid
    # takes the validated tree and then drops the record, which forget_pid must not be
    # asked to do twice.
    threading.Thread(
        target = terminate_pid, args = (pid,), kwargs = {"owner_verified": True}, daemon = True
    ).start()


class ScientistSupervisor:
    def __init__(
        self,
        app: Any,
        poll_seconds: float = 0.5,
    ) -> None:
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
            self._task = asyncio.create_task(self._loop(), name = "scientist-supervisor")

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
                    # Polling is intentionally sufficient for one local process;
                    # requests never own tasks.
                    pass
        finally:
            for account in job_accounts():
                await asyncio.to_thread(run_as, account, db.release_worker_leases, self.worker_id)

    def wake(self) -> None:
        # Polling is the admission path, exactly as for research runs: the supervisor
        # is one long-lived task and a wake flag would only add a second thing to keep
        # correct across accounts.
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
                # branch may re-raise, since that escapes the while loop and stops the
                # supervisor for the life of the process.
                if is_sqlite_busy_error(exc):
                    logger.warning("scientist.supervisor_db_busy: %s", exc)
                else:
                    logger.exception("scientist.supervisor_iteration_failed")
                await asyncio.sleep(1)
            except Exception:
                logger.exception("scientist.supervisor_iteration_failed")
                await asyncio.sleep(1)

    def _claim_account_run(self):
        # Round robin from the account after the last claim: one training node owns the
        # GPU, so runs are processed strictly one at a time.
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
                logger.exception("scientist.claim_failed account=%s", account.account_id)
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
            logger.exception("scientist.run_failed run_id=%s", run_id)
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
        argv = build_argv(config, env)
        staging = Path(env["experimentRoot"]) / f"studio-{run_id}"
        # A retried run gets a clean directory: the launcher refuses to create an idea
        # dir that already exists, and a stale one would hold the previous attempt's
        # artifacts and results.
        await asyncio.to_thread(_reset_staging, staging)
        child_env = build_child_env(env, run_id, staging)
        await asyncio.to_thread(
            db.append_worker_event,
            run_id,
            self.worker_id,
            "run.spawned",
            {"argv": argv, "experimentDir": str(staging)},
        )
        try:
            process = await asyncio.to_thread(_spawn, argv, env["projectRoot"], child_env)
        except Exception as exc:  # noqa: BLE001 - reported as a failed run
            await asyncio.to_thread(
                db.finish,
                run_id,
                self.worker_id,
                "failed",
                f"Could not start the AI Scientist launcher: {type(exc).__name__}: {exc}",
            )
            return
        self._processes[account_key(run_id)] = process
        lines: queue.Queue = queue.Queue()
        reader = threading.Thread(
            target = _read_lines,
            args = (process, lines),
            name = f"scientist-log-{run_id[:8]}",
            daemon = True,
        )
        reader.start()
        timeout = _timeout_for(config)
        deadline = time.monotonic() + timeout
        flushed: list[str] = []
        last_flush = time.monotonic()
        last_progress: str | None = None
        last_check = time.monotonic()
        try:
            while True:
                flushed, last_flush = await self._flush_log(run_id, lines, flushed, last_flush)
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
                # A cancel and a lost lease both have to be noticed while a run is
                # streaming, not only at the boundaries, so the check is on a cadence
                # rather than once: two reads per tick would be 8 a second for a
                # process that can legitimately take half an hour.
                if now - last_check >= _CHECK_INTERVAL_SECONDS:
                    last_check = now
                    await self._check_active(run_id)
                progress = await asyncio.to_thread(_stage_progress, staging)
                if progress is not None:
                    key = json.dumps(progress, sort_keys = True, default = str)
                    if key != last_progress:
                        last_progress = key
                        await asyncio.to_thread(
                            db.append_worker_event, run_id, self.worker_id, "run.progress", progress
                        )
                if process.poll() is not None and lines.empty():
                    break
                await asyncio.sleep(TICK_SECONDS)
            idea_dir = await asyncio.to_thread(discover_idea_dir, staging)
            if idea_dir is not None:
                await asyncio.to_thread(db.set_artifact_dir, run_id, self.worker_id, str(idea_dir))
                await asyncio.to_thread(
                    db.append_worker_event,
                    run_id,
                    self.worker_id,
                    "run.artifacts",
                    {"artifactDir": str(idea_dir)},
                )
            metrics, curves = await asyncio.to_thread(collect_results, staging)
            returncode = process.returncode
            # _check_active raises on a cancel, so reaching here means the run was never
            # asked to stop. The flag is re-read anyway: a cancel that landed in the
            # last tick's window is the difference between cancelled and completed.
            if await asyncio.to_thread(db.is_cancel_requested, run_id):
                await asyncio.to_thread(db.finish, run_id, self.worker_id, "cancelled")
                return
            # The launcher is fail-closed: it writes run_status.json "complete" only
            # after the BFTS finished and the integrity manifest still matched, so the
            # file, not the exit code, is the authority on what happened.
            status = await asyncio.to_thread(_final_status, idea_dir)
            if status == "complete" or (status is None and returncode == 0):
                await asyncio.to_thread(
                    db.finish, run_id, self.worker_id, "completed", None, metrics, curves
                )
                return
            error = None
            if idea_dir is not None:
                recorded = _read_json(idea_dir / RUN_STATUS_FILENAME)
                if isinstance(recorded, dict):
                    error = recorded.get("error")
            if not error:
                error = f"The AI Scientist launcher exited with code {returncode}"
            await asyncio.to_thread(db.finish, run_id, self.worker_id, "failed", str(error)[:2000])
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
        if not pending or (
            len(pending) < db.LOG_FLUSH_LINES and now - last_flush < db.LOG_FLUSH_SECONDS
        ):
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


def _reset_staging(staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging, ignore_errors = True)
    staging.mkdir(parents = True, exist_ok = True)


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


def _timeout_for(config: dict[str, Any]) -> int:
    raw = config.get("timeoutSeconds")
    try:
        value = int(raw) if raw is not None else MAX_TIMEOUT_SECONDS
    except (TypeError, ValueError):
        value = MAX_TIMEOUT_SECONDS
    return max(MIN_TIMEOUT_SECONDS, min(value, MAX_TIMEOUT_SECONDS))


def _final_status(idea_dir: Path | None) -> str | None:
    if idea_dir is None:
        return None
    payload = _read_json(idea_dir / RUN_STATUS_FILENAME)
    if isinstance(payload, dict):
        status = payload.get("status")
        return status if isinstance(status, str) else None
    return None


def retire_account_scientist(account) -> None:
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
