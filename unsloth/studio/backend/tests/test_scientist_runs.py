# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AI Scientist runs: the launcher's gates, the storage lease, and the HTTP surface.

The point of most of these is that a run cannot be talked into something the launcher
would refuse. The rest is the ordinary durable-run machinery, which has to behave the
same way it does for Deep Research or the feature is a second, weaker implementation
of the same idea.
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import authenticated_via_api_key, get_current_subject
from core import scientist_runs
from routes import scientist_runs as routes
from storage import scientist_runs_db as db
from storage import studio_db
from utils.account_context import (
    OWNER,
    AccountContext,
    bind_account,
    reset_account,
    run_as,
)

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")
ACCOUNTS = {account.username: account for account in (OWNER, ALICE, BOB)}
WORKER = "worker-1"


@pytest.fixture(autouse = True)
def _reset_schemas(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    # This module has no schema of its own: it reads the one storage.studio_db creates,
    # and conftest's home isolation points that at a fresh file per test.
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    yield


@pytest.fixture(autouse = True)
def alice():
    """Bind alice for the test body.

    Managed accounts each own a database, and TestClient serves the app on its own
    thread, so a direct ``db.`` call in a test has to be under the same context the
    request was, or it reads a different install entirely. Autouse so no test can
    quietly assert against the owner's empty database.
    """
    marker = bind_account(ALICE)
    try:
        yield ALICE
    finally:
        reset_account(marker)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A directory that looks enough like a nanochat checkout for the route to pass."""
    root = tmp_path / "nanochat"
    (root / "ai_scientist" / "ideas").mkdir(parents = True)
    (root / "launch_scientist_bfts.py").write_text("# launcher\n", encoding = "utf-8")
    (root / "bfts_config.yaml").write_text("agent: {}\n", encoding = "utf-8")
    (root / "ai_scientist" / "ideas" / "one.json").write_text("[]", encoding = "utf-8")
    (root / "ai_scientist" / "ideas" / "one.py").write_text("# code\n", encoding = "utf-8")
    cache = tmp_path / "cache"
    (cache / "tokenizer").mkdir(parents = True)
    for name in ("tokenizer.pkl", "token_bytes.pt"):
        (cache / "tokenizer" / name).write_bytes(b"\x00")
    experiments = tmp_path / "experiments"
    monkeypatch.setenv(scientist_runs._PROJECT_ROOT_ENV, str(root))
    monkeypatch.setenv(scientist_runs._PYTHON_ENV, sys.executable)
    monkeypatch.setenv(scientist_runs._SHARED_CACHE_ENV, str(cache))
    monkeypatch.setenv(scientist_runs._EXPERIMENT_ROOT_ENV, str(experiments))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-not-a-real-key")
    # The real probe runs an interpreter, which a temp checkout cannot satisfy. What it
    # decides is covered on its own below; here it is only a gate the route reads.
    monkeypatch.setattr(scientist_runs, "interpreter_can_import", lambda root, python: True)
    return root


def _payload(**overrides):
    base = {
        "model": "opencode/space-bunny-free",
        "loadIdeas": "ai_scientist/ideas/one.json",
        "config": "bfts_config.yaml",
        "allowProviderCalls": True,
    }
    base.update(overrides)
    return base


@pytest.fixture
def client():
    app = FastAPI()

    async def subject(request: Request):
        account = ACCOUNTS[request.headers.get("x-test-account", "alice")]
        marker = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(marker)

    app.dependency_overrides[get_current_subject] = subject
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    app.include_router(routes.router, prefix = "/scientist")
    app.state.scientist_supervisor = None
    with TestClient(app) as test_client:
        yield test_client


# ── environment resolution ──────────────────────────────────────────────────────


def test_environment_reports_a_usable_checkout(checkout):
    env = scientist_runs.environment()
    assert env["available"] is True
    assert env["error"] is None
    assert Path(env["projectRoot"]) == checkout.resolve()
    assert env["python"] == sys.executable
    assert Path(env["sharedCache"]).is_dir()
    assert Path(env["experimentRoot"]).is_dir()


def test_environment_says_why_it_cannot_run(tmp_path, monkeypatch):
    # Neither the configured root nor anything above this module holds a launcher, so
    # the probe must report that rather than settle for a directory that does not look
    # like a checkout at all.
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "studio" / "backend" / "core").mkdir(parents = True)
    monkeypatch.setenv(scientist_runs._PROJECT_ROOT_ENV, str(elsewhere))
    monkeypatch.setattr(
        scientist_runs, "__file__", str(elsewhere / "studio" / "backend" / "core" / "x.py")
    )
    env = scientist_runs.environment()
    assert env["available"] is False
    assert scientist_runs.LAUNCHER_FILENAME in env["error"]


def test_environment_reports_a_missing_shared_cache(checkout, tmp_path, monkeypatch):
    monkeypatch.delenv(scientist_runs._SHARED_CACHE_ENV)
    monkeypatch.setenv("NANOCHAT_SHARED_CACHE", str(tmp_path / "absent"))
    for name in ("data", "cache"):
        candidate = checkout / name
        if candidate.is_dir():
            candidate.rmdir()
    env = scientist_runs.environment()
    assert env["available"] is False
    assert "no-shared-cache" in {problem["code"] for problem in env["problems"]}


def test_a_cache_without_a_tokenizer_is_not_usable(checkout, monkeypatch):
    # The directory exists, so a check that only tests is_dir() would call this ready
    # and hand the launcher a run whose results carry no provenance at all.
    tokenizer = Path(scientist_runs.environment()["sharedCache"]) / "tokenizer"
    (tokenizer / "tokenizer.pkl").unlink()
    env = scientist_runs.environment()
    assert env["available"] is False
    problem = next(p for p in env["problems"] if p["code"] == "empty-shared-cache")
    assert str(tokenizer.parent) in problem["message"]


def test_cache_is_usable_needs_both_tokenizer_files(tmp_path):
    cache = tmp_path / "cache"
    (cache / "tokenizer").mkdir(parents = True)
    assert scientist_runs.cache_is_usable(cache) is False
    (cache / "tokenizer" / "tokenizer.pkl").write_bytes(b"\x00")
    assert scientist_runs.cache_is_usable(cache) is False
    (cache / "tokenizer" / "token_bytes.pt").write_bytes(b"\x00")
    assert scientist_runs.cache_is_usable(cache) is True


def test_no_provider_key_is_reported_rather_than_found_at_preflight(checkout, monkeypatch):
    for name in ("OPENCODE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising = False)
    env = scientist_runs.environment()
    assert env["available"] is False
    assert "no-provider-key" in {problem["code"] for problem in env["problems"]}
    # The remedy names the variables, not just the fact.
    assert "OPENROUTER_API_KEY" in next(
        p["fix"] for p in env["problems"] if p["code"] == "no-provider-key"
    )


def test_studios_own_interpreter_is_reported_as_a_fallback(checkout, monkeypatch):
    # The trap this closes: sys.executable has torch, so it looks like a working
    # default, and the launcher then dies on its first import.
    monkeypatch.delenv(scientist_runs._PYTHON_ENV, raising = False)
    env = scientist_runs.environment()
    assert env["pythonIsFallback"] is True
    problem = next(p for p in env["problems"] if p["code"] == "python-fallback")
    assert sys.executable in problem["message"]
    assert scientist_runs._PYTHON_ENV in problem["fix"]


def test_a_configured_interpreter_that_cannot_import_is_reported(checkout, monkeypatch):
    monkeypatch.setattr(scientist_runs, "interpreter_can_import", lambda root, python: False)
    env = scientist_runs.environment()
    assert env["pythonIsFallback"] is False
    assert "python-cannot-import" in {problem["code"] for problem in env["problems"]}


def test_interpreter_can_import_really_runs_the_interpreter(tmp_path):
    # Not stubbed: this is the check that decides whether a real install can start, so
    # a stubbed version of it would only prove the stub returns what it was told.
    root = tmp_path / "checkout"
    (root / "ai_scientist").mkdir(parents = True)
    (root / "nanochat").mkdir(parents = True)
    assert scientist_runs.interpreter_can_import(root, sys.executable) is False

    (root / "ai_scientist" / "__init__.py").write_text("", encoding = "utf-8")
    (root / "ai_scientist" / "providers.py").write_text("DEFAULT_MODEL = 'x'\n", encoding = "utf-8")
    (root / "nanochat" / "__init__.py").write_text("", encoding = "utf-8")
    (root / "nanochat" / "research_results.py").write_text("", encoding = "utf-8")
    assert scientist_runs.interpreter_can_import(root, sys.executable) is True

    # Half a package is still a no: the launcher imports both, and providers without
    # research_results is exactly the state a partial checkout is in.
    (root / "nanochat" / "research_results.py").unlink()
    assert scientist_runs.interpreter_can_import(root, sys.executable) is False


def test_an_unusable_interpreter_is_reported_rather_than_raising(tmp_path):
    # A path that is not an interpreter at all must not take the page down with it.
    missing = tmp_path / "not-python"
    missing.write_text("", encoding = "utf-8")
    assert scientist_runs.interpreter_can_import(tmp_path, str(missing)) is False
    assert scientist_runs.interpreter_can_import(tmp_path, str(tmp_path / "absent")) is False


def test_every_problem_carries_a_remedy(checkout, monkeypatch):
    monkeypatch.delenv(scientist_runs._PYTHON_ENV, raising = False)
    monkeypatch.delenv(scientist_runs._SHARED_CACHE_ENV, raising = False)
    for name in ("NANOCHAT_SHARED_CACHE", "OPENCODE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising = False)
    problems = scientist_runs.environment()["problems"]
    assert len(problems) >= 3
    for problem in problems:
        assert problem["message"].strip()
        assert problem["fix"].strip(), problem["code"]


# ── argv and environment ────────────────────────────────────────────────────────


def test_argv_always_carries_the_launcher_gates(checkout):
    env = scientist_runs.environment()
    argv = scientist_runs.build_argv(
        {"model": "opencode/space-bunny-free", "ideaIdx": 2, "attemptId": 3}, env
    )
    # nanochat refuses anything but --max-nodes 1 and refuses to run at all without an
    # explicit provider-call opt-in. Both are non-negotiable, so a run built here can
    # never produce the child that would have died on that check.
    assert argv[argv.index("--max-nodes") + 1] == "1"
    assert "--allow-provider-calls" in argv
    assert argv[argv.index("--idea-idx") + 1] == "2"
    assert argv[argv.index("--attempt-id") + 1] == "3"
    assert argv[0] == sys.executable
    assert argv[1].endswith("launch_scientist_bfts.py")


def test_argv_carries_lineage_only_when_complete(checkout):
    env = scientist_runs.environment()
    argv = scientist_runs.build_argv(
        {
            "model": "m",
            "parentJournal": "journal.json",
            "parentNodeId": "node-3",
            "parentStage": 2,
            "lineageId": "lin-1",
        },
        env,
    )
    assert argv[argv.index("--parent-journal") + 1] == "journal.json"
    assert argv[argv.index("--parent-stage") + 1] == "2"
    assert argv[argv.index("--lineage-id") + 1] == "lin-1"


def test_child_env_is_an_allowlist_not_a_copy(checkout, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-not-a-real-key")
    monkeypatch.setenv("HF_TOKEN", "hf_not_a_real_key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "not-a-real-key")
    env = scientist_runs.environment()
    staging = tmp_path / "staging"
    child = scientist_runs.build_child_env(env, "run-1", staging)
    # The provider key the run needs, and the run's own coordinates.
    assert child["OPENROUTER_API_KEY"] == "sk-or-not-a-real-key"
    assert child["AI_SCIENTIST_RUN_ID"] == "run-1"
    assert child["AI_SCIENTIST_EXPERIMENT_DIR"] == str(staging)
    assert child["NANOCHAT_SHARED_CACHE"] == env["sharedCache"]
    # Nothing else of ours crosses over. An allowlist, not a copy with a denylist: a
    # credential added to the server's environment later must not silently reach a run.
    assert "HF_TOKEN" not in child
    assert "AWS_SECRET_ACCESS_KEY" not in child
    assert "OPENAI_API_KEY" not in child


# ── request validation ──────────────────────────────────────────────────────────


def test_a_run_without_the_provider_opt_in_is_refused(client, checkout):
    response = client.post(
        "/scientist",
        headers = {"x-test-account": "alice"},
        json = _payload(allowProviderCalls = False),
    )
    assert response.status_code == 400
    assert "provider calls" in response.json()["detail"]


def test_a_traversing_path_is_refused(client, checkout):
    response = client.post(
        "/scientist",
        headers = {"x-test-account": "alice"},
        json = _payload(loadIdeas = "../../../etc/passwd"),
    )
    assert response.status_code == 400
    assert "relative path" in response.json()["detail"]


def test_an_absolute_path_is_refused(client, checkout, tmp_path):
    outside = tmp_path / "ideas.json"
    outside.write_text("[]", encoding = "utf-8")
    response = client.post(
        "/scientist",
        headers = {"x-test-account": "alice"},
        json = _payload(loadIdeas = str(outside)),
    )
    assert response.status_code == 400


def test_a_file_that_is_not_in_the_checkout_is_refused(client, checkout):
    response = client.post(
        "/scientist",
        headers = {"x-test-account": "alice"},
        json = _payload(config = "bfts_config.yaml"),
    )
    # The fixture writes bfts_config.yaml, so this one passes; the absent one must not.
    assert response.status_code == 202
    response = client.post(
        "/scientist",
        headers = {"x-test-account": "bob"},
        json = _payload(config = "nope.yaml"),
    )
    assert response.status_code == 400
    assert "not found" in response.json()["detail"]


def test_partial_lineage_is_refused(client, checkout):
    response = client.post(
        "/scientist",
        headers = {"x-test-account": "alice"},
        json = _payload(parentJournal = "journal.json"),
    )
    assert response.status_code == 400
    assert "together" in response.json()["detail"]


def test_an_out_of_range_timeout_is_refused(client, checkout):
    # Beyond what the launcher could honour: the route clamps, it does not accept.
    response = client.post(
        "/scientist",
        headers = {"x-test-account": "alice"},
        json = _payload(timeoutSeconds = 10**9),
    )
    assert response.status_code == 400
    assert "timeoutSeconds" in response.json()["detail"]


def test_a_negative_timeout_is_refused_by_the_schema(client, checkout):
    response = client.post(
        "/scientist",
        headers = {"x-test-account": "alice"},
        json = _payload(timeoutSeconds = -1),
    )
    assert response.status_code == 422


def test_an_unavailable_checkout_refuses_the_run_with_503(client, monkeypatch, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "studio" / "backend" / "core").mkdir(parents = True)
    monkeypatch.setenv(scientist_runs._PROJECT_ROOT_ENV, str(elsewhere))
    monkeypatch.setattr(
        scientist_runs, "__file__", str(elsewhere / "studio" / "backend" / "core" / "x.py")
    )
    response = client.post("/scientist", headers = {"x-test-account": "alice"}, json = _payload())
    assert response.status_code == 503


def test_a_model_id_that_reads_as_a_flag_is_refused(client, checkout):
    response = client.post(
        "/scientist",
        headers = {"x-test-account": "alice"},
        json = _payload(model = "--config"),
    )
    assert response.status_code == 422


def test_an_unknown_field_is_refused(client, checkout):
    response = client.post(
        "/scientist",
        headers = {"x-test-account": "alice"},
        json = {**_payload(), "environment": {"KEY": "value"}},
    )
    assert response.status_code == 422


def test_a_second_concurrent_run_is_refused(client, checkout):
    first = client.post("/scientist", headers = {"x-test-account": "alice"}, json = _payload())
    assert first.status_code == 202
    second = client.post("/scientist", headers = {"x-test-account": "alice"}, json = _payload())
    assert second.status_code == 409
    assert "already in progress" in second.json()["detail"]


def test_a_run_is_never_a_second_active_run_for_another_account(client, checkout):
    assert (
        client.post("/scientist", headers = {"x-test-account": "alice"}, json = _payload()).status_code
        == 202
    )
    # A managed account owns its own GPU request; bob is not blocked by alice.
    assert (
        client.post("/scientist", headers = {"x-test-account": "bob"}, json = _payload()).status_code
        == 202
    )


# ── the run surface ─────────────────────────────────────────────────────────────


def test_a_created_run_is_queued_and_records_its_argv(client, checkout):
    run = client.post(
        "/scientist", headers = {"x-test-account": "alice"}, json = _payload(loadCode = True)
    ).json()
    assert run["status"] == "queued"
    assert run["config"]["allowProviderCalls"] is True
    assert "--load-code" in run["argv"]
    assert run["config"]["loadCode"] is True
    assert run["metrics"] is None


def test_listing_is_scoped_to_the_caller(client, checkout):
    client.post("/scientist", headers = {"x-test-account": "alice"}, json = _payload())
    client.post("/scientist", headers = {"x-test-account": "bob"}, json = _payload())
    alice = client.get("/scientist", headers = {"x-test-account": "alice"}).json()
    bob = client.get("/scientist", headers = {"x-test-account": "bob"}).json()
    assert alice["total"] == 1
    assert bob["total"] == 1
    assert alice["runs"][0]["id"] != bob["runs"][0]["id"]


def test_listing_rejects_an_unknown_status(client, checkout):
    response = client.get("/scientist?status=imaginary", headers = {"x-test-account": "alice"})
    assert response.status_code == 400


def test_a_missing_run_is_a_404(client, checkout):
    assert client.get("/scientist/nope", headers = {"x-test-account": "alice"}).status_code == 404


def test_cancelling_a_queued_run_ends_it_immediately(client, checkout):
    run_id = client.post("/scientist", headers = {"x-test-account": "alice"}, json = _payload()).json()[
        "id"
    ]
    cancelled = client.post(
        f"/scientist/{run_id}/cancel", headers = {"x-test-account": "alice"}
    ).json()
    # Nothing has claimed it, so there is no child to reap and the row can go straight
    # to its terminal state rather than sitting in `cancelling` forever.
    assert cancelled["status"] == "cancelled"
    assert cancelled["cancelRequested"] is True


def test_cancelling_a_running_run_waits_for_the_worker(client, checkout):
    run_id = _claim(client, "alice")
    cancelling = client.post(
        f"/scientist/{run_id}/cancel", headers = {"x-test-account": "alice"}
    ).json()
    assert cancelling["status"] == "cancelling"
    assert db.finish(run_id, WORKER, "completed") == "cancelled"


def test_retry_requeues_a_finished_run_and_clears_its_result(client, checkout):
    run_id = _claim(client, "alice")
    db.finish(run_id, WORKER, "failed", "boom", {"val_bpb": 1.0}, {"loss": [1.0]})
    retried = client.post(f"/scientist/{run_id}/retry", headers = {"x-test-account": "alice"}).json()
    assert retried["status"] == "queued"
    assert retried["error"] is None
    assert retried["metrics"] is None
    assert retried["curves"] is None
    assert retried["retryCount"] == 1


def test_retry_refuses_a_run_that_is_still_going(client, checkout):
    run_id = _claim(client, "alice")
    response = client.post(f"/scientist/{run_id}/retry", headers = {"x-test-account": "alice"})
    assert response.status_code == 409


def test_retry_has_a_budget(client, checkout):
    run_id = _claim(client, "alice")
    db.finish(run_id, WORKER, "failed", "boom")
    for attempt in range(db.MAX_RETRIES):
        assert (
            client.post(
                f"/scientist/{run_id}/retry", headers = {"x-test-account": "alice"}
            ).status_code
            == 200
        )
        db.claim_next(WORKER)
        db.finish(run_id, WORKER, "failed", "boom")
    response = client.post(f"/scientist/{run_id}/retry", headers = {"x-test-account": "alice"})
    assert response.status_code == 409
    assert "budget" in response.json()["detail"]


def test_the_environment_endpoint_explains_a_broken_install(client, checkout):
    payload = client.get("/scientist/environment", headers = {"x-test-account": "alice"}).json()
    assert payload["available"] is True
    assert payload["projectRoot"] == str(checkout.resolve())


def test_logs_read_back_the_streamed_stdout(client, checkout):
    run_id = _claim(client, "alice")
    db.append_log(run_id, WORKER, "line one\n")
    db.append_log(run_id, WORKER, "line two\n")
    payload = client.get(f"/scientist/{run_id}/logs", headers = {"x-test-account": "alice"}).json()
    assert payload["text"] == "line one\nline two\n"
    assert payload["truncated"] is False
    db.finish(run_id, WORKER, "completed")


def test_logs_honour_the_cursor(client, checkout):
    run_id = _claim(client, "alice")
    db.append_log(run_id, WORKER, "first\n")
    last = db.list_events(run_id)[-1]["seq"]
    db.append_log(run_id, WORKER, "second\n")
    payload = client.get(
        f"/scientist/{run_id}/logs?after={last}", headers = {"x-test-account": "alice"}
    ).json()
    assert payload["text"] == "second\n"
    db.finish(run_id, WORKER, "completed")


def test_the_event_stream_replays_then_ends_on_a_terminal_run(client, checkout):
    run_id = _claim(client, "alice")
    db.append_worker_event(run_id, WORKER, "run.log", {"text": "hello\n"})
    db.finish(run_id, WORKER, "completed", None, {"val_bpb": 0.9})
    with client.stream(
        "GET", f"/scientist/{run_id}/events", headers = {"x-test-account": "alice"}
    ) as response:
        body = "".join(response.iter_text())
    assert "event: run.log" in body
    assert "event: run.completed" in body
    # The terminal frame carries the run, so a client that missed a poll still learns
    # the final status and its metrics.
    final = [line for line in body.splitlines() if line.startswith("data: ")][-1]
    assert json.loads(final[6:])["run"]["metrics"] == {"val_bpb": 0.9}


def test_the_event_stream_accepts_a_last_event_id(client, checkout):
    run_id = _claim(client, "alice")
    db.append_log(run_id, WORKER, "old\n")
    cursor = db.list_events(run_id)[-1]["seq"]
    db.append_log(run_id, WORKER, "new\n")
    db.finish(run_id, WORKER, "completed")
    with client.stream(
        "POST",
        f"/scientist/{run_id}/events",
        headers = {"x-test-account": "alice", "Last-Event-ID": str(cursor)},
    ) as response:
        body = "".join(response.iter_text())
    assert "old" not in body
    assert "new" in body


# ── storage ─────────────────────────────────────────────────────────────────────


def _claim(client, username: str) -> str:
    run_id = client.post(
        "/scientist", headers = {"x-test-account": username}, json = _payload()
    ).json()["id"]
    claimed = run_as(ACCOUNTS[username], db.claim_next, WORKER)
    assert claimed is not None
    return run_id


def test_claiming_takes_a_lease_and_a_queued_run_is_only_claimed_once(client, checkout):
    run_id = _claim(client, "alice")
    assert run_as(ALICE, db.claim_next, "worker-2") is None
    assert db.owns_lease(run_id, WORKER) is True
    assert db.owns_lease(run_id, "worker-2") is False


def test_a_heartbeat_extends_the_lease_and_a_stale_one_does_not(client, checkout):
    run_id = _claim(client, "alice")
    assert db.heartbeat(run_id, WORKER) is True
    assert db.heartbeat(run_id, "worker-2") is False
    # An expired lease stops counting, so a second worker can take the run over.
    conn = db.get_connection()
    conn.execute(
        "UPDATE scientist_runs SET lease_expires_at = ? WHERE id = ?", (db.now_ms() - 1, run_id)
    )
    conn.commit()
    conn.close()
    assert db.owns_lease(run_id, WORKER) is False
    assert run_as(ALICE, db.claim_next, "worker-2")["id"] == run_id


def test_a_reclaimed_run_drops_the_dead_workers_log(client, checkout):
    run_id = _claim(client, "alice")
    db.append_log(run_id, WORKER, "from the first worker\n")
    conn = db.get_connection()
    conn.execute(
        "UPDATE scientist_runs SET lease_expires_at = ? WHERE id = ?", (db.now_ms() - 1, run_id)
    )
    conn.commit()
    conn.close()
    reclaimed = run_as(ALICE, db.claim_next, "worker-2")
    assert reclaimed["claimedFromStatus"] == "running"
    # The old child is gone, so its output would otherwise read as this attempt's.
    assert db.read_log(run_id) == ""
    assert db.get_run(run_id)["logBytes"] == 0


def test_a_worker_without_the_lease_cannot_write(client, checkout):
    run_id = _claim(client, "alice")
    assert db.append_log(run_id, "worker-2", "not mine\n") is None
    assert db.append_worker_event(run_id, "worker-2", "run.note", {"message": "no"}) is None
    # The lease holder can, and the rejected writes left nothing behind.
    assert db.append_log(run_id, WORKER, "mine\n") is not None
    assert db.read_log(run_id) == "mine\n"
    assert [
        event["data"].get("message")
        for event in db.list_events(run_id)
        if event["type"] == "run.note"
    ] == []


def test_a_cancel_requested_run_stops_accepting_worker_writes(client, checkout):
    run_id = _claim(client, "alice")
    db.request_cancel(run_id)
    assert db.append_log(run_id, WORKER, "too late\n") is None


def test_events_are_numbered_from_one_with_no_gaps(client, checkout):
    run_id = _claim(client, "alice")
    for index in range(4):
        db.append_log(run_id, WORKER, f"line {index}\n")
    seqs = [event["seq"] for event in db.list_events(run_id)]
    assert seqs == [1, 2, 3, 4, 5, 6]
    assert db.get_run(run_id)["lastEventSeq"] == 6


def test_the_log_is_capped(client, checkout, monkeypatch):
    monkeypatch.setattr(db, "MAX_LOG_BYTES", 64)
    run_id = _claim(client, "alice")
    for _ in range(20):
        db.append_log(run_id, WORKER, "x" * 32 + "\n")
    run = db.get_run(run_id)
    assert run["logBytes"] == 64
    assert run["logTruncated"] is True
    # Past the cap a run keeps its status readable and simply stops storing output,
    # rather than growing the studio database for the life of the install.
    assert db.append_log(run_id, WORKER, "more\n") is None


def test_finishing_requires_the_lease_and_a_live_one(client, checkout):
    run_id = _claim(client, "alice")
    assert db.finish(run_id, "worker-2", "completed") is None
    conn = db.get_connection()
    conn.execute(
        "UPDATE scientist_runs SET lease_expires_at = ? WHERE id = ?", (db.now_ms() - 1, run_id)
    )
    conn.commit()
    conn.close()
    assert db.finish(run_id, WORKER, "completed") is None
    assert db.finish(run_id, WORKER, "completed", allow_expired = True) == "completed"


def test_a_cancelled_run_stores_no_result_and_no_error(client, checkout):
    run_id = _claim(client, "alice")
    db.request_cancel(run_id)
    assert db.finish(run_id, WORKER, "completed", "ignored", {"val_bpb": 1.0}) == "cancelled"
    run = db.get_run(run_id)
    assert run["error"] is None
    assert run["metrics"] is None


def test_recover_expired_frees_a_stalled_run(client, checkout):
    run_id = _claim(client, "alice")
    conn = db.get_connection()
    conn.execute(
        "UPDATE scientist_runs SET lease_expires_at = ? WHERE id = ?", (db.now_ms() - 1, run_id)
    )
    conn.commit()
    conn.close()
    assert run_as(ALICE, db.recover_expired) == 1
    assert run_as(ALICE, db.recover_expired) == 0
    assert run_as(ALICE, db.claim_next, "worker-2")["id"] == run_id


def test_releasing_leases_leaves_the_run_claimable(client, checkout):
    _claim(client, "alice")
    assert run_as(ALICE, db.release_worker_leases, WORKER) == 1
    assert db.owns_lease(db.list_active("alice")[0]["id"], WORKER) is False


def test_a_run_waits_for_events_rather_than_polling(client, checkout):
    import threading

    run_id = _claim(client, "alice")
    cursor = db.get_run(run_id)["lastEventSeq"]

    def publish():
        time.sleep(0.05)
        # A new thread does not inherit the bound account, so it has to say whose
        # database it is writing to, exactly as the supervisor's own threads do.
        run_as(ALICE, db.append_log, run_id, WORKER, "late\n")

    threading.Thread(target = publish, daemon = True).start()
    events = db.wait_for_events(run_id, cursor, timeout = 5)
    assert [event["data"].get("text") for event in events] == ["late\n"]
    db.finish(run_id, WORKER, "completed")


def test_a_second_account_never_sees_a_removed_run(client, checkout):
    run_id = _claim(client, "alice")
    conn = db.get_connection()
    conn.execute("DELETE FROM scientist_runs WHERE id = ?", (run_id,))
    conn.commit()
    conn.close()
    assert run_as(BOB, db.get_run, run_id) is None
    assert (
        client.get(f"/scientist/{run_id}", headers = {"x-test-account": "alice"}).status_code == 404
    )


def test_an_event_for_a_run_that_is_gone_is_refused(client, checkout):
    run_id = _claim(client, "alice")
    conn = db.get_connection()
    conn.execute("DELETE FROM scientist_runs WHERE id = ?", (run_id,))
    conn.commit()
    conn.close()
    with pytest.raises(KeyError):
        db.append_event(run_id, "run.note", {})


def test_a_retired_account_cannot_open_the_connection(monkeypatch, client, checkout):
    monkeypatch.setattr(db, "account_is_retired", lambda: True)
    with pytest.raises(RuntimeError):
        db.get_connection()


# ── artifact reading ────────────────────────────────────────────────────────────


def test_the_idea_directory_is_discovered_once_the_launcher_makes_it(tmp_path):
    staging = tmp_path / "studio-run"
    staging.mkdir()
    assert scientist_runs.discover_idea_dir(staging) is None
    (staging / "2026-01-01_idea_attempt_0").mkdir()
    assert scientist_runs.discover_idea_dir(staging).name == "2026-01-01_idea_attempt_0"


def test_results_are_read_from_the_newest_artifact(tmp_path):
    staging = tmp_path / "studio-run"
    older = staging / "node_1"
    newer = staging / "node_2"
    older.mkdir(parents = True)
    newer.mkdir(parents = True)
    (older / "results.json").write_text(json.dumps({"metrics": {"val_bpb": 1.5}}), encoding = "utf-8")
    payload = {
        "primary_metric": {"name": "val_bpb", "value": 0.9, "lower_is_better": True},
        "metrics": {"val_bpb": 0.9, "core_metric": 0.3},
        "curves": {"val_bpb": [1.2, 1.0, 0.9]},
    }
    (newer / "results.json").write_text(json.dumps(payload), encoding = "utf-8")
    os.utime(older / "results.json", (0, 0))
    metrics, curves = scientist_runs.collect_results(staging)
    assert metrics["val_bpb"] == 0.9
    assert metrics["primary_metric"]["name"] == "val_bpb"
    assert curves == {"val_bpb": [1.2, 1.0, 0.9]}


def test_a_run_with_no_result_artifact_reports_no_metrics(tmp_path):
    staging = tmp_path / "studio-run"
    staging.mkdir()
    assert scientist_runs.collect_results(staging) == (None, None)


def test_non_finite_metrics_are_dropped_rather_than_stored(tmp_path):
    # json.dumps(allow_nan=False) raises on NaN, and a run's terminal write must not
    # die on a metric the trainer happened to report.
    staging = tmp_path / "studio-run"
    staging.mkdir()
    (staging / "results.json").write_text(
        '{"metrics": {"val_bpb": NaN, "core_metric": 0.4}}', encoding = "utf-8"
    )
    metrics, _ = scientist_runs.collect_results(staging)
    assert "val_bpb" not in metrics
    assert metrics["core_metric"] == 0.4
    json.dumps(metrics, allow_nan = False)


def test_curves_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(scientist_runs, "MAX_CURVE_POINTS", 3)
    monkeypatch.setattr(scientist_runs, "MAX_CURVE_SERIES", 1)
    staging = tmp_path / "studio-run"
    staging.mkdir()
    (staging / "results.json").write_text(
        json.dumps({"curves": {"a": [1, 2, 3, 4], "b": [5, 6]}}), encoding = "utf-8"
    )
    _, curves = scientist_runs.collect_results(staging)
    assert curves == {"a": [1.0, 2.0, 3.0]}


def test_the_final_status_comes_from_the_launcher_not_the_exit_code(tmp_path):
    idea = tmp_path / "idea"
    idea.mkdir()
    assert scientist_runs._final_status(idea) is None
    (idea / "run_status.json").write_text(json.dumps({"status": "complete"}), encoding = "utf-8")
    assert scientist_runs._final_status(idea) == "complete"
    assert scientist_runs._final_status(None) is None


def test_the_timeout_is_clamped_to_the_launchers_range():
    assert scientist_runs._timeout_for({}) == scientist_runs.MAX_TIMEOUT_SECONDS
    assert scientist_runs._timeout_for({"timeoutSeconds": 1}) == scientist_runs.MIN_TIMEOUT_SECONDS
    assert (
        scientist_runs._timeout_for({"timeoutSeconds": 10**9}) == scientist_runs.MAX_TIMEOUT_SECONDS
    )
    assert scientist_runs._timeout_for({"timeoutSeconds": "nonsense"}) == (
        scientist_runs.MAX_TIMEOUT_SECONDS
    )
    assert scientist_runs._timeout_for({"timeoutSeconds": 600}) == 600


def test_a_staging_directory_is_reset_so_a_retry_starts_clean(tmp_path):
    staging = tmp_path / "studio-run"
    staging.mkdir()
    (staging / "stale").mkdir()
    scientist_runs._reset_staging(staging)
    assert staging.is_dir()
    assert list(staging.iterdir()) == []


def test_the_supervisor_wakes_the_suspender_for_a_cancel(client, checkout):
    seen: list[str] = []

    class Supervisor:
        def wake(self):
            seen.append("wake")

        def cancel(self, run_id: str):
            seen.append(run_id)

    client.app.state.scientist_supervisor = Supervisor()
    run_id = _claim(client, "alice")
    client.post(f"/scientist/{run_id}/cancel", headers = {"x-test-account": "alice"})
    assert run_id in seen


def test_a_create_asks_the_supervisor_to_wake(client, checkout):
    seen: list[str] = []

    class Supervisor:
        def wake(self):
            seen.append("wake")

    client.app.state.scientist_supervisor = Supervisor()
    client.post("/scientist", headers = {"x-test-account": "alice"}, json = _payload())
    assert seen == ["wake"]


def test_the_schema_is_created_by_the_shared_initializer(client, checkout):
    client.post("/scientist", headers = {"x-test-account": "alice"}, json = _payload())
    conn = db.get_connection()
    names = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    conn.close()
    assert {"scientist_runs", "scientist_events"} <= names


def test_a_bad_status_in_a_rejected_run_never_reaches_the_table(client, checkout):
    with pytest.raises(sqlite3.IntegrityError):
        conn = db.get_connection()
        try:
            conn.execute(
                "INSERT INTO scientist_runs (id, owner_subject, status, config_json, argv_json, "
                "created_at, updated_at) VALUES ('x', 'y', 'imaginary', '{}', '[]', 0, 0)"
            )
            conn.commit()
        finally:
            conn.close()
