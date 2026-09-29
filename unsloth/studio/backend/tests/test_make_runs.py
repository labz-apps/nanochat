# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Make runs: the speedrun translation, the storage lease, and the HTTP surface.

The translation is the part worth pinning. ``runs/speedrun.sh`` is written for a blank
8xH100 node; running it on one consumer GPU fails on ``--fp8`` and on a device-batch size
that assumes eight devices. These tests exist so that translation cannot quietly regress
back into "just run the script".
"""

import json
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import authenticated_via_api_key, get_current_subject
from core import make_runs
from routes import make_runs as routes
from storage import make_runs_db as db
from storage import studio_db
from utils.account_context import AccountContext, bind_account, reset_account, run_as

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")
ACCOUNTS = {account.username: account for account in (ALICE, BOB)}
WORKER = "worker-1"


@pytest.fixture(autouse = True)
def _reset_schemas(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    # No schema of its own: it reads the one storage.studio_db creates.
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    yield


@pytest.fixture(autouse = True)
def alice():
    """Bind alice for the test body.

    Managed accounts each own a database, and TestClient serves the app on its own
    thread, so a direct ``db.`` call has to be under the same context the request was.
    Autouse so no test can quietly assert against the owner's empty database.
    """
    marker = bind_account(ALICE)
    try:
        yield ALICE
    finally:
        reset_account(marker)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A checkout complete enough for the environment probe to pass."""
    root = tmp_path / "nanochat"
    (root / "nanochat").mkdir(parents = True)
    (root / "scripts").mkdir(parents = True)
    (root / "launch_scientist_bfts.py").write_text("# launcher\n", encoding = "utf-8")
    (root / "nanochat" / "__init__.py").write_text("", encoding = "utf-8")
    (root / "scripts" / "__init__.py").write_text("", encoding = "utf-8")
    (root / "scripts" / "base_train.py").write_text("", encoding = "utf-8")
    cache = tmp_path / "cache"
    (cache / "tokenizer").mkdir(parents = True)
    (cache / "base_data_climbmix").mkdir(parents = True)
    (cache / "eval_bundle").mkdir(parents = True)
    for name in ("tokenizer.pkl", "token_bytes.pt"):
        (cache / "tokenizer" / name).write_bytes(b"\x00")
    (cache / "base_data_climbmix" / "shard_06542.parquet").write_bytes(b"\x00")
    monkeypatch.setenv(make_runs._PROJECT_ROOT_ENV, str(root))
    monkeypatch.setenv(make_runs._PYTHON_ENV, sys.executable)
    monkeypatch.setenv(make_runs._SHARED_CACHE_ENV, str(cache))
    # The real probe runs an interpreter, which a temp checkout cannot satisfy. What it
    # decides is covered on its own below.
    monkeypatch.setattr(make_runs, "interpreter_can_import", lambda root, python: True)
    return root


def _payload(**overrides):
    base = {"preset": "small"}
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
    app.include_router(routes.router, prefix = "/make")
    app.state.make_supervisor = None
    with TestClient(app) as test_client:
        yield test_client


def _claim(client, username: str = "alice") -> str:
    run_id = client.post(
        "/make", headers = {"x-test-account": username}, json = _payload()
    ).json()["id"]
    assert run_as(ACCOUNTS[username], db.claim_next, WORKER) is not None
    return run_id


# ── the speedrun translation ────────────────────────────────────────────────────


def test_the_argv_is_single_device_and_has_no_fp8(checkout):
    config = make_runs.normalize_config({"preset": "small"})
    argv = make_runs.build_argv(config)
    # These two are what make runs/speedrun.sh unusable here, and both have to stay out.
    assert "--fp8" not in argv
    assert not any("nproc_per_node" in token for token in argv)
    assert "torchrun" not in argv
    assert argv[:2] == ["-m", "scripts.base_train"]


def test_the_iteration_budget_is_pinned(checkout):
    # Left at nanochat's -1, the count is derived from a FLOPs budget sized for eight
    # H100s. Pinning it is the whole "shorter than a speedrun" requirement.
    for preset in make_runs.PRESETS:
        config = make_runs.normalize_config({"preset": preset})
        argv = make_runs.build_argv(config)
        assert f"--num-iterations={config['iterations']}" in argv
        assert make_runs.PRESETS[preset]["iterations"] > 0


def test_presets_get_shorter_as_they_get_smaller(checkout):
    order = ["tiny", "small", "medium"]
    iterations = [make_runs.PRESETS[p]["iterations"] for p in order]
    depths = [make_runs.PRESETS[p]["depth"] for p in order]
    assert iterations == sorted(iterations)
    assert depths == sorted(depths)
    # Every preset must be shorter than the depth-24 speedrun it is derived from.
    assert max(iterations) < 20_000


def test_every_preset_is_internally_consistent(checkout):
    for preset, values in make_runs.PRESETS.items():
        config = make_runs.normalize_config({"preset": preset})
        assert config["totalBatchSize"] % config["deviceBatchSize"] == 0, preset
        assert config["maxSeqLen"] % 128 == 0, preset
        assert config["depth"] <= make_runs.MAX_DEPTH, preset
        # A run tag is a directory name; an empty one would collide with the root.
        assert config["runTag"], preset


def test_an_override_replaces_the_preset_value(checkout):
    config = make_runs.normalize_config({"preset": "small", "depth": 5})
    assert config["depth"] == 5
    # And leaves the rest of the preset alone.
    assert config["iterations"] == make_runs.PRESETS["small"]["iterations"]


def test_an_out_of_range_override_is_refused_rather_than_clamped(checkout):
    # A silently clamped value is a run that trains something other than what the dialog
    # said, and the only sign is a loss curve that does not match the last one.
    for bad in (
        {"depth": 99},
        {"iterations": 0},
        {"deviceBatchSize": 0},
        {"maxSeqLen": 100},
        # totalBatchSize is tokens per step, not sequences, so it has its own range.
        {"totalBatchSize": 200},
    ):
        with pytest.raises(ValueError):
            make_runs.normalize_config({"preset": "small", **bad})


def test_the_two_batch_sizes_have_different_ranges(checkout):
    # The regression this guards: sharing one bound caps a real run at 256 tokens.
    config = make_runs.normalize_config({"preset": "small"})
    assert config["deviceBatchSize"] <= make_runs.MAX_DEVICE_BATCH
    assert config["totalBatchSize"] > make_runs.MAX_DEVICE_BATCH
    assert config["totalBatchSize"] <= make_runs.MAX_TOTAL_BATCH


def test_a_total_batch_that_is_not_a_multiple_is_refused(checkout):
    # In range, but not a whole number of accumulation steps.
    with pytest.raises(ValueError, match = "multiple"):
        make_runs.normalize_config(
            {"preset": "small", "deviceBatchSize": 64, "totalBatchSize": 33_000}
        )


def test_an_unknown_preset_is_refused(checkout):
    with pytest.raises(ValueError, match = "Unknown preset"):
        make_runs.normalize_config({"preset": "enormous"})


def test_a_run_tag_cannot_become_a_path(checkout):
    for raw in ("../../escape", "a/b", "..", "", "with spaces"):
        config = make_runs.normalize_config({"preset": "tiny", "runTag": raw})
        tag = config["runTag"]
        assert "/" not in tag and "\\" not in tag
        assert tag not in {"", ".", ".."}
        assert Path(tag).name == tag


def test_evaluation_is_opt_out(checkout):
    assert "--results-json=results.json" in make_runs.build_argv(
        make_runs.normalize_config({"preset": "tiny"})
    )
    assert "--results-json=results.json" not in make_runs.build_argv(
        make_runs.normalize_config({"preset": "tiny", "evaluate": False})
    )


def test_the_child_environment_is_an_allowlist_with_no_credentials(checkout):
    import os

    os.environ["OPENROUTER_API_KEY"] = "sk-or-should-not-be-here"
    try:
        env = make_runs.environment()
        child = make_runs.build_child_env(env)
        # Pretraining calls no provider, so a key reaching the child is a key the run
        # had no reason to hold.
        assert "OPENROUTER_API_KEY" not in child
        # And the trainer must resolve the same cache the probe validated, not a
        # different one under a different home.
        assert child["NANOCHAT_BASE_DIR"] == env["sharedCache"]
        assert child["NANOCHAT_SHARED_CACHE"] == env["sharedCache"]
    finally:
        del os.environ["OPENROUTER_API_KEY"]


def test_the_checkpoint_directory_is_derived_not_requested(checkout):
    env = make_runs.environment()
    # A run tag of "../.." must not become a path: it is joined onto a base we chose.
    config = make_runs.normalize_config({"preset": "tiny", "runTag": "../../escape"})
    checkpoint = Path(env["checkpointRoot"]) / config["runTag"]
    assert Path(env["checkpointRoot"]) in checkpoint.parents


# ── environment probe ───────────────────────────────────────────────────────────


def test_environment_reports_a_usable_checkout(checkout):
    env = make_runs.environment()
    assert env["available"] is True
    assert env["problems"] == []
    assert Path(env["projectRoot"]) == checkout.resolve()
    assert env["python"] == sys.executable
    assert {preset["id"] for preset in env["presets"]} == set(make_runs.PRESETS)


def test_environment_reports_a_missing_checkout(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "studio" / "backend" / "core").mkdir(parents = True)
    monkeypatch.setenv(make_runs._PROJECT_ROOT_ENV, str(elsewhere))
    monkeypatch.setattr(make_runs, "__file__", str(elsewhere / "studio" / "backend" / "core" / "x.py"))
    env = make_runs.environment()
    assert env["available"] is False
    assert "no-project-root" in {problem["code"] for problem in env["problems"]}


def test_environment_names_each_missing_asset(checkout, monkeypatch):
    cache = Path(make_runs.environment()["sharedCache"])
    (cache / "tokenizer" / "tokenizer.pkl").unlink()
    (cache / "eval_bundle" / "x").write_text("", encoding = "utf-8")
    (cache / "eval_bundle" / "x").unlink()
    (cache / "eval_bundle").rmdir()
    env = make_runs.environment()
    problem = next(p for p in env["problems"] if p["code"] == "incomplete-shared-cache")
    assert "tokenizer" in problem["message"]
    assert "evaluation bundle" in problem["message"]


def test_cache_missing_assets_reports_only_what_is_absent(checkout):
    cache = Path(make_runs.environment()["sharedCache"])
    assert make_runs.cache_missing_assets(cache) == []
    (cache / "tokenizer" / "token_bytes.pt").unlink()
    assert make_runs.cache_missing_assets(cache) == ["tokenizer"]


def test_interpreter_can_import_really_runs_the_interpreter(tmp_path):
    # Not stubbed: this is the check that decides whether a real install can pretrain, so
    # a stubbed version would only prove the stub returns what it was told.
    root = tmp_path / "checkout"
    (root / "nanochat").mkdir(parents = True)
    assert make_runs.interpreter_can_import(root, sys.executable) is False

    (root / "nanochat" / "__init__.py").write_text("", encoding = "utf-8")
    (root / "nanochat" / "common.py").write_text("VAL = 1\n", encoding = "utf-8")
    (root / "scripts").mkdir()
    (root / "scripts" / "__init__.py").write_text("", encoding = "utf-8")
    (root / "scripts" / "base_train.py").write_text("", encoding = "utf-8")
    assert make_runs.interpreter_can_import(root, sys.executable) is True

    # Half a checkout is still a no: the probe imports all three, and a package with only
    # an __init__ is exactly the state an interrupted checkout is in.
    (root / "nanochat" / "common.py").unlink()
    assert make_runs.interpreter_can_import(root, sys.executable) is False


def test_an_unusable_interpreter_is_reported_rather_than_raising(tmp_path):
    # A path that is not an interpreter at all must not take the page down with it.
    missing = tmp_path / "not-python"
    missing.write_text("", encoding = "utf-8")
    assert make_runs.interpreter_can_import(tmp_path, str(missing)) is False
    assert make_runs.interpreter_can_import(tmp_path, str(tmp_path / "absent")) is False


# ── progress and results ────────────────────────────────────────────────────────


def test_progress_is_parsed_from_the_trainers_own_line():
    lines = ["noise\n", "step 42 num_params=1234 val_bpb=1.234 train_loss=2.5\n"]
    progress = make_runs._progress_from_log(lines)
    assert progress["num_params"] == 1234
    assert progress["val_bpb"] == pytest.approx(1.234)
    assert progress["train_loss"] == pytest.approx(2.5)


def test_progress_ignores_lines_it_cannot_read():
    assert make_runs._progress_from_log(["nothing here\n", "still nothing\n"]) is None
    assert make_runs._progress_from_log([]) is None


def test_results_are_read_from_the_checkpoint_directory(tmp_path):
    checkpoint = tmp_path / "base_checkpoints" / "make"
    checkpoint.mkdir(parents = True)
    (checkpoint / "results.json").write_text(
        json.dumps({"metrics": {"val_bpb": 1.2, "core_metric": 0.3}}), encoding = "utf-8"
    )
    metrics = make_runs.collect_results(checkpoint)
    assert metrics["val_bpb"] == 1.2


def test_a_run_with_no_checkpoint_reports_no_metrics(tmp_path):
    assert make_runs.collect_results(None) is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert make_runs.collect_results(empty) is None


def test_non_finite_metrics_are_dropped_rather_than_stored(tmp_path):
    checkpoint = tmp_path / "cp"
    checkpoint.mkdir()
    (checkpoint / "results.json").write_text(
        '{"metrics": {"val_bpb": NaN, "core_metric": 0.4}}', encoding = "utf-8"
    )
    metrics = make_runs.collect_results(checkpoint)
    assert "val_bpb" not in metrics
    assert metrics["core_metric"] == 0.4
    json.dumps(metrics, allow_nan = False)


# ── the run surface ─────────────────────────────────────────────────────────────


def test_a_created_run_is_queued_and_records_its_argv(client, checkout):
    run = client.post(
        "/make", headers = {"x-test-account": "alice"}, json = _payload(depth = 5)
    ).json()
    assert run["status"] == "queued"
    assert run["preset"] == "small"
    assert run["config"]["depth"] == 5
    assert "--num-iterations=300" in run["argv"]


def test_a_bad_override_is_a_400_with_the_reason(client, checkout):
    response = client.post(
        "/make",
        headers = {"x-test-account": "alice"},
        json = _payload(deviceBatchSize = 64, totalBatchSize = 33_000),
    )
    assert response.status_code == 400
    assert "multiple" in response.json()["detail"]


def test_an_unavailable_checkout_refuses_the_run_with_503(client, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "studio" / "backend" / "core").mkdir(parents = True)
    monkeypatch.setenv(make_runs._PROJECT_ROOT_ENV, str(elsewhere))
    monkeypatch.setattr(make_runs, "__file__", str(elsewhere / "studio" / "backend" / "core" / "x.py"))
    response = client.post("/make", headers = {"x-test-account": "alice"}, json = _payload())
    assert response.status_code == 503


def test_an_unknown_field_is_refused(client, checkout):
    response = client.post(
        "/make",
        headers = {"x-test-account": "alice"},
        json = {**_payload(), "argv": ["rm", "-rf"]},
    )
    assert response.status_code == 422


def test_a_second_concurrent_run_is_refused(client, checkout):
    assert client.post("/make", headers = {"x-test-account": "alice"}, json = _payload()).status_code == 202
    second = client.post("/make", headers = {"x-test-account": "alice"}, json = _payload())
    assert second.status_code == 409
    assert "already in progress" in second.json()["detail"]


def test_listing_is_scoped_to_the_caller(client, checkout):
    client.post("/make", headers = {"x-test-account": "alice"}, json = _payload())
    client.post("/make", headers = {"x-test-account": "bob"}, json = _payload())
    alice = client.get("/make", headers = {"x-test-account": "alice"}).json()
    bob = client.get("/make", headers = {"x-test-account": "bob"}).json()
    assert alice["total"] == 1 and bob["total"] == 1
    assert alice["runs"][0]["id"] != bob["runs"][0]["id"]


def test_a_missing_run_is_a_404(client, checkout):
    assert client.get("/make/nope", headers = {"x-test-account": "alice"}).status_code == 404


def test_cancelling_a_queued_run_ends_it_immediately(client, checkout):
    # Not claimed: a queued run has no child to reap, so it goes straight to cancelled
    # rather than sitting in `cancelling` waiting for a worker that will never see it.
    run_id = client.post(
        "/make", headers = {"x-test-account": "alice"}, json = _payload()
    ).json()["id"]
    cancelled = client.post(f"/make/{run_id}/cancel", headers = {"x-test-account": "alice"}).json()
    assert cancelled["status"] == "cancelled"
    assert cancelled["cancelRequested"] is True


def test_cancelling_a_running_run_waits_for_the_worker(client, checkout):
    run_id = _claim(client)
    cancelling = client.post(f"/make/{run_id}/cancel", headers = {"x-test-account": "alice"}).json()
    assert cancelling["status"] == "cancelling"
    assert db.finish(run_id, WORKER, "completed") == "cancelled"


def test_retry_requeues_a_finished_run_and_clears_its_result(client, checkout):
    run_id = _claim(client)
    db.finish(run_id, WORKER, "failed", "boom", {"val_bpb": 1.0})
    retried = client.post(f"/make/{run_id}/retry", headers = {"x-test-account": "alice"}).json()
    assert retried["status"] == "queued"
    assert retried["error"] is None
    assert retried["metrics"] is None
    assert retried["checkpointDir"] is None
    assert retried["retryCount"] == 1


def test_retry_refuses_a_run_that_is_still_going(client, checkout):
    run_id = _claim(client)
    assert client.post(f"/make/{run_id}/retry", headers = {"x-test-account": "alice"}).status_code == 409


def test_retry_has_a_budget(client, checkout):
    run_id = _claim(client)
    db.finish(run_id, WORKER, "failed", "boom")
    for _ in range(db.MAX_RETRIES):
        assert client.post(f"/make/{run_id}/retry", headers = {"x-test-account": "alice"}).status_code == 200
        db.claim_next(WORKER)
        db.finish(run_id, WORKER, "failed", "boom")
    response = client.post(f"/make/{run_id}/retry", headers = {"x-test-account": "alice"})
    assert response.status_code == 409
    assert "budget" in response.json()["detail"]


def test_the_presets_endpoint_lists_every_preset(client, checkout):
    payload = client.get("/make/presets", headers = {"x-test-account": "alice"}).json()
    assert {preset["id"] for preset in payload["presets"]} == set(make_runs.PRESETS)


def test_logs_read_back_the_streamed_stdout(client, checkout):
    run_id = _claim(client)
    db.append_log(run_id, WORKER, "step 1\n")
    db.append_log(run_id, WORKER, "step 2\n")
    payload = client.get(f"/make/{run_id}/logs", headers = {"x-test-account": "alice"}).json()
    assert payload["text"] == "step 1\nstep 2\n"
    db.finish(run_id, WORKER, "completed")


def test_the_event_stream_replays_then_ends_on_a_terminal_run(client, checkout):
    run_id = _claim(client)
    db.append_worker_event(run_id, WORKER, "run.log", {"text": "training\n"})
    db.finish(run_id, WORKER, "completed", None, {"val_bpb": 0.9})
    with client.stream(
        "GET", f"/make/{run_id}/events", headers = {"x-test-account": "alice"}
    ) as response:
        body = "".join(response.iter_text())
    assert "event: run.log" in body
    assert "event: run.completed" in body
    final = [line for line in body.splitlines() if line.startswith("data: ")][-1]
    assert json.loads(final[6:])["run"]["metrics"] == {"val_bpb": 0.9}


# ── storage ─────────────────────────────────────────────────────────────────────


def test_claiming_takes_a_lease_and_a_queued_run_is_only_claimed_once(client, checkout):
    run_id = _claim(client)
    assert run_as(ALICE, db.claim_next, "worker-2") is None
    assert db.owns_lease(run_id, WORKER) is True
    assert db.owns_lease(run_id, "worker-2") is False


def test_a_heartbeat_extends_the_lease(client, checkout):
    run_id = _claim(client)
    assert db.heartbeat(run_id, WORKER) is True
    assert db.heartbeat(run_id, "worker-2") is False


def test_a_reclaimed_run_drops_the_dead_workers_log(client, checkout):
    run_id = _claim(client)
    db.append_log(run_id, WORKER, "from the first worker\n")
    _expire(run_id)
    reclaimed = run_as(ALICE, db.claim_next, "worker-2")
    assert reclaimed["claimedFromStatus"] == "running"
    # The old child is gone, so its output would otherwise read as this attempt's.
    assert db.read_log(run_id) == ""
    assert db.get_run(run_id)["logBytes"] == 0


def test_a_worker_without_the_lease_cannot_write(client, checkout):
    run_id = _claim(client)
    assert db.append_log(run_id, "worker-2", "not mine\n") is None
    assert db.append_worker_event(run_id, "worker-2", "run.note", {"message": "no"}) is None
    assert db.append_log(run_id, WORKER, "mine\n") is not None
    assert db.read_log(run_id) == "mine\n"


def test_a_cancel_requested_run_stops_accepting_worker_writes(client, checkout):
    run_id = _claim(client)
    db.request_cancel(run_id)
    assert db.append_log(run_id, WORKER, "too late\n") is None


def test_events_are_numbered_from_one_with_no_gaps(client, checkout):
    run_id = _claim(client)
    for index in range(4):
        db.append_log(run_id, WORKER, f"line {index}\n")
    seqs = [event["seq"] for event in db.list_events(run_id)]
    assert seqs == [1, 2, 3, 4, 5, 6]
    assert db.get_run(run_id)["lastEventSeq"] == 6


def test_the_log_is_capped(client, checkout, monkeypatch):
    monkeypatch.setattr(db, "MAX_LOG_BYTES", 64)
    run_id = _claim(client)
    for _ in range(20):
        db.append_log(run_id, WORKER, "x" * 32 + "\n")
    run = db.get_run(run_id)
    assert run["logBytes"] == 64
    assert run["logTruncated"] is True
    # Past the cap a run keeps its status readable and stops storing output, rather than
    # growing the studio database for the life of the install.
    assert db.append_log(run_id, WORKER, "more\n") is None


def test_finishing_requires_the_lease(client, checkout):
    run_id = _claim(client)
    assert db.finish(run_id, "worker-2", "completed") is None
    _expire(run_id)
    assert db.finish(run_id, WORKER, "completed") is None
    assert db.finish(run_id, WORKER, "completed", allow_expired = True) == "completed"


def test_a_cancelled_run_stores_no_result_and_no_error(client, checkout):
    run_id = _claim(client)
    db.request_cancel(run_id)
    assert db.finish(run_id, WORKER, "completed", "ignored", {"val_bpb": 1.0}) == "cancelled"
    run = db.get_run(run_id)
    assert run["error"] is None
    assert run["metrics"] is None


def test_recover_expired_frees_a_stalled_run(client, checkout):
    run_id = _claim(client)
    _expire(run_id)
    assert run_as(ALICE, db.recover_expired) == 1
    assert run_as(ALICE, db.recover_expired) == 0
    assert run_as(ALICE, db.claim_next, "worker-2")["id"] == run_id


def test_releasing_leases_leaves_the_run_claimable(client, checkout):
    _claim(client)
    assert run_as(ALICE, db.release_worker_leases, WORKER) == 1
    assert db.owns_lease(db.list_active("alice")[0]["id"], WORKER) is False


def test_a_run_waits_for_events_rather_than_polling(client, checkout):
    run_id = _claim(client)
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


def test_an_event_for_a_run_that_is_gone_is_refused(client, checkout):
    run_id = _claim(client)
    _delete(run_id)
    with pytest.raises(KeyError):
        db.append_event(run_id, "run.note", {})


def test_a_retired_account_cannot_open_the_connection(monkeypatch, client, checkout):
    monkeypatch.setattr(db, "account_is_retired", lambda: True)
    with pytest.raises(RuntimeError):
        db.get_connection()


def test_the_schema_is_created_by_the_shared_initializer(client, checkout):
    client.post("/make", headers = {"x-test-account": "alice"}, json = _payload())
    conn = db.get_connection()
    names = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    conn.close()
    assert {"make_runs", "make_events"} <= names


def test_a_bad_status_never_reaches_the_table(client, checkout):
    with pytest.raises(sqlite3.IntegrityError):
        conn = db.get_connection()
        try:
            conn.execute(
                "INSERT INTO make_runs (id, owner_subject, status, preset, config_json, "
                "argv_json, created_at, updated_at) VALUES ('x', 'y', 'imaginary', 'small', "
                "'{}', '[]', 0, 0)"
            )
            conn.commit()
        finally:
            conn.close()


def _expire(run_id: str) -> None:
    conn = db.get_connection()
    conn.execute(
        "UPDATE make_runs SET lease_expires_at = ? WHERE id = ?", (db.now_ms() - 1, run_id)
    )
    conn.commit()
    conn.close()


def _delete(run_id: str) -> None:
    conn = db.get_connection()
    conn.execute("DELETE FROM make_runs WHERE id = ?", (run_id,))
    conn.commit()
    conn.close()
