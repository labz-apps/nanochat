// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

type ScientistRun = import("../src/features/scientist-runs/types/api.ts").ScientistRun;

const {
  canCancelScientistRun,
  canRetryScientistRun,
  formatBytes,
  formatDurationMs,
  headlineMetrics,
  isScientistRunActive,
  primaryMetric,
  scientistRunDurationMs,
} = await import("../src/features/scientist-runs/lib/run-display.ts");

function run(overrides: Partial<ScientistRun> = {}): ScientistRun {
  return {
    id: "r1",
    ownerSubject: "alice",
    status: "running",
    config: {
      model: "opencode/space-bunny-free",
      ideaIdx: 0,
      attemptId: 0,
      loadCode: true,
      loadIdeas: "ai_scientist/ideas/one.json",
      config: "bfts_config.yaml",
      allowProviderCalls: true,
      timeoutSeconds: null,
      parentJournal: null,
      parentNodeId: null,
      parentStage: null,
      lineageId: null,
    },
    argv: ["python", "launch_scientist_bfts.py"],
    artifactDir: null,
    logBytes: 0,
    logTruncated: false,
    cancelRequested: false,
    retryCount: 0,
    error: null,
    metrics: null,
    curves: null,
    createdAt: 1_000,
    updatedAt: 1_000,
    startedAt: 2_000,
    completedAt: null,
    heartbeatAt: 2_000,
    lastEventSeq: 0,
    ...overrides,
  };
}

test("only the statuses with a child or a claim are active", () => {
  for (const status of ["queued", "running", "cancelling"] as const) {
    assert.equal(isScientistRunActive(status), true, status);
  }
  for (const status of ["cancelled", "completed", "failed"] as const) {
    assert.equal(isScientistRunActive(status), false, status);
  }
});

test("cancel and retry are offered only where they do something", () => {
  assert.equal(canCancelScientistRun(run({ status: "queued" })), true);
  assert.equal(canCancelScientistRun(run({ status: "running" })), true);
  // Cancelling is already in flight; offering it again would be a no-op button.
  assert.equal(canCancelScientistRun(run({ status: "cancelling" })), false);
  assert.equal(canCancelScientistRun(run({ status: "completed" })), false);

  assert.equal(canRetryScientistRun(run({ status: "failed" })), true);
  assert.equal(canRetryScientistRun(run({ status: "cancelled" })), true);
  assert.equal(canRetryScientistRun(run({ status: "running" })), false);
  assert.equal(canRetryScientistRun(run({ status: "completed" })), false);
});

test("a run that has not started has no duration", () => {
  assert.equal(scientistRunDurationMs(run({ startedAt: null })), null);
  // A running run measures against now, so the instant is passed explicitly.
  assert.equal(scientistRunDurationMs(run({ startedAt: 2_000 }), 7_000), 5_000);
  assert.equal(
    scientistRunDurationMs(run({ startedAt: 2_000, completedAt: 9_000 })),
    7_000,
  );
  // A clock that went backwards must not render a negative duration.
  assert.equal(scientistRunDurationMs(run({ startedAt: 10_000 }), 4_000), 0);
});

test("durations read as a clock, and a missing one is a dash", () => {
  assert.equal(formatDurationMs(null), "—");
  assert.equal(formatDurationMs(0), "0:00");
  assert.equal(formatDurationMs(9_000), "0:09");
  assert.equal(formatDurationMs(65_000), "1:05");
  assert.equal(formatDurationMs(3_725_000), "1:02:05");
  assert.equal(formatDurationMs(Number.NaN), "—");
});

test("byte counts are readable and never negative", () => {
  assert.equal(formatBytes(0), "0 B");
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(1024), "1.0 KB");
  assert.equal(formatBytes(1024 * 1024 * 17), "17.0 MB");
  assert.equal(formatBytes(1024 * 1024 * 1024 * 204), "204 GB");
  assert.equal(formatBytes(-1), "—");
  assert.equal(formatBytes(null), "—");
});

test("the headline metrics come first and the primary one is lifted out", () => {
  // The API speaks snake_case; the client camelizes before the type applies.
  const metrics = {
    primary_metric: { name: "val_bpb", value: 0.912, lowerIsBetter: true },
    val_bpb: 0.912,
    peak_vram_bytes: 16_106_127_360,
    mfu_percent: 41.25,
    custom_thing: "kept",
  };
  assert.deepEqual(primaryMetric(metrics), {
    name: "val_bpb",
    value: 0.912,
    lowerIsBetter: true,
  });

  const shown = headlineMetrics(metrics);
  // primary_metric is shown on its own, so it must not also be a tile.
  assert.deepEqual(
    shown.map((entry) => entry.name),
    ["val_bpb", "mfu_percent", "peak_vram_bytes", "custom_thing"],
  );
  assert.equal(shown[0].value, "0.9120");
  assert.equal(shown.at(-1)?.value, "kept");
});

test("byte-valued metrics are formatted as sizes, not as raw integers", () => {
  const shown = headlineMetrics({
    peak_vram_bytes: 16_106_127_360,
    mfu_percent: 41.25,
    total_training_tokens: 983040,
  });
  assert.deepEqual(shown, [
    { name: "mfu_percent", value: "41.3%" },
    { name: "peak_vram_bytes", value: "15.0 GB" },
    { name: "total_training_tokens", value: "983,040" },
  ]);
});

test("a run with no result reports no metrics at all", () => {
  assert.deepEqual(headlineMetrics(null), []);
  assert.equal(primaryMetric(null), null);
  // NaN is a number, and "NaN" in a headline tile is worse than no headline.
  assert.equal(
    primaryMetric({
      primary_metric: { name: "x", value: Number.NaN, lowerIsBetter: true },
    }),
    null,
  );
});

