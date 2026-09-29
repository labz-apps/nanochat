// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

type MakeRun = import("../src/features/make/types/api.ts").MakeRun;

const {
  canCancelMakeRun,
  canRetryMakeRun,
  estimateParameters,
  formatBytes,
  formatDurationMs,
  formatParameterCount,
  isMakePresetId,
  isMakeRunActive,
  headlineMetrics,
  makeRunDurationMs,
  validateMakeForm,
} = await import("../src/features/make/lib/run-display.ts");

function run(overrides: Partial<MakeRun> = {}): MakeRun {
  return {
    id: "m1",
    ownerSubject: "alice",
    status: "running",
    preset: "small",
    config: {
      preset: "small",
      depth: 6,
      iterations: 300,
      deviceBatchSize: 16,
      totalBatchSize: 65_536,
      aspectRatio: 64,
      headDim: 128,
      maxSeqLen: 2048,
      runTag: "make",
      timeoutSeconds: null,
      evaluate: true,
    },
    argv: ["python", "-m", "scripts.base_train"],
    checkpointDir: null,
    logBytes: 0,
    logTruncated: false,
    cancelRequested: false,
    retryCount: 0,
    error: null,
    metrics: null,
    createdAt: 1_000,
    updatedAt: 1_000,
    startedAt: 2_000,
    completedAt: null,
    heartbeatAt: 2_000,
    lastEventSeq: 0,
    ...overrides,
  };
}

const VALID_FORM = {
  preset: "small",
  depth: 6,
  iterations: 300,
  deviceBatchSize: 16,
  totalBatchSize: 65_536,
  maxSeqLen: 2048,
};

test("only the statuses with a child or a claim are active", () => {
  for (const status of ["queued", "running", "cancelling"] as const) {
    assert.equal(isMakeRunActive(status), true, status);
  }
  for (const status of ["cancelled", "completed", "failed"] as const) {
    assert.equal(isMakeRunActive(status), false, status);
  }
});

test("cancel and retry are offered only where they do something", () => {
  assert.equal(canCancelMakeRun(run({ status: "queued" })), true);
  assert.equal(canCancelMakeRun(run({ status: "running" })), true);
  // Cancelling is already in flight; offering it again would be a no-op button.
  assert.equal(canCancelMakeRun(run({ status: "cancelling" })), false);
  assert.equal(canCancelMakeRun(run({ status: "completed" })), false);

  assert.equal(canRetryMakeRun(run({ status: "failed" })), true);
  assert.equal(canRetryMakeRun(run({ status: "cancelled" })), true);
  assert.equal(canRetryMakeRun(run({ status: "running" })), false);
});

test("a run that has not started has no duration", () => {
  assert.equal(makeRunDurationMs(run({ startedAt: null })), null);
  // A running run measures against now, so the instant is passed explicitly.
  assert.equal(makeRunDurationMs(run({ startedAt: 2_000 }), 7_000), 5_000);
  assert.equal(makeRunDurationMs(run({ startedAt: 2_000, completedAt: 9_000 })), 7_000);
  // A clock that went backwards must not render a negative duration.
  assert.equal(makeRunDurationMs(run({ startedAt: 10_000 }), 4_000), 0);
});

test("durations read as a clock, and a missing one is a dash", () => {
  assert.equal(formatDurationMs(null), "—");
  assert.equal(formatDurationMs(0), "0:00");
  assert.equal(formatDurationMs(3_725_000), "1:02:05");
  assert.equal(formatDurationMs(Number.NaN), "—");
});

test("a valid form passes and each broken invariant names itself", () => {
  assert.equal(validateMakeForm(VALID_FORM), null);
  // The two nanochat invariants: grad accumulation must be a whole number of steps, and
  // the context must divide the model's block size.
  assert.match(
    validateMakeForm({ ...VALID_FORM, deviceBatchSize: 64, totalBatchSize: 33_000 }) ?? "",
    /multiple/,
  );
  assert.match(validateMakeForm({ ...VALID_FORM, maxSeqLen: 2_000 }) ?? "", /multiple/);
  // And the ranges, which are restated from the server rather than invented here.
  assert.match(validateMakeForm({ ...VALID_FORM, depth: 99 }) ?? "", /depth/);
  assert.match(validateMakeForm({ ...VALID_FORM, iterations: 0 }) ?? "", /iterations/);
  assert.match(validateMakeForm({ ...VALID_FORM, totalBatchSize: 200 }) ?? "", /totalBatchSize/);
  assert.match(validateMakeForm({ ...VALID_FORM, deviceBatchSize: Number.NaN }) ?? "", /deviceBatchSize/);
});

test("parameter estimates grow with depth and read in human units", () => {
  const tiny = estimateParameters(4, 64);
  const medium = estimateParameters(8, 64);
  assert.ok(medium > tiny, "a deeper model must estimate larger");
  // The two batch quantities have different ranges, which is why depth is the knob
  // that moves size at all: 12 * d^2 * depth.
  assert.equal(estimateParameters(6, 64), 12 * (6 * 64) ** 2 * 6);
  assert.equal(formatParameterCount(0), "0");
  assert.equal(formatParameterCount(1_500), "2K");
  assert.equal(formatParameterCount(2_400_000), "2M");
  assert.equal(formatParameterCount(1_500_000_000), "1.5B");
});

test("preset ids are a closed set so a label can never go missing", () => {
  assert.equal(isMakePresetId("tiny"), true);
  assert.equal(isMakePresetId("medium"), true);
  // A preset added to the server without a label falls back to its own label rather
  // than rendering a raw key.
  assert.equal(isMakePresetId("enormous"), false);
});

test("headline metrics come first and byte values are formatted as sizes", () => {
  const shown = headlineMetrics({
    val_bpb: 1.2345,
    mfu_percent: 41.25,
    peak_vram_bytes: 16_106_127_360,
    total_training_tokens: 983040,
  });
  assert.deepEqual(shown, [
    { name: "val_bpb", value: "1.2345" },
    { name: "mfu_percent", value: "41.3%" },
    { name: "peak_vram_bytes", value: "15.0 GB" },
    { name: "total_training_tokens", value: "983,040" },
  ]);
  assert.deepEqual(headlineMetrics(null), []);
});

test("byte counts are readable and never negative", () => {
  assert.equal(formatBytes(0), "0 B");
  assert.equal(formatBytes(1024), "1.0 KB");
  assert.equal(formatBytes(1024 * 1024 * 1024 * 204), "204 GB");
  assert.equal(formatBytes(-1), "—");
  assert.equal(formatBytes(null), "—");
});
