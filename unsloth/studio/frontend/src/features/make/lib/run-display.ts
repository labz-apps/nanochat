// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { formatRelativeTime, type Locale } from "@/i18n";
import {
  ACTIVE_MAKE_RUN_STATUSES,
  type MakeMetrics,
  type MakePreset,
  type MakeRun,
  type MakeRunStatus,
} from "../types/api";

/** Shown when the environment probe has not answered yet, so the dialog can render. */
export const FALLBACK_PRESET: MakePreset = {
  id: "small",
  label: "Small",
  depth: 6,
  iterations: 300,
  deviceBatchSize: 16,
  totalBatchSize: 65_536,
  aspectRatio: 64,
  headDim: 128,
  maxSeqLen: 2048,
};

/**
 * The preset ids that have a label. Kept as a union rather than interpolated, so
 * `t(\`make.preset.${id}\`)` is checked at compile time and a preset added to the server
 * without a label here is a type error rather than a raw key on screen.
 */
export type MakePresetId = "tiny" | "small" | "medium";

export function isMakePresetId(value: string): value is MakePresetId {
  return value === "tiny" || value === "small" || value === "medium";
}

export function isMakeRunActive(status: MakeRunStatus): boolean {
  return ACTIVE_MAKE_RUN_STATUSES.has(status);
}

export function canCancelMakeRun(run: MakeRun): boolean {
  return run.status === "queued" || run.status === "running";
}

export function canRetryMakeRun(run: MakeRun): boolean {
  return run.status === "failed" || run.status === "cancelled";
}

export function makeRunDurationMs(run: MakeRun, now = Date.now()): number | null {
  if (run.startedAt === null) {
    return null;
  }
  return Math.max(0, (run.completedAt ?? now) - run.startedAt);
}

export function formatDurationMs(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) {
    return "—";
  }
  const totalSeconds = Math.floor(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  const pad = (value: number) => String(value).padStart(2, "0");
  return hours > 0 ? `${hours}:${pad(minutes)}:${pad(seconds)}` : `${minutes}:${pad(seconds)}`;
}

export function formatWhen(timestamp: number | null, locale: Locale): string {
  if (timestamp === null) return "—";
  const diff = Date.now() - timestamp;
  if (diff < 60_000) return formatRelativeTime(locale, 0, "minute");
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 60) return formatRelativeTime(locale, -minutes, "minute");
  const hours = Math.floor(diff / 3_600_000);
  if (hours < 24) return formatRelativeTime(locale, -hours, "hour");
  const days = Math.floor(diff / 86_400_000);
  if (days < 30) return formatRelativeTime(locale, -days, "day");
  return formatRelativeTime(locale, -Math.floor(days / 30), "month");
}

/** The headroom the server enforces, restated so the inputs can be bounded in the UI. */
export const MAKE_LIMITS = {
  depth: { min: 2, max: 12 },
  iterations: { min: 10, max: 20_000 },
  deviceBatchSize: { min: 1, max: 256 },
  totalBatchSize: { min: 1024, max: 4 * 1024 * 1024 },
  maxSeqLen: { min: 256, max: 8192 },
} as const;

/**
 * Whether the form can be submitted, and if not, why.
 *
 * Checked client-side so the button explains itself instead of 400ing, and checked
 * server-side anyway: this is a convenience, not the boundary.
 */
export function validateMakeForm(values: {
  preset: string;
  depth: number;
  iterations: number;
  deviceBatchSize: number;
  totalBatchSize: number;
  maxSeqLen: number;
}): string | null {
  const ranges = [
    ["depth", values.depth, MAKE_LIMITS.depth],
    ["iterations", values.iterations, MAKE_LIMITS.iterations],
    ["deviceBatchSize", values.deviceBatchSize, MAKE_LIMITS.deviceBatchSize],
    ["totalBatchSize", values.totalBatchSize, MAKE_LIMITS.totalBatchSize],
    ["maxSeqLen", values.maxSeqLen, MAKE_LIMITS.maxSeqLen],
  ] as const;
  for (const [name, value, range] of ranges) {
    if (!Number.isFinite(value) || value < range.min || value > range.max) {
      return `${name} must be between ${range.min} and ${range.max}`;
    }
  }
  if (values.totalBatchSize % values.deviceBatchSize !== 0) {
    // nanochat's grad accumulation is total/device; a non-integer factor either errors
    // inside the trainer or silently changes the effective batch.
    return "totalBatchSize must be a multiple of deviceBatchSize";
  }
  if (values.maxSeqLen % 128 !== 0) {
    return "maxSeqLen must be a multiple of 128";
  }
  return null;
}

/**
 * A rough parameter count, so a preset's size is legible before it is started.
 *
 * 12 * L^2 * d for a transformer body of depth L and model_dim d, which is the usual
 * leading term. It ignores the embedding head and the attention/MLP sublayer ratios, so
 * it is the right order of magnitude and nothing more -- which is all a picker needs.
 */
export function estimateParameters(depth: number, aspectRatio: number): number {
  const modelDim = depth * aspectRatio;
  return Math.round(12 * modelDim * modelDim * depth);
}

export function formatParameterCount(count: number): string {
  if (count >= 1_000_000_000) return `${(count / 1_000_000_000).toFixed(1)}B`;
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(0)}M`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(0)}K`;
  return String(count);
}

/** Which numbers are worth putting in front of someone, in the order that reads. */
const HEADLINE_METRICS = [
  "val_bpb",
  "core_metric",
  "train_loss",
  "mfu_percent",
  "peak_vram_bytes",
  "total_training_tokens",
  "train_seconds",
] as const;

export function headlineMetrics(
  metrics: MakeMetrics | null,
): { name: string; value: string }[] {
  if (!metrics) {
    return [];
  }
  const render = (name: string, raw: unknown): string => {
    if (typeof raw === "number") {
      if (name.endsWith("_bytes")) return formatBytes(raw);
      if (name === "mfu_percent") return `${raw.toFixed(1)}%`;
      if (Number.isInteger(raw)) return raw.toLocaleString();
      return raw.toFixed(4);
    }
    return String(raw);
  };
  const names = [
    ...HEADLINE_METRICS.filter((name) => name in metrics),
    ...Object.keys(metrics).filter((name) => !(HEADLINE_METRICS as readonly string[]).includes(name)),
  ];
  return names.map((name) => ({ name, value: render(name, metrics[name]) }));
}

export function formatBytes(bytes: number | null | undefined): string {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes < 0) {
    return "—";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 100 || unit === 0 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

export function makeRunTitle(run: MakeRun): string {
  return run.config?.runTag?.trim() || run.preset || "nanochat";
}
