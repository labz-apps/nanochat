// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { formatRelativeTime, type Locale } from "@/i18n";
import {
  ACTIVE_SCIENTIST_RUN_STATUSES,
  type ScientistMetrics,
  type ScientistRun,
  type ScientistRunStatus,
} from "../types/api";

/** Metrics worth showing first, in the order the results contract defines them. The primary metric is already surfaced on its own. */
const HEADLINE_METRICS = [
  "val_bpb",
  "core_metric",
  "mfu_percent",
  "peak_vram_bytes",
  "total_training_tokens",
  "train_seconds",
] as const;

const isHeadline = (name: string): boolean =>
  (HEADLINE_METRICS as readonly string[]).includes(name);

export function isScientistRunActive(status: ScientistRunStatus): boolean {
  return ACTIVE_SCIENTIST_RUN_STATUSES.has(status);
}

export function canCancelScientistRun(run: ScientistRun): boolean {
  return run.status === "queued" || run.status === "running";
}

export function canRetryScientistRun(run: ScientistRun): boolean {
  return run.status === "failed" || run.status === "cancelled";
}

/** How long a run took, or has been going. Null before it started. */
export function scientistRunDurationMs(run: ScientistRun, now = Date.now()): number | null {
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

/** The metrics worth putting in front of someone, headline first, then whatever else the node reported. */
export function headlineMetrics(
  metrics: ScientistMetrics | null,
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
    ...Object.keys(metrics).filter(
      (name) => !isHeadline(name) && name !== "primary_metric",
    ),
  ];
  return names.map((name) => ({ name, value: render(name, metrics[name]) }));
}

export function primaryMetric(
  metrics: ScientistMetrics | null,
): { name: string; value: number; lowerIsBetter: boolean } | null {
  const primary = metrics?.primary_metric;
  // NaN is a number, and the server already drops non-finite metrics, but a value
  // that renders as "NaN" in a headline tile is worse than no headline at all.
  if (!primary || typeof primary.value !== "number" || !Number.isFinite(primary.value)) {
    return null;
  }
  return {
    name: primary.name,
    value: primary.value,
    lowerIsBetter: primary.lowerIsBetter !== false,
  };
}

/** A stable one-line label for the run, so the list reads without opening anything. */
export function scientistRunTitle(run: ScientistRun): string {
  const model = run.config?.model?.trim();
  return model ? model : "AI Scientist";
}

/** "3 hours ago", in the viewer's locale. Coarse on purpose: a run list is a history, not a log. */
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
