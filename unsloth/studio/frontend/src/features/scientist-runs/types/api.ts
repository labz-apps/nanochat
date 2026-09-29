// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ScientistRunStatus =
  | "queued"
  | "running"
  | "cancelling"
  | "cancelled"
  | "completed"
  | "failed";

export const SCIENTIST_RUN_STATUSES: readonly ScientistRunStatus[] = [
  "queued",
  "running",
  "cancelling",
  "cancelled",
  "completed",
  "failed",
] as const;

export const ACTIVE_SCIENTIST_RUN_STATUSES: ReadonlySet<string> = new Set<ScientistRunStatus>(
  ["queued", "running", "cancelling"],
);

/** What the launcher will be pointed at. Every field here is a request, not a fact: the server decides the interpreter, the checkout, the cache and the artifact directory. */
export interface ScientistRunConfig {
  model: string;
  ideaIdx: number;
  attemptId: number;
  loadCode: boolean;
  loadIdeas: string;
  config: string;
  allowProviderCalls: boolean;
  timeoutSeconds: number | null;
  parentJournal: string | null;
  parentNodeId: string | null;
  parentStage: number | null;
  lineageId: string | null;
}

export type ScientistPrimaryMetric = {
  name: string;
  value: number;
  lowerIsBetter: boolean;
};

export type ScientistMetricValue = number | string | boolean | null | undefined;

/**
 * What the run's `results.json` reported. `primary_metric` is declared rather than
 * folded into the index signature because the results contract nests it, and an
 * intersection with an index signature would make it unsatisfiable for any object
 * that carries it.
 */
export interface ScientistMetrics {
  readonly [name: string]: ScientistMetricValue | ScientistPrimaryMetric;
  readonly primary_metric?: ScientistPrimaryMetric;
}

export interface ScientistRun {
  id: string;
  ownerSubject: string;
  status: ScientistRunStatus;
  config: ScientistRunConfig;
  argv: string[];
  artifactDir: string | null;
  logBytes: number;
  logTruncated: boolean;
  cancelRequested: boolean;
  retryCount: number;
  error: string | null;
  metrics: ScientistMetrics | null;
  curves: Record<string, number[]> | null;
  createdAt: number;
  updatedAt: number;
  startedAt: number | null;
  completedAt: number | null;
  heartbeatAt: number | null;
  lastEventSeq: number;
}

export interface ScientistRunListResponse {
  runs: ScientistRun[];
  total: number;
}

export interface ScientistRunLogsResponse {
  runId: string;
  text: string;
  truncated: boolean;
  lastEventSeq: number;
}

/** One thing standing between this install and a run that can start. */
export interface ScientistEnvironmentProblem {
  code: string;
  message: string;
  /** The literal remedy, because "not available" alone does not say which knob to turn. */
  fix: string;
}

/** What this installation can actually run, so the page can say why it cannot. */
export interface ScientistEnvironment {
  available: boolean;
  error: string | null;
  problems: ScientistEnvironmentProblem[];
  projectRoot: string | null;
  python: string | null;
  /** True when no interpreter was configured and Studio's own was substituted. */
  pythonIsFallback: boolean;
  sharedCache: string | null;
  experimentRoot: string | null;
  providerKeys: string[];
}

export interface CreateScientistRunInput {
  model: string;
  ideaIdx?: number;
  attemptId?: number;
  loadCode?: boolean;
  loadIdeas?: string;
  config?: string;
  allowProviderCalls: true;
  timeoutSeconds?: number | null;
  parentJournal?: string | null;
  parentNodeId?: string | null;
  parentStage?: number | null;
  lineageId?: string | null;
}

export type ScientistRunEventName =
  | "run.queued"
  | "run.started"
  | "run.spawned"
  | "run.log"
  | "run.progress"
  | "run.note"
  | "run.artifacts"
  | "run.completed"
  | "run.failed"
  | "run.cancelled"
  | "run.retried";

export interface ScientistRunEvent {
  id: number;
  event: ScientistRunEventName;
  createdAt: number;
  attempt: number;
  data: Record<string, unknown>;
  run?: ScientistRun;
}
