// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type MakeRunStatus =
  | "queued"
  | "running"
  | "cancelling"
  | "cancelled"
  | "completed"
  | "failed";

export const MAKE_RUN_STATUSES: readonly MakeRunStatus[] = [
  "queued",
  "running",
  "cancelling",
  "cancelled",
  "completed",
  "failed",
] as const;

export const ACTIVE_MAKE_RUN_STATUSES: ReadonlySet<string> = new Set<MakeRunStatus>(
  ["queued", "running", "cancelling"],
);

/** A size-and-length combination. The server is the authority on which are legal. */
export interface MakePreset {
  id: string;
  label: string;
  depth: number;
  iterations: number;
  deviceBatchSize: number;
  totalBatchSize: number;
  aspectRatio: number;
  headDim: number;
  maxSeqLen: number;
}

/** What a run was configured with. Every field is a request the server range-checked. */
export interface MakeRunConfig {
  preset: string;
  depth: number;
  iterations: number;
  deviceBatchSize: number;
  totalBatchSize: number;
  aspectRatio: number;
  headDim: number;
  maxSeqLen: number;
  runTag: string;
  timeoutSeconds: number | null;
  evaluate: boolean;
}

export type MakeMetrics = Record<string, number | string | boolean | null>;

export interface MakeRun {
  id: string;
  ownerSubject: string;
  status: MakeRunStatus;
  preset: string;
  config: MakeRunConfig;
  argv: string[];
  checkpointDir: string | null;
  logBytes: number;
  logTruncated: boolean;
  cancelRequested: boolean;
  retryCount: number;
  error: string | null;
  metrics: MakeMetrics | null;
  createdAt: number;
  updatedAt: number;
  startedAt: number | null;
  completedAt: number | null;
  heartbeatAt: number | null;
  lastEventSeq: number;
}

export interface MakeRunListResponse {
  runs: MakeRun[];
  total: number;
}

export interface MakeRunLogsResponse {
  runId: string;
  text: string;
  truncated: boolean;
  lastEventSeq: number;
}

export interface MakeEnvironmentProblem {
  code: string;
  message: string;
  fix: string;
}

export interface MakeEnvironment {
  available: boolean;
  error: string | null;
  problems: MakeEnvironmentProblem[];
  projectRoot: string | null;
  python: string | null;
  sharedCache: string | null;
  checkpointRoot: string | null;
  presets: MakePreset[];
}

export interface CreateMakeRunInput {
  preset: string;
  depth?: number;
  iterations?: number;
  deviceBatchSize?: number;
  totalBatchSize?: number;
  aspectRatio?: number;
  headDim?: number;
  maxSeqLen?: number;
  runTag?: string | null;
  timeoutSeconds?: number | null;
  evaluate?: boolean;
}

/** The trainer's own progress line, parsed server-side. */
export interface MakeProgress {
  step?: number;
  [key: string]: number | string | undefined;
}

export type MakeRunEventName =
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

export interface MakeRunEvent {
  id: number;
  event: MakeRunEventName;
  createdAt: number;
  attempt: number;
  data: Record<string, unknown>;
  run?: MakeRun;
}
