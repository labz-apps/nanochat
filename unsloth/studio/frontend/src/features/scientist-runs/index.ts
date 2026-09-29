// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The page itself is NOT re-exported: the /scientist-runs route lazy-imports it
// directly (app/routes/scientist-runs.tsx) so it stays code-split out of the
// always-mounted root layout. This barrel is what the rest of the app may use.
export {
  cancelScientistRun,
  createScientistRun,
  getScientistEnvironment,
  getScientistRun,
  getScientistRunLogs,
  listScientistRuns,
  retryScientistRun,
  ScientistApiError,
  streamScientistRunEvents,
} from "./api/scientist-runs-api";
export {
  useScientistEnvironment,
  useScientistRuns,
} from "./hooks/use-scientist-runs";
export type { ScientistRunsState } from "./hooks/use-scientist-runs";
export { useScientistRunStream } from "./hooks/use-scientist-run-stream";
export type { ScientistRunStream } from "./hooks/use-scientist-run-stream";
export {
  selectScientistRunInProgress,
  useScientistRuntimeStore,
} from "./stores/scientist-runtime-store";
export type { ScientistRuntimeStore } from "./stores/scientist-runtime-store";
export {
  canCancelScientistRun,
  canRetryScientistRun,
  formatBytes,
  formatDurationMs,
  formatWhen,
  headlineMetrics,
  isScientistRunActive,
  primaryMetric,
  scientistRunDurationMs,
  scientistRunTitle,
} from "./lib/run-display";
export {
  ACTIVE_SCIENTIST_RUN_STATUSES,
  SCIENTIST_RUN_STATUSES,
} from "./types/api";
export type {
  CreateScientistRunInput,
  ScientistEnvironment,
  ScientistEnvironmentProblem,
  ScientistMetrics,
  ScientistPrimaryMetric,
  ScientistRun,
  ScientistRunConfig,
  ScientistRunEvent,
  ScientistRunEventName,
  ScientistRunListResponse,
  ScientistRunLogsResponse,
  ScientistRunStatus,
} from "./types/api";
