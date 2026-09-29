// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The page itself is NOT re-exported: the /make route lazy-imports it directly
// (app/routes/make.tsx) so it stays code-split out of the always-mounted root layout.
export {
  cancelMakeRun,
  createMakeRun,
  getMakeEnvironment,
  getMakePresets,
  getMakeRun,
  getMakeRunLogs,
  listMakeRuns,
  MakeApiError,
  retryMakeRun,
  streamMakeRunEvents,
} from "./api/make-api";
export { useMakeEnvironment, useMakeRuns } from "./hooks/use-make-runs";
export type { MakeRunsState } from "./hooks/use-make-runs";
export { useMakeRunStream } from "./hooks/use-make-run-stream";
export type { MakeRunStream } from "./hooks/use-make-run-stream";
export {
  selectMakeRunInProgress,
  useMakeRuntimeStore,
} from "./stores/make-runtime-store";
export type { MakeRuntimeStore } from "./stores/make-runtime-store";
export {
  canCancelMakeRun,
  canRetryMakeRun,
  estimateParameters,
  formatBytes,
  formatDurationMs,
  formatParameterCount,
  formatWhen,
  headlineMetrics,
  isMakeRunActive,
  makeRunDurationMs,
  makeRunTitle,
  MAKE_LIMITS,
  validateMakeForm,
} from "./lib/run-display";
export { ACTIVE_MAKE_RUN_STATUSES, MAKE_RUN_STATUSES } from "./types/api";
export type {
  CreateMakeRunInput,
  MakeEnvironment,
  MakeEnvironmentProblem,
  MakeMetrics,
  MakePreset,
  MakeProgress,
  MakeRun,
  MakeRunConfig,
  MakeRunEvent,
  MakeRunEventName,
  MakeRunListResponse,
  MakeRunLogsResponse,
  MakeRunStatus,
} from "./types/api";
