// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import {
  cancelMakeRun as cancelRun,
  createMakeRun,
  getMakeEnvironment,
  listMakeRuns,
  retryMakeRun as retryRun,
} from "../api/make-api";
import { useMakeRuntimeStore } from "../stores/make-runtime-store";
import { isMakeRunActive } from "../lib/run-display";
import type { CreateMakeRunInput, MakeEnvironment, MakeRun } from "../types/api";

/** Poll cadence for the list. Fast enough that a run looks alive, slow enough that an idle page is not a request a second. */
const IDLE_POLL_MS = 5_000;
const BUSY_POLL_MS = 1_500;

export interface MakeRunsState {
  runs: MakeRun[];
  total: number;
  loading: boolean;
  error: string | null;
  creating: boolean;
  cancel: (runId: string) => Promise<void>;
  retry: (runId: string) => Promise<void>;
  create: (input: CreateMakeRunInput) => Promise<MakeRun | null>;
  refresh: () => void;
}

export function useMakeRuns(): MakeRunsState {
  const [runs, setRuns] = useState<MakeRun[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const busy = runs.some((run) => isMakeRunActive(run.status));
  const setRunInProgress = useMakeRuntimeStore((state) => state.setRunInProgress);
  const inFlight = useRef<AbortController | null>(null);

  const refresh = useCallback(() => {
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    listMakeRuns({ limit: 50 }, controller.signal)
      .then((payload) => {
        if (controller.signal.aborted) return;
        setRuns(payload.runs);
        setTotal(payload.total);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setError(cause instanceof Error ? cause.message : String(cause));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
  }, []);

  useEffect(() => {
    refresh();
    return () => {
      inFlight.current?.abort();
    };
  }, [refresh]);

  // Two timers rather than one that re-reads a ref: the cadence is chosen by whether
  // something is running, which only changes when a run starts or stops.
  useEffect(() => {
    if (!busy) return undefined;
    const timer = window.setInterval(refresh, BUSY_POLL_MS);
    return () => window.clearInterval(timer);
  }, [busy, refresh]);

  useEffect(() => {
    if (busy) return undefined;
    const timer = window.setInterval(refresh, IDLE_POLL_MS);
    return () => window.clearInterval(timer);
  }, [busy, refresh]);

  // The sidebar's spinner, written from here so the flag follows the same poll that
  // decides what the page shows.
  useEffect(() => {
    setRunInProgress(busy);
  }, [busy, setRunInProgress]);

  const create = useCallback(
    async (input: CreateMakeRunInput) => {
      setCreating(true);
      try {
        const run = await createMakeRun(input);
        refresh();
        return run;
      } finally {
        setCreating(false);
      }
    },
    [refresh],
  );

  const cancel = useCallback(
    async (runId: string) => {
      await cancelRun(runId);
      refresh();
    },
    [refresh],
  );

  const retry = useCallback(
    async (runId: string) => {
      await retryRun(runId);
      refresh();
    },
    [refresh],
  );

  return { runs, total, loading, error, creating, cancel, retry, create, refresh };
}

export function useMakeEnvironment(): { environment: MakeEnvironment | null } {
  const [environment, setEnvironment] = useState<MakeEnvironment | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    getMakeEnvironment(controller.signal)
      .then(setEnvironment)
      .catch(() => {
        /* The page renders its own "cannot check" state; a failed probe is not fatal. */
      });
    return () => controller.abort();
  }, []);
  return { environment };
}
