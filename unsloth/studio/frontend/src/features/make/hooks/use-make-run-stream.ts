// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { getMakeRun, streamMakeRunEvents } from "../api/make-api";
import { isMakeRunActive } from "../lib/run-display";
import type { MakeProgress, MakeRun, MakeRunEvent } from "../types/api";

/** Cap on what is kept in the DOM. A pretrain prints a line per step and never stops. */
const MAX_LOG_CHARS = 400_000;
/** Backoff between a dropped stream and the next attempt. The server replays from the last seq. */
const RECONNECT_BASE_MS = 500;
const RECONNECT_MAX_MS = 15_000;

export interface MakeRunStream {
  run: MakeRun | null;
  log: string;
  truncated: boolean;
  progress: MakeProgress | null;
  checkpointDir: string | null;
  notes: string[];
  connected: boolean;
}

const EMPTY: MakeRunStream = {
  run: null,
  log: "",
  truncated: false,
  progress: null,
  checkpointDir: null,
  notes: [],
  connected: false,
};

type BoundStream = MakeRunStream & { runId: string | null };

export function useMakeRunStream(runId: string | null): MakeRunStream {
  const [bound, setBound] = useState<BoundStream>({ ...EMPTY, runId: null });
  // The stream owns the cursor; React state would re-render the whole detail pane on
  // every one of a training run's log chunks.
  const cursor = useRef(0);

  // Switching runs resets during render rather than in an effect: the next effect run
  // would otherwise inherit the previous run's log for a frame, and a setState in an
  // effect body is the cascading-render pattern React warns about.
  if (bound.runId !== runId) {
    setBound({ ...EMPTY, runId });
  }
  const state: MakeRunStream = bound.runId === runId ? bound : EMPTY;

  const patch = useCallback(
    (update: (previous: MakeRunStream) => MakeRunStream) => {
      setBound((previous) => ({ ...update(previous), runId: previous.runId }));
    },
    [],
  );

  useEffect(() => {
    if (!runId) {
      return undefined;
    }
    // The render above already cleared the state for this runId; the cursor is a ref,
    // and refs may not be written during render, so it is reset here instead.
    cursor.current = 0;
    const controller = new AbortController();
    let cancelled = false;
    let timer: number | undefined;

    const sleep = (ms: number) =>
      new Promise<void>((resolve) => {
        if (cancelled) {
          resolve();
          return;
        }
        timer = window.setTimeout(resolve, ms);
        controller.signal.addEventListener(
          "abort",
          () => {
            if (timer !== undefined) window.clearTimeout(timer);
            resolve();
          },
          { once: true },
        );
      });

    const apply = (event: MakeRunEvent) => {
      cursor.current = Math.max(cursor.current, event.id);
      patch((previous) => {
        const next: MakeRunStream = {
          ...previous,
          run: event.run ?? previous.run,
          connected: true,
        };
        switch (event.event) {
          case "run.log": {
            const chunk = typeof event.data.text === "string" ? event.data.text : "";
            const combined = previous.log + chunk;
            next.truncated = previous.truncated || event.data.truncated === true;
            // Keep the tail: a pretrain log is unbounded and the end is what is read.
            next.log =
              combined.length > MAX_LOG_CHARS
                ? combined.slice(combined.length - MAX_LOG_CHARS)
                : combined;
            if (next.log.length < combined.length) next.truncated = true;
            break;
          }
          case "run.progress":
            next.progress = event.data as MakeProgress;
            break;
          case "run.artifacts":
            next.checkpointDir =
              typeof event.data.checkpointDir === "string"
                ? event.data.checkpointDir
                : previous.checkpointDir;
            break;
          case "run.note":
            if (typeof event.data.message === "string") {
              next.notes = [...previous.notes, event.data.message];
            }
            break;
          default:
            break;
        }
        return next;
      });
    };

    void (async () => {
      let attempts = 0;
      while (!cancelled) {
        try {
          if (cursor.current === 0) {
            // Opening at 0 replays the whole run, so there is no separate log request.
            const snapshot = await getMakeRun(runId, controller.signal);
            if (cancelled) return;
            patch((previous) => ({ ...previous, run: snapshot }));
          }
          for await (const event of streamMakeRunEvents(
            runId,
            cursor.current,
            controller.signal,
          )) {
            attempts = 0;
            apply(event);
          }
          if (cancelled) return;
          const settled = await getMakeRun(runId, controller.signal);
          if (cancelled) return;
          patch((previous) => ({ ...previous, run: settled }));
          // The route closes the stream once a terminal run has been fully replayed, so
          // a clean close on a finished run is the end and not a dropped connection.
          if (!isMakeRunActive(settled.status)) return;
        } catch {
          if (cancelled || controller.signal.aborted) return;
          patch((previous) => ({ ...previous, connected: false }));
        }
        attempts += 1;
        await sleep(
          Math.min(RECONNECT_MAX_MS, RECONNECT_BASE_MS * 2 ** Math.min(attempts - 1, 5)),
        );
      }
    })();

    return () => {
      cancelled = true;
      controller.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [runId, patch]);

  return state;
}
