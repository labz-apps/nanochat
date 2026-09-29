// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/**
 * Whether an AI Scientist run is in flight, for the sidebar row's spinner.
 *
 * Deliberately not the run list itself: the sidebar is mounted for the life of the
 * app, and a run can start from anywhere (a deep link, another tab, a run started
 * before a restart). So this is one boolean, written from whichever surface happens
 * to have polled, and read by everything else.
 */
export interface ScientistRuntimeStore {
  runInProgress: boolean;
  setRunInProgress: (value: boolean) => void;
}

export const useScientistRuntimeStore = create<ScientistRuntimeStore>((set) => ({
  runInProgress: false,
  setRunInProgress: (value) =>
    set((state) =>
      state.runInProgress === value ? state : { runInProgress: value },
    ),
}));

export const selectScientistRunInProgress = (state: ScientistRuntimeStore): boolean =>
  state.runInProgress;
