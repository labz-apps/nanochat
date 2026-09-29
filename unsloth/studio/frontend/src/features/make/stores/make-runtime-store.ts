// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/**
 * Whether a pretrain is in flight, for the sidebar row's spinner.
 *
 * One boolean written from whichever surface polled, read by everything else. The
 * sidebar is mounted for the life of the app, and a run can start from a deep link or
 * survive a Studio restart, so this is deliberately not the run list itself.
 */
export interface MakeRuntimeStore {
  runInProgress: boolean;
  setRunInProgress: (value: boolean) => void;
}

export const useMakeRuntimeStore = create<MakeRuntimeStore>((set) => ({
  runInProgress: false,
  setRunInProgress: (value) =>
    set((state) => (state.runInProgress === value ? state : { runInProgress: value })),
}));

export const selectMakeRunInProgress = (state: MakeRuntimeStore): boolean =>
  state.runInProgress;
