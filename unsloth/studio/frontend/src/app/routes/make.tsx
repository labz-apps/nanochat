// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const MakeRunsPage = lazyRouteComponent(
  () => import("@/features/make/make-runs-page"),
  "MakeRunsPage",
);

export type MakeRunsSearch = {
  // Which run to open on arrival, so a link points at one run rather than the list.
  run?: string;
};

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/make",
  staticData: { title: "Make" },
  beforeLoad: () => requireAuth(),
  validateSearch: (search: Record<string, unknown>): MakeRunsSearch => ({
    run: typeof search.run === "string" ? search.run : undefined,
  }),
  component: MakeRunsPage,
});
