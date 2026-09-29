// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const ScientistRunsPage = lazyRouteComponent(
  () => import("@/features/scientist-runs/scientist-runs-page"),
  "ScientistRunsPage",
);

export type ScientistRunsSearch = {
  // Which run to open on arrival, so a link from a notification or a chat points at
  // one run rather than at the list.
  run?: string;
};

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/scientist-runs",
  staticData: { title: "AI Scientist" },
  beforeLoad: () => requireAuth(),
  validateSearch: (search: Record<string, unknown>): ScientistRunsSearch => ({
    run: typeof search.run === "string" ? search.run : undefined,
  }),
  component: ScientistRunsPage,
});
