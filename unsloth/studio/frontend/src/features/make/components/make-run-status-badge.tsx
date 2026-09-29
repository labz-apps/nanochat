// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { useT } from "@/i18n";
import type { MakeRunStatus } from "../types/api";

const TONE: Record<MakeRunStatus, string> = {
  queued: "bg-muted text-muted-foreground",
  running: "bg-blue-500/15 text-blue-700 dark:text-blue-300",
  cancelling: "bg-orange-500/15 text-orange-700 dark:text-orange-300",
  completed: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300",
  failed: "bg-red-500/15 text-red-700 dark:text-red-300",
  cancelled: "bg-muted text-muted-foreground",
};

const DOT: Record<MakeRunStatus, string> = {
  queued: "bg-muted-foreground",
  running: "bg-blue-500 animate-pulse",
  cancelling: "bg-orange-500 animate-pulse",
  completed: "bg-emerald-500",
  failed: "bg-red-500",
  cancelled: "bg-muted-foreground",
};

export function MakeRunStatusBadge({ status }: { status: MakeRunStatus }) {
  const t = useT();
  return (
    <Badge variant="secondary" className={`gap-1.5 ${TONE[status]}`}>
      <span className={`size-1.5 rounded-full ${DOT[status]}`} aria-hidden />
      {t(`make.status.${status}`)}
    </Badge>
  );
}
