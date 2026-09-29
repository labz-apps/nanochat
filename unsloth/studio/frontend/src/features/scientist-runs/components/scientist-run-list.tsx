// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Spinner } from "@/components/ui/spinner";
import { useLocale, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { MicroscopeIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ScientistRunStatusBadge } from "./scientist-run-status-badge";
import {
  formatDurationMs,
  formatWhen,
  primaryMetric,
  scientistRunDurationMs,
  scientistRunTitle,
} from "../lib/run-display";
import type { ScientistRun } from "../types/api";

function RunRow({
  run,
  selected,
  onSelect,
}: {
  run: ScientistRun;
  selected: boolean;
  onSelect: () => void;
}) {
  const t = useT();
  const locale = useLocale();
  const metric = primaryMetric(run.metrics);
  const duration = formatDurationMs(scientistRunDurationMs(run));
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-current={selected}
      className={cn(
        "flex w-full flex-col gap-1.5 rounded-2xl px-3 py-2.5 text-left transition-colors",
        "hover:bg-muted/60 focus-visible:ring-ring focus-visible:ring-2 focus-visible:outline-none",
        selected && "bg-muted ring-ring/20 ring-1",
      )}
    >
      <div className="flex items-center gap-2">
        <span className="truncate text-sm font-medium">
          {scientistRunTitle(run)}
        </span>
        <ScientistRunStatusBadge status={run.status} />
      </div>
      <div className="text-muted-foreground flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs">
        <span>{formatWhen(run.createdAt, locale)}</span>
        <span>{t("scientist.list.duration", { value: duration })}</span>
        {metric ? (
          <span className="font-medium">
            {metric.name}={metric.value}
          </span>
        ) : null}
        {run.retryCount > 0 ? (
          <span>{t("scientist.list.attempts", { count: run.retryCount + 1 })}</span>
        ) : null}
      </div>
    </button>
  );
}

export function ScientistRunList({
  runs,
  selectedId,
  onSelect,
  loading,
  onNew,
  canStart,
  starting,
}: {
  runs: ScientistRun[];
  selectedId: string | null;
  onSelect: (runId: string) => void;
  loading: boolean;
  onNew: () => void;
  canStart: boolean;
  starting: boolean;
}) {
  const t = useT();
  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <h2 className="font-heading text-sm font-semibold tracking-tight">
          {t("scientist.list.title")}
        </h2>
        <Button size="sm" onClick={onNew} disabled={!canStart || starting}>
          {starting ? <Spinner /> : null}
          {t("scientist.action.newRun")}
        </Button>
      </div>
      <ScrollArea className="min-h-0 flex-1">
        {loading && runs.length === 0 ? (
          <div className="text-muted-foreground flex items-center justify-center gap-2 py-10 text-sm">
            <Spinner />
            {t("common.loading")}
          </div>
        ) : runs.length === 0 ? (
          <Empty className="border">
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <HugeiconsIcon icon={MicroscopeIcon} />
              </EmptyMedia>
              <EmptyTitle>{t("scientist.list.emptyTitle")}</EmptyTitle>
              <EmptyDescription>
                {t("scientist.list.emptyDescription")}
              </EmptyDescription>
            </EmptyHeader>
          </Empty>
        ) : (
          <div className="flex flex-col gap-1 pr-2">
            {runs.map((run) => (
              <RunRow
                key={run.id}
                run={run}
                selected={run.id === selectedId}
                onSelect={() => onSelect(run.id)}
              />
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
