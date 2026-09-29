// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Empty,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useLocale, useT } from "@/i18n";
import { toastError } from "@/shared/toast";
import {
  Cancel01Icon,
  Copy02Icon,
  LoadingIcon,
  RefreshIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useState } from "react";
import { ScientistRunStatusBadge } from "./scientist-run-status-badge";
import {
  canCancelScientistRun,
  canRetryScientistRun,
  formatDurationMs,
  formatWhen,
  headlineMetrics,
  primaryMetric,
  scientistRunDurationMs,
} from "../lib/run-display";
import type { ScientistRunStream } from "../hooks/use-scientist-run-stream";
import type { ScientistRun } from "../types/api";

function MetricGrid({ run }: { run: ScientistRun }) {
  const t = useT();
  const metrics = headlineMetrics(run.metrics);
  if (metrics.length === 0) {
    return (
      <p className="text-muted-foreground text-sm">
        {t("scientist.detail.noMetrics")}
      </p>
    );
  }
  return (
    <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3">
      {metrics.map(({ name, value }) => (
        <div
          key={name}
          className="bg-muted/40 flex flex-col gap-0.5 rounded-xl px-3 py-2"
        >
          <dt className="text-muted-foreground truncate text-xs">{name}</dt>
          <dd className="font-medium tabular-nums">{value}</dd>
        </div>
      ))}
      <span className="sr-only">{t("scientist.detail.metrics")}</span>
    </dl>
  );
}

function StageProgress({
  progress,
}: {
  progress: Record<string, unknown> | null;
}) {
  const t = useT();
  if (!progress) {
    return null;
  }
  const stage = typeof progress.stage === "string" ? progress.stage : null;
  const total = typeof progress.total_nodes === "number" ? progress.total_nodes : null;
  const good = typeof progress.good_nodes === "number" ? progress.good_nodes : null;
  const best =
    typeof progress.best_metric === "string" && progress.best_metric !== "None"
      ? progress.best_metric
      : null;
  const findings =
    typeof progress.current_findings === "string" ? progress.current_findings : null;
  if (!stage && total === null) {
    return null;
  }
  return (
    <div className="flex flex-col gap-2 text-sm">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        {stage ? (
          <span className="font-medium">{t("scientist.detail.stage", { name: stage })}</span>
        ) : null}
        {total !== null ? (
          <span className="text-muted-foreground">
            {t("scientist.detail.nodes", { total, good: good ?? 0 })}
          </span>
        ) : null}
        {best ? (
          <span className="text-muted-foreground">
            {t("scientist.detail.bestMetric", { value: best })}
          </span>
        ) : null}
      </div>
      {findings ? (
        <p className="text-muted-foreground text-xs whitespace-pre-wrap">
          {findings}
        </p>
      ) : null}
    </div>
  );
}

export function ScientistRunDetail({
  stream,
  onCancel,
  onRetry,
  busy,
}: {
  stream: ScientistRunStream;
  onCancel: () => void;
  onRetry: () => void;
  busy: boolean;
}) {
  const t = useT();
  const locale = useLocale();
  const [copied, setCopied] = useState<string | null>(null);
  const run = stream.run;

  const copy = useCallback(async (label: string, value: string) => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(label);
      window.setTimeout(() => setCopied(null), 1500);
    } catch {
      toastError(t("scientist.detail.copyFailed"));
    }
  }, [t]);

  if (!run) {
    return (
      <Empty className="h-full border">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <Spinner />
          </EmptyMedia>
          <EmptyTitle>{t("scientist.detail.loading")}</EmptyTitle>
        </EmptyHeader>
      </Empty>
    );
  }

  const metric = primaryMetric(run.metrics);
  const duration = formatDurationMs(scientistRunDurationMs(run));

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-1">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="font-heading truncate text-base font-semibold tracking-tight">
              {run.config?.model}
            </h2>
            <ScientistRunStatusBadge status={run.status} />
            {!stream.connected ? (
              <span className="text-muted-foreground text-xs">
                {t("scientist.detail.reconnecting")}
              </span>
            ) : null}
          </div>
          <p className="text-muted-foreground text-xs">
            {t("scientist.detail.created", { when: formatWhen(run.createdAt, locale) })}
            {" · "}
            {t("scientist.detail.elapsed", { value: duration })}
            {metric ? ` · ${metric.name}=${metric.value}` : ""}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {canCancelScientistRun(run) ? (
            <Button
              variant="outline"
              size="sm"
              onClick={onCancel}
              disabled={busy || run.cancelRequested}
            >
              <HugeiconsIcon icon={Cancel01Icon} />
              {run.cancelRequested
                ? t("scientist.action.cancelling")
                : t("scientist.action.cancel")}
            </Button>
          ) : null}
          {canRetryScientistRun(run) ? (
            <Button size="sm" onClick={onRetry} disabled={busy}>
              {busy ? <Spinner /> : <HugeiconsIcon icon={RefreshIcon} />}
              {t("scientist.action.retry")}
            </Button>
          ) : null}
        </div>
      </div>

      {run.error ? (
        <div className="rounded-xl bg-destructive/10 text-destructive px-3 py-2 text-sm">
          {run.error}
        </div>
      ) : null}

      <Tabs defaultValue="log" className="flex min-h-0 flex-1 flex-col gap-3">
        <TabsList>
          <TabsTrigger value="log">{t("scientist.detail.tabLog")}</TabsTrigger>
          <TabsTrigger value="metrics">
            {t("scientist.detail.tabMetrics")}
          </TabsTrigger>
          <TabsTrigger value="config">
            {t("scientist.detail.tabConfig")}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="log" className="mt-0 min-h-0 flex-1">
          <StageProgress progress={stream.progress} />
          {stream.notes.length > 0 ? (
            <ul className="text-muted-foreground mt-2 flex list-disc flex-col gap-0.5 pl-5 text-xs">
              {stream.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          ) : null}
          <ScrollArea className="bg-muted/30 mt-3 h-[min(52vh,32rem)] rounded-xl">
            <pre className="p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap">
              {stream.log || t("scientist.detail.noLog")}
            </pre>
          </ScrollArea>
          {stream.truncated ? (
            <p className="text-muted-foreground mt-1 text-xs">
              {t("scientist.detail.logTruncated")}
            </p>
          ) : null}
        </TabsContent>

        <TabsContent value="metrics" className="mt-0">
          <MetricGrid run={run} />
        </TabsContent>

        <TabsContent value="config" className="mt-0">
          <div className="flex flex-col gap-3 text-sm">
            <dl className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {(
                [
                  ["ideaIdx", run.config?.ideaIdx],
                  ["attemptId", run.config?.attemptId],
                  ["loadCode", run.config?.loadCode],
                  ["loadIdeas", run.config?.loadIdeas],
                  ["config", run.config?.config],
                  ["timeoutSeconds", run.config?.timeoutSeconds ?? "—"],
                ] as const
              ).map(([name, value]) => (
                <div key={name} className="flex items-baseline gap-2">
                  <dt className="text-muted-foreground w-32 shrink-0 text-xs">{name}</dt>
                  <dd className="truncate font-mono text-xs">{String(value)}</dd>
                </div>
              ))}
            </dl>
            {stream.artifactDir ? (
              <>
                <Separator />
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground text-xs">
                    {t("scientist.detail.artifacts")}
                  </span>
                  <code className="bg-muted/50 min-w-0 flex-1 truncate rounded px-2 py-1 font-mono text-xs">
                    {stream.artifactDir}
                  </code>
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label={t("scientist.detail.copyPath")}
                    onClick={() => void copy("artifactDir", stream.artifactDir ?? "")}
                  >
                    <HugeiconsIcon
                      icon={copied === "artifactDir" ? LoadingIcon : Copy02Icon}
                    />
                  </Button>
                </div>
              </>
            ) : null}
            <Separator />
            <div className="flex flex-col gap-1">
              <span className="text-muted-foreground text-xs">argv</span>
              <code className="bg-muted/50 overflow-x-auto rounded px-2 py-1 font-mono text-xs whitespace-pre-wrap">
                {run.argv.join(" ")}
              </code>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
