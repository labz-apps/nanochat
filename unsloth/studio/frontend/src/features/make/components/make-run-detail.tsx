// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Empty, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useLocale, useT } from "@/i18n";
import { toastError } from "@/shared/toast";
import { Cancel01Icon, Copy02Icon, RefreshIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useState } from "react";
import { MakeRunStatusBadge } from "./make-run-status-badge";
import {
  canCancelMakeRun,
  canRetryMakeRun,
  formatDurationMs,
  formatWhen,
  headlineMetrics,
  makeRunDurationMs,
} from "../lib/run-display";
import type { MakeRunStream } from "../hooks/use-make-run-stream";
import type { MakeRun } from "../types/api";

function MetricGrid({ run }: { run: MakeRun }) {
  const t = useT();
  const metrics = headlineMetrics(run.metrics);
  if (metrics.length === 0) {
    return <p className="text-muted-foreground text-sm">{t("make.detail.noMetrics")}</p>;
  }
  return (
    <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3">
      {metrics.map(({ name, value }) => (
        <div key={name} className="bg-muted/40 flex flex-col gap-0.5 rounded-xl px-3 py-2">
          <dt className="text-muted-foreground truncate text-xs">{name}</dt>
          <dd className="font-medium tabular-nums">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

/** How far through its pinned iteration budget a run is, when it has said. */
function StepProgress({ stream }: { stream: MakeRunStream }) {
  const t = useT();
  const step = stream.progress?.step;
  const total = stream.run?.config?.iterations;
  if (typeof step !== "number" || !total) {
    return null;
  }
  const percent = Math.min(100, Math.round((step / total) * 100));
  return (
    <div className="flex flex-col gap-1.5">
      <div className="text-muted-foreground flex items-center justify-between text-xs">
        <span>{t("make.detail.step", { step, total })}</span>
        <span>{percent}%</span>
      </div>
      <Progress value={percent} />
    </div>
  );
}

export function MakeRunDetail({
  stream,
  onCancel,
  onRetry,
  busy,
}: {
  stream: MakeRunStream;
  onCancel: () => void;
  onRetry: () => void;
  busy: boolean;
}) {
  const t = useT();
  const locale = useLocale();
  const [copied, setCopied] = useState(false);
  const run = stream.run;

  const copy = useCallback(
    async (value: string) => {
      try {
        await navigator.clipboard.writeText(value);
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1500);
      } catch {
        toastError(t("make.detail.copyFailed"));
      }
    },
    [t],
  );

  if (!run) {
    return (
      <Empty className="h-full border">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <Spinner />
          </EmptyMedia>
          <EmptyTitle>{t("make.detail.loading")}</EmptyTitle>
        </EmptyHeader>
      </Empty>
    );
  }

  const config = run.config;

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-1">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="font-heading truncate text-base font-semibold tracking-tight">
              {config?.runTag}
            </h2>
            <MakeRunStatusBadge status={run.status} />
            {!stream.connected ? (
              <span className="text-muted-foreground text-xs">
                {t("make.detail.reconnecting")}
              </span>
            ) : null}
          </div>
          <p className="text-muted-foreground text-xs">
            {t("make.detail.created", { when: formatWhen(run.createdAt, locale) })}
            {" · "}
            {t("make.detail.elapsed", { value: formatDurationMs(makeRunDurationMs(run)) })}
            {` · depth ${config?.depth} · ${config?.iterations} steps`}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {canCancelMakeRun(run) ? (
            <Button
              variant="outline"
              size="sm"
              onClick={onCancel}
              disabled={busy || run.cancelRequested}
            >
              <HugeiconsIcon icon={Cancel01Icon} />
              {run.cancelRequested ? t("make.action.cancelling") : t("make.action.cancel")}
            </Button>
          ) : null}
          {canRetryMakeRun(run) ? (
            <Button size="sm" onClick={onRetry} disabled={busy}>
              {busy ? <Spinner /> : <HugeiconsIcon icon={RefreshIcon} />}
              {t("make.action.retry")}
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
          <TabsTrigger value="log">{t("make.detail.tabLog")}</TabsTrigger>
          <TabsTrigger value="metrics">{t("make.detail.tabMetrics")}</TabsTrigger>
          <TabsTrigger value="config">{t("make.detail.tabCommand")}</TabsTrigger>
        </TabsList>

        <TabsContent value="log" className="mt-0 min-h-0 flex-1">
          <StepProgress stream={stream} />
          {stream.notes.length > 0 ? (
            <ul className="text-muted-foreground mt-2 flex list-disc flex-col gap-0.5 pl-5 text-xs">
              {stream.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          ) : null}
          <ScrollArea className="bg-muted/30 mt-3 h-[min(52vh,32rem)] rounded-xl">
            <pre className="p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap">
              {stream.log || t("make.detail.noLog")}
            </pre>
          </ScrollArea>
          {stream.truncated ? (
            <p className="text-muted-foreground mt-1 text-xs">
              {t("make.detail.logTruncated")}
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
                  ["preset", config?.preset],
                  ["depth", config?.depth],
                  ["iterations", config?.iterations],
                  ["aspectRatio", config?.aspectRatio],
                  ["headDim", config?.headDim],
                  ["maxSeqLen", config?.maxSeqLen],
                  ["deviceBatchSize", config?.deviceBatchSize],
                  ["totalBatchSize", config?.totalBatchSize],
                  ["evaluate", config?.evaluate],
                ] as const
              ).map(([name, value]) => (
                <div key={name} className="flex items-baseline gap-2">
                  <dt className="text-muted-foreground w-32 shrink-0 text-xs">{name}</dt>
                  <dd className="truncate font-mono text-xs">{String(value)}</dd>
                </div>
              ))}
            </dl>
            {stream.checkpointDir ? (
              <>
                <Separator />
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground text-xs">
                    {t("make.detail.checkpoint")}
                  </span>
                  <code className="bg-muted/50 min-w-0 flex-1 truncate rounded px-2 py-1 font-mono text-xs">
                    {stream.checkpointDir}
                  </code>
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label={t("make.detail.copyPath")}
                    onClick={() => void copy(stream.checkpointDir ?? "")}
                  >
                    <HugeiconsIcon icon={Copy02Icon} data-copied={copied} />
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
