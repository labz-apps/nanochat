// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAppShellReadySignal } from "@/components/app-readiness";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Spinner } from "@/components/ui/spinner";
import { useT } from "@/i18n";
import { toastError } from "@/shared/toast";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useSearch } from "@tanstack/react-router";
import { useCallback, useEffect, useState } from "react";
import { NewScientistRunDialog } from "./components/new-scientist-run-dialog";
import { ScientistEnvironmentNotice } from "./components/scientist-environment-notice";
import { ScientistRunDetail } from "./components/scientist-run-detail";
import { ScientistRunList } from "./components/scientist-run-list";
import {
  useScientistEnvironment,
  useScientistRuns,
} from "./hooks/use-scientist-runs";
import { useScientistRunStream } from "./hooks/use-scientist-run-stream";
import type { CreateScientistRunInput } from "./types/api";

type ScientistRunsSearch = { run?: string };

export function ScientistRunsPage() {
  const t = useT();
  const onShellReady = useAppShellReadySignal();
  const search = useSearch({ strict: false }) as ScientistRunsSearch;
  const { runs, loading, error, creating, cancel, retry, create, refresh } =
    useScientistRuns();
  const { environment } = useScientistEnvironment();
  const [chosenId, setChosenId] = useState<string | null>(search.run ?? null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [busyAction, setBusyAction] = useState(false);

  // Derived, not synchronised: the chosen run when it is still there, otherwise the
  // one that is working, otherwise the newest. An effect that repaired this would
  // render once with the old selection and then again with the new one.
  const selectedId =
    chosenId && runs.some((run) => run.id === chosenId)
      ? chosenId
      : (runs.find((run) =>
          ["queued", "running", "cancelling"].includes(run.status),
        )?.id ??
        runs[0]?.id ??
        null);
  const stream = useScientistRunStream(selectedId);

  useEffect(() => {
    onShellReady();
  }, [onShellReady]);

  const guard = useCallback(
    async (action: () => Promise<void>) => {
      setBusyAction(true);
      try {
        await action();
      } catch (cause) {
        toastError(
          t("scientist.action.failed"),
          cause instanceof Error ? cause.message : String(cause),
        );
      } finally {
        setBusyAction(false);
      }
    },
    [t],
  );

  const start = useCallback(
    (input: CreateScientistRunInput) =>
      guard(async () => {
        const run = await create(input);
        if (run) {
          setChosenId(run.id);
        }
      }),
    [create, guard],
  );

  const canStart = environment?.available === true && !creating;

  return (
    <div className="flex h-full min-h-0 flex-col gap-5 p-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex flex-col gap-1">
          <h1 className="font-heading text-xl font-semibold tracking-tight">
            {t("scientist.title")}
          </h1>
          <p className="text-muted-foreground text-sm">
            {t("scientist.description")}
          </p>
        </div>
        {loading && runs.length === 0 ? <Spinner /> : null}
      </header>

      {environment && !environment.available ? (
        <ScientistEnvironmentNotice environment={environment} />
      ) : null}

      {error ? (
        <Alert variant="destructive">
          <HugeiconsIcon icon={InformationCircleIcon} />
          <AlertTitle>{t("scientist.list.loadFailed")}</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-5 lg:grid-cols-[minmax(16rem,20rem)_minmax(0,1fr)]">
        <ScientistRunList
          runs={runs}
          selectedId={selectedId}
          onSelect={setChosenId}
          loading={loading}
          onNew={() => setDialogOpen(true)}
          canStart={canStart}
          starting={creating}
        />
        <div className="min-h-0">
          {selectedId ? (
            <ScientistRunDetail
              stream={stream}
              busy={busyAction}
              onCancel={() =>
                void guard(async () => {
                  await cancel(selectedId);
                  refresh();
                })
              }
              onRetry={() =>
                void guard(async () => {
                  await retry(selectedId);
                  refresh();
                })
              }
            />
          ) : (
            <div className="text-muted-foreground flex h-full items-center justify-center text-sm">
              {t("scientist.detail.selectRun")}
            </div>
          )}
        </div>
      </div>

      <NewScientistRunDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        environment={environment}
        busy={creating}
        onCreate={start}
      />
    </div>
  );
}
