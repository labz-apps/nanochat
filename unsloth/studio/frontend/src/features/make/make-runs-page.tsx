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
import { MakeEnvironmentNotice } from "./components/make-environment-notice";
import { MakeRunDetail } from "./components/make-run-detail";
import { MakeRunList } from "./components/make-run-list";
import { NewMakeRunDialog } from "./components/new-make-run-dialog";
import { useMakeEnvironment, useMakeRuns } from "./hooks/use-make-runs";
import { useMakeRunStream } from "./hooks/use-make-run-stream";
import type { CreateMakeRunInput } from "./types/api";

type MakeRunsSearch = { run?: string };

export function MakeRunsPage() {
  const t = useT();
  const onShellReady = useAppShellReadySignal();
  const search = useSearch({ strict: false }) as MakeRunsSearch;
  const { runs, loading, error, creating, cancel, retry, create, refresh } = useMakeRuns();
  const { environment } = useMakeEnvironment();
  const [chosenId, setChosenId] = useState<string | null>(search.run ?? null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [busyAction, setBusyAction] = useState(false);

  // Derived, not synchronised: the chosen run when it is still there, otherwise the one
  // that is training, otherwise the newest. An effect that repaired this would render
  // once with the old selection and again with the new one.
  const selectedId =
    chosenId && runs.some((run) => run.id === chosenId)
      ? chosenId
      : (runs.find((run) => ["queued", "running", "cancelling"].includes(run.status))?.id ??
        runs[0]?.id ??
        null);
  const stream = useMakeRunStream(selectedId);

  useEffect(() => {
    onShellReady();
  }, [onShellReady]);

  const guard = useCallback(
    async (action: () => Promise<void>) => {
      setBusyAction(true);
      try {
        await action();
      } catch (cause) {
        toastError(t("make.action.failed"), cause instanceof Error ? cause.message : String(cause));
      } finally {
        setBusyAction(false);
      }
    },
    [t],
  );

  const start = useCallback(
    (input: CreateMakeRunInput) =>
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
          <h1 className="font-heading text-xl font-semibold tracking-tight">{t("make.title")}</h1>
          <p className="text-muted-foreground text-sm">{t("make.description")}</p>
        </div>
        {loading && runs.length === 0 ? <Spinner /> : null}
      </header>

      {environment && !environment.available ? (
        <MakeEnvironmentNotice environment={environment} />
      ) : null}

      {error ? (
        <Alert variant="destructive">
          <HugeiconsIcon icon={InformationCircleIcon} />
          <AlertTitle>{t("make.list.loadFailed")}</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-5 lg:grid-cols-[minmax(16rem,20rem)_minmax(0,1fr)]">
        <MakeRunList
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
            <MakeRunDetail
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
              {t("make.detail.selectRun")}
            </div>
          )}
        </div>
      </div>

      <NewMakeRunDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        environment={environment}
        busy={creating}
        onCreate={start}
      />
    </div>
  );
}
