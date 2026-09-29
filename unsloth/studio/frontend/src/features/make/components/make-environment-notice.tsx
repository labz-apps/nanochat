// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { useT } from "@/i18n";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { MakeEnvironment } from "../types/api";

/**
 * Why a run cannot start here, one remedy per line.
 *
 * A single "not available" sentence leaves the reader to work out which of the cache's
 * three assets is missing, which is the single most common thing to be wrong here.
 */
export function MakeEnvironmentNotice({ environment }: { environment: MakeEnvironment }) {
  const t = useT();
  const problems = environment.problems ?? [];
  if (environment.available && problems.length === 0) {
    return null;
  }
  return (
    <Alert variant="destructive">
      <HugeiconsIcon icon={InformationCircleIcon} />
      <AlertTitle>{t("make.environment.unavailable")}</AlertTitle>
      <AlertDescription>
        <ul className="flex list-disc flex-col gap-2 pl-4">
          {problems.map((problem) => (
            <li key={problem.code}>
              <span className="font-medium">{problem.message}</span>
              <span className="text-destructive/80 block">{problem.fix}</span>
            </li>
          ))}
        </ul>
        {environment.projectRoot ? (
          <>
            <Separator className="my-2" />
            <p className="text-destructive/80 font-mono text-xs break-all">
              {environment.projectRoot}
            </p>
          </>
        ) : null}
      </AlertDescription>
    </Alert>
  );
}
