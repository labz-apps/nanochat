// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { useT } from "@/i18n";
import { toastError } from "@/shared/toast";
import { useCallback, useState } from "react";
import type {
  CreateScientistRunInput,
  ScientistEnvironment,
} from "../types/api";

/** The launcher's own defaults, restated so the form opens on what it would use. */
const DEFAULT_MODEL = "opencode/space-bunny-free";
const DEFAULT_IDEAS = "ai_scientist/ideas/nanochat_pretraining.json";
const DEFAULT_CONFIG = "bfts_config.yaml";

export function NewScientistRunDialog({
  open,
  onOpenChange,
  environment,
  busy,
  onCreate,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  environment: ScientistEnvironment | null;
  busy: boolean;
  onCreate: (input: CreateScientistRunInput) => Promise<void>;
}) {
  const t = useT();
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [loadIdeas, setLoadIdeas] = useState(DEFAULT_IDEAS);
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [ideaIdx, setIdeaIdx] = useState("0");
  const [attemptId, setAttemptId] = useState("0");
  const [loadCode, setLoadCode] = useState(true);
  // Provider calls cost money and reach the network, so this starts off and the button
  // stays closed until it is turned on. The launcher refuses to run without it too.
  const [allowProviderCalls, setAllowProviderCalls] = useState(false);

  const submit = useCallback(async () => {
    try {
      await onCreate({
        model: model.trim(),
        ideaIdx: Number.parseInt(ideaIdx, 10) || 0,
        attemptId: Number.parseInt(attemptId, 10) || 0,
        loadCode,
        loadIdeas: loadIdeas.trim(),
        config: config.trim(),
        allowProviderCalls: true,
      });
      onOpenChange(false);
    } catch (cause) {
      toastError(
        t("scientist.new.failed"),
        cause instanceof Error ? cause.message : String(cause),
      );
    }
  }, [
    attemptId,
    config,
    ideaIdx,
    loadCode,
    loadIdeas,
    model,
    onCreate,
    onOpenChange,
    t,
  ]);

  const ready =
    environment?.available === true &&
    model.trim().length > 0 &&
    allowProviderCalls &&
    !busy;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{t("scientist.new.title")}</DialogTitle>
          <DialogDescription>{t("scientist.new.description")}</DialogDescription>
        </DialogHeader>

        {environment && !environment.available ? (
          <div className="rounded-xl bg-destructive/10 flex flex-col gap-1 px-3 py-2 text-sm">
            <span className="text-destructive font-medium">
              {t("scientist.new.blockedTitle")}
            </span>
            <ul className="text-destructive/90 flex list-disc flex-col gap-1 pl-4 text-xs">
              {(environment.problems ?? []).map((problem) => (
                <li key={problem.code}>{problem.fix}</li>
              ))}
            </ul>
          </div>
        ) : null}

        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="scientist-model">{t("scientist.new.model")}</Label>
            <Input
              id="scientist-model"
              value={model}
              onChange={(event) => setModel(event.target.value)}
              placeholder={DEFAULT_MODEL}
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="scientist-idea-idx">
                {t("scientist.new.ideaIndex")}
              </Label>
              <Input
                id="scientist-idea-idx"
                type="number"
                min={0}
                value={ideaIdx}
                onChange={(event) => setIdeaIdx(event.target.value)}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="scientist-attempt-id">
                {t("scientist.new.attemptId")}
              </Label>
              <Input
                id="scientist-attempt-id"
                type="number"
                min={0}
                value={attemptId}
                onChange={(event) => setAttemptId(event.target.value)}
              />
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <Label htmlFor="scientist-ideas">{t("scientist.new.ideasFile")}</Label>
            <Input
              id="scientist-ideas"
              value={loadIdeas}
              onChange={(event) => setLoadIdeas(event.target.value)}
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <Label htmlFor="scientist-config">
              {t("scientist.new.configFile")}
            </Label>
            <Input
              id="scientist-config"
              value={config}
              onChange={(event) => setConfig(event.target.value)}
            />
          </div>

          <div className="flex items-center justify-between gap-3 rounded-xl border px-3 py-2">
            <Label htmlFor="scientist-load-code" className="font-normal">
              {t("scientist.new.loadCode")}
            </Label>
            <Switch
              id="scientist-load-code"
              checked={loadCode}
              onCheckedChange={setLoadCode}
            />
          </div>

          <div className="flex items-center justify-between gap-3 rounded-xl border px-3 py-2">
            <Label htmlFor="scientist-provider-calls" className="font-normal">
              {t("scientist.new.allowProviderCalls")}
            </Label>
            <Switch
              id="scientist-provider-calls"
              checked={allowProviderCalls}
              onCheckedChange={setAllowProviderCalls}
            />
          </div>

          {environment?.projectRoot ? (
            <p className="text-muted-foreground truncate font-mono text-xs">
              {environment.projectRoot}
            </p>
          ) : null}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t("common.cancel")}
          </Button>
          <Button disabled={!ready} onClick={() => void submit()}>
            {busy ? <Spinner /> : null}
            {t("scientist.action.start")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
