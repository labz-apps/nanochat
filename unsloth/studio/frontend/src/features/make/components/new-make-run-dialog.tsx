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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { useT } from "@/i18n";
import { toastError } from "@/shared/toast";
import { useCallback, useMemo, useState, type ChangeEvent } from "react";
import {
  FALLBACK_PRESET,
  estimateParameters,
  formatParameterCount,
  isMakePresetId,
  validateMakeForm,
} from "../lib/run-display";
import type { CreateMakeRunInput, MakeEnvironment, MakePreset } from "../types/api";

type FormValues = {
  depth: number;
  iterations: number;
  deviceBatchSize: number;
  totalBatchSize: number;
  maxSeqLen: number;
};

const valuesOf = (preset: MakePreset): FormValues => ({
  depth: preset.depth,
  iterations: preset.iterations,
  deviceBatchSize: preset.deviceBatchSize,
  totalBatchSize: preset.totalBatchSize,
  maxSeqLen: preset.maxSeqLen,
});

export function NewMakeRunDialog({
  open,
  onOpenChange,
  environment,
  busy,
  onCreate,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  environment: MakeEnvironment | null;
  busy: boolean;
  onCreate: (input: CreateMakeRunInput) => Promise<void>;
}) {
  const t = useT();
  const presets = useMemo(
    () => (environment?.presets?.length ? environment.presets : [FALLBACK_PRESET]),
    [environment],
  );
  const [presetId, setPresetId] = useState(FALLBACK_PRESET.id);
  const [values, setValues] = useState<FormValues>(() => valuesOf(FALLBACK_PRESET));
  const [runTag, setRunTag] = useState("");
  const [evaluate, setEvaluate] = useState(true);

  // Switching preset overwrites the overrides: leaving a depth-6 batch size behind on a
  // depth-4 model is exactly the kind of thing that trains something other than what
  // the dialog says.
  const choosePreset = useCallback(
    (id: string) => {
      setPresetId(id);
      const preset = presets.find((candidate) => candidate.id === id) ?? FALLBACK_PRESET;
      setValues(valuesOf(preset));
    },
    [presets],
  );

  const set = useCallback(
    (key: keyof FormValues) => (event: ChangeEvent<HTMLInputElement>) => {
      const parsed = Number.parseInt(event.target.value, 10);
      setValues((previous) => ({ ...previous, [key]: Number.isNaN(parsed) ? 0 : parsed }));
    },
    [],
  );

  const problem = validateMakeForm({ preset: presetId, ...values });
  const params = estimateParameters(values.depth, FALLBACK_PRESET.aspectRatio);

  const submit = useCallback(async () => {
    try {
      await onCreate({
        preset: presetId,
        ...values,
        runTag: runTag.trim() || undefined,
        evaluate,
      });
      onOpenChange(false);
    } catch (cause) {
      toastError(t("make.new.failed"), cause instanceof Error ? cause.message : String(cause));
    }
  }, [evaluate, onCreate, onOpenChange, presetId, runTag, t, values]);

  const ready = environment?.available === true && problem === null && !busy;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{t("make.new.title")}</DialogTitle>
          <DialogDescription>{t("make.new.description")}</DialogDescription>
        </DialogHeader>

        {environment && !environment.available ? (
          <div className="rounded-xl bg-destructive/10 flex flex-col gap-1 px-3 py-2 text-sm">
            <span className="text-destructive font-medium">{t("make.new.blockedTitle")}</span>
            <ul className="text-destructive/90 flex list-disc flex-col gap-1 pl-4 text-xs">
              {(environment.problems ?? []).map((entry) => (
                <li key={entry.code}>{entry.fix}</li>
              ))}
            </ul>
          </div>
        ) : null}

        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label>{t("make.new.preset")}</Label>
            <Select value={presetId} onValueChange={choosePreset}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {presets.map((preset) => (
                  <SelectItem key={preset.id} value={preset.id}>
                    {isMakePresetId(preset.id)
                      ? t(`make.preset.${preset.id}`)
                      : preset.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-muted-foreground text-xs">
              {t("make.new.approxParams", { count: formatParameterCount(params) })}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="make-depth">{t("make.new.depth")}</Label>
              <Input id="make-depth" type="number" value={values.depth} onChange={set("depth")} />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="make-iterations">{t("make.new.iterations")}</Label>
              <Input
                id="make-iterations"
                type="number"
                value={values.iterations}
                onChange={set("iterations")}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="make-device-batch">{t("make.new.deviceBatch")}</Label>
              <Input
                id="make-device-batch"
                type="number"
                value={values.deviceBatchSize}
                onChange={set("deviceBatchSize")}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="make-total-batch">{t("make.new.totalBatch")}</Label>
              <Input
                id="make-total-batch"
                type="number"
                value={values.totalBatchSize}
                onChange={set("totalBatchSize")}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="make-seq-len">{t("make.new.maxSeqLen")}</Label>
              <Input
                id="make-seq-len"
                type="number"
                value={values.maxSeqLen}
                onChange={set("maxSeqLen")}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="make-run-tag">{t("make.new.runTag")}</Label>
              <Input
                id="make-run-tag"
                value={runTag}
                onChange={(event) => setRunTag(event.target.value)}
                placeholder="make"
              />
            </div>
          </div>

          <div className="flex items-center justify-between gap-3 rounded-xl border px-3 py-2">
            <Label htmlFor="make-evaluate" className="font-normal">
              {t("make.new.evaluate")}
            </Label>
            <Switch id="make-evaluate" checked={evaluate} onCheckedChange={setEvaluate} />
          </div>

          {problem ? <p className="text-destructive text-xs">{problem}</p> : null}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t("common.cancel")}
          </Button>
          <Button disabled={!ready} onClick={() => void submit()}>
            {busy ? <Spinner /> : null}
            {t("make.action.start")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
