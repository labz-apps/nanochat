// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- Avoid the auth barrel's React login page.
import { authFetch } from "@/features/auth/api";
import type {
  CreateScientistRunInput,
  ScientistEnvironment,
  ScientistRun,
  ScientistRunEvent,
  ScientistRunListResponse,
  ScientistRunLogsResponse,
} from "../types/api";

type JsonObject = Record<string, unknown>;

export class ScientistApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ScientistApiError";
    this.status = status;
  }
}

function camelize(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(camelize);
  }
  if (!value || typeof value !== "object") {
    return value;
  }
  return Object.fromEntries(
    Object.entries(value as JsonObject).map(([key, child]) => [
      key.replace(/_([a-z])/g, (_, letter: string) => letter.toUpperCase()),
      camelize(child),
    ]),
  );
}

async function json<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = (body as { detail?: unknown } | null)?.detail;
    throw new ScientistApiError(
      typeof detail === "string"
        ? detail
        : `AI Scientist request failed (${response.status})`,
      response.status,
    );
  }
  return camelize(body) as T;
}

const runPath = (runId: string, suffix = ""): string =>
  `/api/scientist-runs/${encodeURIComponent(runId)}${suffix}`;

export async function getScientistEnvironment(
  signal?: AbortSignal,
): Promise<ScientistEnvironment> {
  return json<ScientistEnvironment>(
    await authFetch("/api/scientist-runs/environment", { signal }),
  );
}

export async function listScientistRuns(
  options: { limit?: number; offset?: number; status?: string } = {},
  signal?: AbortSignal,
): Promise<ScientistRunListResponse> {
  const query = new URLSearchParams();
  if (options.limit !== undefined) query.set("limit", String(options.limit));
  if (options.offset !== undefined) query.set("offset", String(options.offset));
  if (options.status) query.set("status", options.status);
  const suffix = query.size > 0 ? `?${query.toString()}` : "";
  return json<ScientistRunListResponse>(
    await authFetch(`/api/scientist-runs${suffix}`, { signal }),
  );
}

export async function getScientistRun(
  runId: string,
  signal?: AbortSignal,
): Promise<ScientistRun> {
  return json<ScientistRun>(await authFetch(runPath(runId), { signal }));
}

export async function getScientistRunLogs(
  runId: string,
  after = 0,
  signal?: AbortSignal,
): Promise<ScientistRunLogsResponse> {
  return json<ScientistRunLogsResponse>(
    await authFetch(runPath(runId, `/logs?after=${Math.max(0, after)}`), { signal }),
  );
}

export async function createScientistRun(
  input: CreateScientistRunInput,
): Promise<ScientistRun> {
  return json<ScientistRun>(
    await authFetch("/api/scientist-runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
}

const post = async (runId: string, action: "cancel" | "retry") =>
  json<ScientistRun>(
    await authFetch(runPath(runId, `/${action}`), { method: "POST" }),
  );

export const cancelScientistRun = (runId: string): Promise<ScientistRun> =>
  post(runId, "cancel");

export const retryScientistRun = (runId: string): Promise<ScientistRun> =>
  post(runId, "retry");

// biome-ignore lint/complexity/noExcessiveCognitiveComplexity: Incremental SSE parsing must retain framing state across reader chunks.
export async function* streamScientistRunEvents(
  runId: string,
  after: number,
  signal?: AbortSignal,
): AsyncGenerator<ScientistRunEvent> {
  // POST, not GET: proxies hold a streamed GET until it closes. The route accepts both.
  const response = await authFetch(
    runPath(runId, `/events?after=${Math.max(0, after)}`),
    { method: "POST", headers: { accept: "text/event-stream" }, signal },
  );
  if (!response.ok) {
    await json(response);
  }
  if (!response.body) {
    throw new Error("AI Scientist event stream returned no response body");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      // Normalize on the whole buffer so a CRLF split across chunks still frames.
      buffer = buffer.replace(/\r\n/g, "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        let event = "message";
        let eventId = after;
        const data: string[] = [];
        for (const line of block.split("\n")) {
          if (line.startsWith("id:")) {
            eventId = Number(line.slice(3).trim()) || eventId;
          } else if (line.startsWith("event:")) {
            event = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            data.push(line.slice(5).trimStart());
          }
        }
        if (data.length > 0) {
          const parsed = camelize(JSON.parse(data.join("\n"))) as JsonObject;
          const snapshot = parsed.run as ScientistRun | undefined;
          yield {
            id: eventId,
            event: event as ScientistRunEvent["event"],
            createdAt:
              typeof parsed.createdAt === "number"
                ? parsed.createdAt
                : (snapshot?.updatedAt ?? Date.now()),
            attempt: typeof parsed.attempt === "number" ? parsed.attempt : 0,
            data: parsed,
            ...(snapshot?.id && snapshot.status ? { run: snapshot } : {}),
          };
        }
        boundary = buffer.indexOf("\n\n");
      }
      if (done) {
        return;
      }
    }
  } finally {
    await reader.cancel().catch(() => undefined);
  }
}
