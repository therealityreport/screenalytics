import type {
  ApiError,
  AssetUploadResponse,
  EpisodeCreateRequest,
  EpisodeCreateResponse,
  EpisodeEvent,
  EpisodeStatus,
  EpisodePhase,
} from "./types";

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000").replace(/\/$/, "");

export type JobTriggerResponse = {
  job_id?: string;
  status?: string;
  phase?: string;
};

function normalizeError(err: unknown): ApiError {
  if (typeof err === "object" && err !== null && "code" in err && "message" in err) {
    const maybe = err as { code?: string; message?: string; details?: unknown };
    return {
      code: maybe.code || "unknown_error",
      message: maybe.message || "Unknown error",
      details: maybe.details,
    };
  }

  if (err instanceof Error) {
    return { code: "error", message: err.message };
  }

  return { code: "error", message: "Unknown error" };
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = path.startsWith("http") ? path : `/api${path.startsWith("/") ? "" : "/"}${path}`;
  const response = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });

  if (!response.ok) {
    let payload: unknown;
    try {
      payload = await response.json();
    } catch (parseErr) {
      throw normalizeError(parseErr);
    }
    throw normalizeError(payload);
  }

  if (response.status === 204) {
    // @ts-expect-error allow void responses
    return undefined;
  }

  return (await response.json()) as T;
}

export async function createEpisode(payload: EpisodeCreateRequest) {
  return apiFetch<EpisodeCreateResponse>("/episodes", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function presignEpisodeAssets(epId: string): Promise<AssetUploadResponse> {
  return apiFetch<AssetUploadResponse>(`/episodes/${epId}/assets`, {
    method: "POST",
  });
}

export async function triggerJob(episodeId: string, phase: EpisodePhase): Promise<JobTriggerResponse> {
  const path =
    phase === "detect-track"
      ? "/jobs/detect_track"
      : phase === "faces"
        ? "/jobs/faces_embed"
        : phase === "cluster"
          ? "/jobs/cluster"
          : phase === "screentime"
            ? "/jobs/screen_time/analyze"
            : null;

  if (!path) {
    throw normalizeError({ code: "unsupported_phase", message: `Unsupported phase ${phase}` });
  }

  return apiFetch<JobTriggerResponse>(path, {
    method: "POST",
    body: JSON.stringify({ ep_id: episodeId }),
  });
}

export async function fetchEpisodeStatus(episodeId: string): Promise<EpisodeStatus> {
  return apiFetch<EpisodeStatus>(`/episodes/${episodeId}/status`);
}

export function eventsUrl(episodeId: string): string {
  return `/api/episodes/${episodeId}/events`;
}

export function mapEventStream(event: MessageEvent<string>): EpisodeEvent | null {
  try {
    const parsed = JSON.parse(event.data) as EpisodeEvent;
    return parsed;
  } catch (err) {
    console.warn("Failed to parse event", err);
    return null;
  }
}

export { normalizeError };
