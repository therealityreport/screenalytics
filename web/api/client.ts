import type {
  ApiErrorEnvelope,
  AssetUploadResponse,
  EpisodeCreateRequest,
  EpisodeCreateResponse,
  EpisodeEvent,
  EpisodePhase,
  EpisodeStatus,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "");

export type JobTriggerResponse = {
  job_id?: string;
  status?: string;
  phase?: string;
};

function buildUrl(path: string): string {
  if (path.startsWith("http")) return path;
  const normalized = path.startsWith("/") ? path : `/${path}`;
  if (API_BASE) {
    return `${API_BASE}${normalized}`;
  }
  // Default: rely on Next.js rewrite /api -> backend during local dev
  return `/api${normalized}`;
}

function ensureEnvelope(err: unknown, fallbackCode = "UNKNOWN_ERROR"): ApiErrorEnvelope {
  if (typeof err === "object" && err !== null && "code" in err && "message" in err) {
    const maybe = err as { code?: string; message?: string; details?: unknown };
    return {
      code: maybe.code || fallbackCode,
      message: maybe.message || "Unknown error",
      details: maybe.details,
    };
  }
  if (err instanceof Error) {
    return { code: fallbackCode, message: err.message };
  }
  return { code: fallbackCode, message: "Unknown error" };
}

async function parseErrorResponse(response: Response): Promise<ApiErrorEnvelope> {
  try {
    const payload = await response.json();
    return ensureEnvelope(payload, `HTTP_${response.status}`);
  } catch (parseErr) {
    return ensureEnvelope(parseErr, `HTTP_${response.status}`);
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(buildUrl(path), {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers || {}),
      },
    });
  } catch (networkErr) {
    throw ensureEnvelope(networkErr, "NETWORK_ERROR");
  }

  if (!response.ok) {
    throw await parseErrorResponse(response);
  }

  if (response.status === 204) {
    // @ts-expect-error allow void responses
    return undefined;
  }

  try {
    return (await response.json()) as T;
  } catch (parseErr) {
    throw ensureEnvelope(parseErr, "PARSE_ERROR");
  }
}

export const apiClient = {
  get: <T>(path: string, init?: RequestInit) => apiFetch<T>(path, { ...init, method: "GET" }),
  post: <TBody, TResponse>(path: string, body?: TBody, init?: RequestInit) =>
    apiFetch<TResponse>(path, {
      ...init,
      method: "POST",
      body: body !== undefined ? JSON.stringify(body) : init?.body,
    }),
};

export function normalizeError(err: unknown): ApiErrorEnvelope {
  return ensureEnvelope(err, "UNKNOWN_ERROR");
}

export async function createEpisode(payload: EpisodeCreateRequest) {
  return apiClient.post<EpisodeCreateRequest, EpisodeCreateResponse>("/episodes", payload);
}

export async function presignEpisodeAssets(epId: string): Promise<AssetUploadResponse> {
  return apiClient.post<undefined, AssetUploadResponse>(`/episodes/${epId}/assets`);
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
    throw normalizeError({ code: "UNSUPPORTED_PHASE", message: `Unsupported phase ${phase}` });
  }

  return apiClient.post<{ ep_id: string }, JobTriggerResponse>(path, { ep_id: episodeId });
}

export async function fetchEpisodeStatus(episodeId: string): Promise<EpisodeStatus> {
  return apiClient.get<EpisodeStatus>(`/episodes/${episodeId}/status`);
}

export function eventsUrl(episodeId: string): string {
  return buildUrl(`/episodes/${episodeId}/events`);
}

export function mapEventStream(event: MessageEvent<string>): EpisodeEvent | null {
  try {
    return JSON.parse(event.data) as EpisodeEvent;
  } catch (err) {
    console.warn("Failed to parse event", err);
    return null;
  }
}
