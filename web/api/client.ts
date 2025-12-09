import type {
  ApiError,
  AssetUploadResponse,
  AudioPipelineRequest,
  ClusterJobRequest,
  DetectTrackJobRequest,
  EpisodeArtifactStatus,
  EpisodeCreateRequest,
  EpisodeCreateResponse,
  EpisodeDetail,
  EpisodeDetailResponse,
  EpisodeEvent,
  EpisodeListResponse,
  EpisodeQuickStats,
  EpisodeStatus,
  EpisodePhase,
  EpisodeSummary,
  FacesJobRequest,
  Job,
  JobsResponse,
  S3VideosResponse,
  Show,
  ShowCreateRequest,
  StorageStatus,
  TimestampPreviewResponse,
  VideoMeta,
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

// Shows API
export async function fetchShows(): Promise<Show[]> {
  const response = await apiFetch<{ shows: Show[] } | Show[]>("/shows");
  return Array.isArray(response) ? response : response.shows || [];
}

export async function createShow(payload: ShowCreateRequest): Promise<Show> {
  return apiFetch<Show>("/shows", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// S3 Videos API
export async function fetchS3Videos(): Promise<S3VideosResponse> {
  return apiFetch<S3VideosResponse>("/episodes/s3_videos");
}

// Jobs API
export async function fetchJobs(episodeId?: string): Promise<Job[]> {
  const path = episodeId ? `/jobs?ep_id=${episodeId}` : "/jobs";
  const response = await apiFetch<JobsResponse | Job[]>(path);
  return Array.isArray(response) ? response : response.jobs || [];
}

export async function fetchAllRunningJobs(): Promise<Job[]> {
  const response = await apiFetch<JobsResponse | Job[]>("/jobs?state=running");
  return Array.isArray(response) ? response : response.jobs || [];
}

export async function fetchJobProgress(jobId: string): Promise<Job> {
  return apiFetch<Job>(`/jobs/${jobId}/progress`);
}

export async function cancelJob(jobId: string): Promise<void> {
  await apiFetch<void>(`/jobs/${jobId}/cancel`, { method: "POST" });
}

// Episode Detail API
export async function fetchEpisodeDetail(episodeId: string): Promise<EpisodeDetail> {
  return apiFetch<EpisodeDetail>(`/episodes/${episodeId}`);
}

export async function deleteEpisode(episodeId: string): Promise<void> {
  await apiFetch<void>(`/episodes/${episodeId}`, { method: "DELETE" });
}

// Mirror from S3
export async function mirrorEpisodeFromS3(episodeId: string): Promise<{ local_video_path: string; bytes?: number }> {
  return apiFetch<{ local_video_path: string; bytes?: number }>(`/episodes/${episodeId}/mirror`, {
    method: "POST",
  });
}

// Audio Pipeline
export async function triggerAudioPipeline(payload: AudioPipelineRequest): Promise<JobTriggerResponse> {
  return apiFetch<JobTriggerResponse>("/jobs/episode_audio_pipeline", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// Upsert episode by ID (for S3 browser "create in store" action)
export async function upsertEpisodeById(payload: {
  ep_id: string;
  show_slug: string;
  season: number;
  episode: number;
}): Promise<{ ep_id: string; created: boolean }> {
  return apiFetch<{ ep_id: string; created: boolean }>("/episodes/upsert_by_id", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// Episodes List
export async function fetchEpisodes(): Promise<EpisodeSummary[]> {
  const response = await apiFetch<EpisodeListResponse>("/episodes");
  return response.episodes || [];
}

// Bulk delete episodes
export async function bulkDeleteEpisodes(
  episodeIds: string[],
  includeS3: boolean = true
): Promise<{ deleted: number; errors: string[] }> {
  const results = await Promise.allSettled(
    episodeIds.map((epId) =>
      apiFetch<void>(`/episodes/${epId}/delete`, {
        method: "POST",
        body: JSON.stringify({ include_s3: includeS3 }),
      })
    )
  );

  const errors: string[] = [];
  let deleted = 0;

  results.forEach((result, idx) => {
    if (result.status === "fulfilled") {
      deleted++;
    } else {
      errors.push(`${episodeIds[idx]}: ${result.reason?.message || "Unknown error"}`);
    }
  });

  return { deleted, errors };
}

// Timestamp preview (for featured thumbnail selection)
export async function fetchTimestampPreview(
  episodeId: string,
  timestampS: number
): Promise<TimestampPreviewResponse> {
  return apiFetch<TimestampPreviewResponse>(
    `/episodes/${episodeId}/timestamp/${timestampS}/preview`
  );
}

// Set featured thumbnail for episode
export async function setFeaturedThumbnail(
  episodeId: string,
  timestampS: number
): Promise<{ url: string }> {
  return apiFetch<{ url: string }>(`/episodes/${episodeId}/featured_thumbnail`, {
    method: "POST",
    body: JSON.stringify({ timestamp_s: timestampS }),
  });
}

// ============================================================================
// Episode Detail API Functions
// ============================================================================

// Fetch extended episode details
export async function fetchEpisodeDetails(
  episodeId: string
): Promise<EpisodeDetailResponse> {
  return apiFetch<EpisodeDetailResponse>(`/episodes/${episodeId}`);
}

// Fetch video metadata
export async function fetchVideoMeta(
  episodeId: string
): Promise<VideoMeta | null> {
  try {
    return await apiFetch<VideoMeta>(`/episodes/${episodeId}/video_meta`);
  } catch {
    return null;
  }
}

// Fetch job history for episode
export async function fetchEpisodeJobHistory(
  episodeId: string,
  limit: number = 5
): Promise<Job[]> {
  const response = await apiFetch<JobsResponse>(`/jobs?ep_id=${episodeId}&limit=${limit}`);
  return response.jobs || [];
}

// Trigger detect/track with custom settings
export async function triggerDetectTrack(
  payload: DetectTrackJobRequest
): Promise<JobTriggerResponse> {
  return apiFetch<JobTriggerResponse>("/jobs/detect_track", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// Trigger faces harvest with custom settings
export async function triggerFacesEmbed(
  payload: FacesJobRequest
): Promise<JobTriggerResponse> {
  return apiFetch<JobTriggerResponse>("/jobs/faces_embed", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// Trigger clustering with custom settings
export async function triggerCluster(
  payload: ClusterJobRequest
): Promise<JobTriggerResponse> {
  return apiFetch<JobTriggerResponse>("/jobs/cluster", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// Fetch artifact status
export async function fetchArtifactStatus(
  episodeId: string
): Promise<EpisodeArtifactStatus | null> {
  try {
    return await apiFetch<EpisodeArtifactStatus>(`/episodes/${episodeId}/artifact_status`);
  } catch {
    return null;
  }
}

// Fetch storage configuration
export async function fetchStorageConfig(): Promise<StorageStatus | null> {
  try {
    return await apiFetch<StorageStatus>("/config/storage");
  } catch {
    return null;
  }
}

// Fetch quick stats for episode
export async function fetchEpisodeQuickStats(
  episodeId: string
): Promise<EpisodeQuickStats | null> {
  try {
    // This would need a dedicated endpoint, for now derive from status
    const status = await fetchEpisodeStatus(episodeId);
    return {
      tracks_count: status.detect_track?.tracks ?? 0,
      identities_count: status.cluster?.identities ?? 0,
      assigned_count: 0, // Would need additional endpoint
      unassigned_count: 0,
      singleton_before: 0,
      singleton_after: 0,
      screen_time_calculated: false,
    };
  } catch {
    return null;
  }
}

export { normalizeError };
