import type {
  ApiError,
  AssetUploadResponse,
  AssignmentResponse,
  AssignTrackRequest,
  AudioPipelineRequest,
  AutoLinkCastResponse,
  BulkAssignmentResponse,
  BulkAssignRequest,
  CastMember,
  CastSuggestionsResponse,
  CleanupPreviewResponse,
  CleanupRequest,
  CleanupResponse,
  ClusterJobRequest,
  ClusterMetrics,
  ClusterTrackRepsResponse,
  DeleteFramesRequest,
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
  IdentitiesResponse,
  Job,
  JobsResponse,
  MoveFramesRequest,
  PeopleResponse,
  RefreshSimilarityResponse,
  ReviewProgress,
  S3VideosResponse,
  Show,
  ShowCreateRequest,
  StorageStatus,
  TimestampPreviewResponse,
  Track,
  TrackFramesResponse,
  TrackMetrics,
  UnlinkedEntitiesResponse,
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

// ============================================================================
// Faces Review API Functions
// ============================================================================

// Fetch cast members for a show
export async function fetchShowCast(
  showSlug: string,
  seasonLabel?: string
): Promise<CastMember[]> {
  const params = new URLSearchParams({ include_featured: "1" });
  if (seasonLabel) params.set("season", seasonLabel);
  const response = await apiFetch<{ cast: CastMember[] }>(
    `/shows/${showSlug}/cast?${params.toString()}`
  );
  return response.cast || [];
}

// Fetch people (cast members with their clusters) for a show
export async function fetchShowPeople(showSlug: string): Promise<PeopleResponse> {
  return apiFetch<PeopleResponse>(`/shows/${showSlug}/people`);
}

// Fetch identities (all clusters) for an episode
export async function fetchEpisodeIdentities(
  episodeId: string
): Promise<IdentitiesResponse> {
  return apiFetch<IdentitiesResponse>(`/episodes/${episodeId}/identities`);
}

// Fetch unlinked entities (needs assignment)
export async function fetchUnlinkedEntities(
  episodeId: string
): Promise<UnlinkedEntitiesResponse> {
  return apiFetch<UnlinkedEntitiesResponse>(`/episodes/${episodeId}/unlinked_entities`);
}

// Fetch track representatives for a cluster
export async function fetchClusterTrackReps(
  episodeId: string,
  clusterId: string,
  framesPerTrack: number = 0
): Promise<ClusterTrackRepsResponse> {
  const params = framesPerTrack > 0 ? `?frames_per_track=${framesPerTrack}` : "";
  return apiFetch<ClusterTrackRepsResponse>(
    `/episodes/${episodeId}/clusters/${clusterId}/track_reps${params}`
  );
}

// Fetch cluster metrics
export async function fetchClusterMetrics(
  episodeId: string,
  clusterId: string
): Promise<ClusterMetrics> {
  return apiFetch<ClusterMetrics>(
    `/episodes/${episodeId}/clusters/${clusterId}/metrics`
  );
}

// Fetch track detail
export async function fetchTrackDetail(
  episodeId: string,
  trackId: number
): Promise<Track> {
  return apiFetch<Track>(`/episodes/${episodeId}/tracks/${trackId}`);
}

// Fetch track metrics
export async function fetchTrackMetrics(
  episodeId: string,
  trackId: number
): Promise<TrackMetrics> {
  return apiFetch<TrackMetrics>(
    `/episodes/${episodeId}/tracks/${trackId}/metrics`
  );
}

// Fetch track frames (paginated)
export async function fetchTrackFrames(
  episodeId: string,
  trackId: number,
  options?: {
    page?: number;
    pageSize?: number;
    sample?: number;
    includeSkipped?: boolean;
  }
): Promise<TrackFramesResponse> {
  const params = new URLSearchParams();
  if (options?.page) params.set("page", String(options.page));
  if (options?.pageSize) params.set("page_size", String(options.pageSize));
  if (options?.sample) params.set("sample", String(options.sample));
  if (options?.includeSkipped) params.set("include_skipped", "true");

  const queryString = params.toString();
  return apiFetch<TrackFramesResponse>(
    `/episodes/${episodeId}/tracks/${trackId}/frames${queryString ? `?${queryString}` : ""}`
  );
}

// Fetch cast suggestions for episode
export async function fetchCastSuggestions(
  episodeId: string
): Promise<CastSuggestionsResponse> {
  return apiFetch<CastSuggestionsResponse>(
    `/episodes/${episodeId}/cast_suggestions`
  );
}

// Assign track to a name/cast
export async function assignTrack(
  episodeId: string,
  trackId: number,
  payload: AssignTrackRequest
): Promise<AssignmentResponse> {
  return apiFetch<AssignmentResponse>(
    `/episodes/${episodeId}/tracks/${trackId}/name`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );
}

// Bulk assign tracks
export async function bulkAssignTracks(
  episodeId: string,
  payload: BulkAssignRequest
): Promise<BulkAssignmentResponse> {
  return apiFetch<BulkAssignmentResponse>(
    `/episodes/${episodeId}/tracks/bulk_assign`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );
}

// Save identity name
export async function saveIdentityName(
  episodeId: string,
  identityId: string,
  name: string,
  show?: string
): Promise<AssignmentResponse> {
  const payload: { name: string; show?: string } = { name };
  if (show) payload.show = show;
  return apiFetch<AssignmentResponse>(
    `/episodes/${episodeId}/identities/${identityId}/name`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );
}

// Move frames between tracks/identities
export async function moveFrames(
  episodeId: string,
  trackId: number,
  payload: MoveFramesRequest
): Promise<{ moved: number; target_name?: string; target_identity_id?: string }> {
  return apiFetch<{ moved: number; target_name?: string; target_identity_id?: string }>(
    `/episodes/${episodeId}/tracks/${trackId}/frames/move`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );
}

// Delete frames from a track
export async function deleteFrames(
  episodeId: string,
  trackId: number,
  payload: DeleteFramesRequest
): Promise<{ deleted: number }> {
  return apiFetch<{ deleted: number }>(
    `/episodes/${episodeId}/tracks/${trackId}/frames`,
    {
      method: "DELETE",
      body: JSON.stringify(payload),
    }
  );
}

// Refresh similarity scores
export async function refreshSimilarity(
  episodeId: string
): Promise<RefreshSimilarityResponse> {
  return apiFetch<RefreshSimilarityResponse>(
    `/episodes/${episodeId}/refresh_similarity`,
    { method: "POST" }
  );
}

// Auto-link clusters to cast
export async function autoLinkCast(
  episodeId: string
): Promise<AutoLinkCastResponse> {
  return apiFetch<AutoLinkCastResponse>(
    `/episodes/${episodeId}/auto_link_cast`,
    { method: "POST" }
  );
}

// Get cleanup preview
export async function fetchCleanupPreview(
  episodeId: string
): Promise<CleanupPreviewResponse> {
  return apiFetch<CleanupPreviewResponse>(
    `/episodes/${episodeId}/cleanup_preview`
  );
}

// Run cleanup job
export async function runCleanup(
  episodeId: string,
  payload: CleanupRequest
): Promise<CleanupResponse> {
  return apiFetch<CleanupResponse>(
    `/jobs/episode_cleanup_async`,
    {
      method: "POST",
      body: JSON.stringify({ ep_id: episodeId, ...payload }),
    }
  );
}

// Create backup before cleanup
export async function createBackup(
  episodeId: string
): Promise<{ backup_id: string }> {
  return apiFetch<{ backup_id: string }>(
    `/episodes/${episodeId}/backup`,
    { method: "POST" }
  );
}

// Restore from backup
export async function restoreBackup(
  episodeId: string,
  backupId: string
): Promise<{ files_restored: number }> {
  return apiFetch<{ files_restored: number }>(
    `/episodes/${episodeId}/restore/${backupId}`,
    { method: "POST" }
  );
}

// List backups
export async function fetchBackups(
  episodeId: string
): Promise<{ backups: Array<{ backup_id: string; created_at?: string }> }> {
  return apiFetch<{ backups: Array<{ backup_id: string; created_at?: string }> }>(
    `/episodes/${episodeId}/backups`
  );
}

// Save assignments (persist to manifest)
export async function saveAssignments(
  episodeId: string
): Promise<{ saved_count: number }> {
  return apiFetch<{ saved_count: number }>(
    `/episodes/${episodeId}/save_assignments`,
    { method: "POST" }
  );
}

// Fetch review progress stats
export async function fetchReviewProgress(
  episodeId: string
): Promise<ReviewProgress> {
  // Derive from identities endpoint
  const identities = await fetchEpisodeIdentities(episodeId);
  const assignedCount = identities.identities.filter(i => i.is_assigned).length;
  const unassignedCount = identities.identities.filter(i => !i.is_assigned).length;
  const singletonCount = identities.identities.filter(i => i.track_count === 1).length;
  const total = identities.identities.length;

  return {
    total_clusters: total,
    assigned_clusters: assignedCount,
    unassigned_clusters: unassignedCount,
    total_tracks: identities.total_tracks,
    singleton_count: singletonCount,
    percent_complete: total > 0 ? (assignedCount / total) * 100 : 0,
  };
}

// Fetch roster names for autocomplete
export async function fetchRosterNames(showSlug: string): Promise<string[]> {
  const response = await apiFetch<{ names: string[] }>(
    `/shows/${showSlug}/cast_names`
  );
  return response.names || [];
}

// Create new cast member
export async function createCastMember(
  showSlug: string,
  name: string
): Promise<{ cast_id: string }> {
  return apiFetch<{ cast_id: string }>(
    `/shows/${showSlug}/cast`,
    {
      method: "POST",
      body: JSON.stringify({ name }),
    }
  );
}

// ============================================================================
// Improve Faces API (Post-Cluster Suggestions)
// ============================================================================

export type ImproveFacesSuggestion = {
  cluster_a: {
    id: string;
    crop_url?: string;
    tracks: number;
    faces: number;
  };
  cluster_b: {
    id: string;
    crop_url?: string;
    tracks: number;
    faces: number;
  };
  similarity: number;
};

export type ImproveFacesSuggestionsResponse = {
  suggestions: ImproveFacesSuggestion[];
  initial_pass_done: boolean;
};

export type FaceReviewDecision = "merge" | "reject";

export type FaceReviewDecisionRequest = {
  pair_type: "unassigned_unassigned" | "assigned_unassigned";
  cluster_a_id: string;
  cluster_b_id: string;
  decision: FaceReviewDecision;
  execution_mode?: "local" | "redis";
};

// Fetch initial unassigned suggestions for improve faces modal
export async function fetchImproveFacesSuggestions(
  episodeId: string
): Promise<ImproveFacesSuggestionsResponse> {
  return apiFetch<ImproveFacesSuggestionsResponse>(
    `/episodes/${episodeId}/face_review/initial_unassigned_suggestions`
  );
}

// Mark initial pass done (user has reviewed all suggestions or skipped)
export async function markInitialPassDone(
  episodeId: string
): Promise<void> {
  await apiFetch<void>(
    `/episodes/${episodeId}/face_review/mark_initial_pass_done`,
    { method: "POST", body: JSON.stringify({}) }
  );
}

// Submit face review decision (merge or reject)
export async function submitFaceReviewDecision(
  episodeId: string,
  payload: FaceReviewDecisionRequest
): Promise<{ status: "queued" | "success"; job_id?: string }> {
  return apiFetch<{ status: "queued" | "success"; job_id?: string }>(
    `/episodes/${episodeId}/face_review/decision/start`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );
}
