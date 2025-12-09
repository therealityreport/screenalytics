"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  assignTrack,
  autoLinkCast,
  bulkAssignTracks,
  bulkDeleteEpisodes,
  cancelJob,
  createBackup,
  createCastMember,
  createEpisode,
  createShow,
  deleteEpisode,
  deleteFrames,
  eventsUrl,
  fetchAllRunningJobs,
  fetchArtifactStatus,
  fetchBackups,
  fetchCastSuggestions,
  fetchCleanupPreview,
  fetchClusterMetrics,
  fetchClusterTrackReps,
  fetchEpisodeDetail,
  fetchEpisodeDetails,
  fetchEpisodeIdentities,
  fetchEpisodeJobHistory,
  fetchEpisodes,
  fetchEpisodeStatus,
  fetchJobs,
  fetchReviewProgress,
  fetchRosterNames,
  fetchS3Videos,
  fetchShowCast,
  fetchShowPeople,
  fetchShows,
  fetchStorageConfig,
  fetchTimestampPreview,
  fetchTrackDetail,
  fetchTrackFrames,
  fetchTrackMetrics,
  fetchUnlinkedEntities,
  fetchVideoMeta,
  mapEventStream,
  mirrorEpisodeFromS3,
  moveFrames,
  presignEpisodeAssets,
  refreshSimilarity,
  restoreBackup,
  runCleanup,
  saveAssignments,
  saveIdentityName,
  setFeaturedThumbnail,
  triggerAudioPipeline,
  triggerCluster,
  triggerDetectTrack,
  triggerFacesEmbed,
  triggerJob,
  upsertEpisodeById,
} from "./client";
import type {
  ApiError,
  AssignmentResponse,
  AssignTrackRequest,
  AssetUploadResponse,
  AudioPipelineRequest,
  AutoLinkCastResponse,
  BulkAssignmentResponse,
  BulkAssignRequest,
  CastMember,
  CastSuggestionsResponse,
  CleanupAction,
  CleanupPreviewResponse,
  CleanupResponse,
  ClusterJobRequest,
  ClusterMetrics,
  ClusterTrackRepsResponse,
  DeleteFramesRequest,
  DetectTrackJobRequest,
  EpisodeArtifactStatus,
  EpisodeCreateRequest,
  EpisodeDetail,
  EpisodeDetailResponse,
  EpisodeEvent,
  EpisodePhase,
  EpisodeStatus,
  EpisodeSummary,
  FacesJobRequest,
  FacesReviewView,
  IdentitiesResponse,
  Identity,
  Job,
  MoveFramesRequest,
  PeopleResponse,
  PipelineSettings,
  RefreshSimilarityResponse,
  ReviewProgress,
  S3VideoItem,
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
import { DEFAULT_PIPELINE_SETTINGS } from "./types";

export function useEpisodeStatus(
  episodeId?: string,
  options?: { enabled?: boolean; refetchInterval?: number },
) {
  return useQuery<EpisodeStatus, ApiError>({
    queryKey: ["episode-status", episodeId],
    queryFn: () => fetchEpisodeStatus(episodeId as string),
    enabled: Boolean(episodeId) && (options?.enabled ?? true),
    refetchInterval: options?.refetchInterval ?? false,
  });
}

export function useCreateEpisode() {
  return useMutation<{ ep_id: string }, ApiError, EpisodeCreateRequest>({
    mutationFn: (payload) => createEpisode(payload),
  });
}

export function usePresignAssets() {
  return useMutation<AssetUploadResponse, ApiError, { epId: string }>({
    mutationFn: ({ epId }) => presignEpisodeAssets(epId),
  });
}

export function useTriggerPhase() {
  const client = useQueryClient();
  return useMutation<unknown, ApiError, { episodeId: string; phase: EpisodePhase }>({
    mutationFn: ({ episodeId, phase }) => triggerJob(episodeId, phase),
    onSuccess: (_data, variables) => {
      client.invalidateQueries({ queryKey: ["episode-status", variables.episodeId] });
    },
  });
}

export function useEpisodeEvents(
  episodeId?: string,
  handlers?: {
    onEvent?: (event: EpisodeEvent) => void;
    onError?: (err: Event) => void;
  },
) {
  // Use ref to avoid re-creating EventSource when handlers change
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;

  useEffect(() => {
    if (!episodeId) return undefined;

    if (process.env.NEXT_PUBLIC_MSW === "1") {
      const timer = setInterval(() => {
        handlersRef.current?.onEvent?.({
          episode_id: episodeId,
          phase: "detect-track",
          event: "progress",
          message: "mock event",
          progress: Math.random(),
        });
      }, 3000);
      return () => clearInterval(timer);
    }

    const url = eventsUrl(episodeId);
    const source = new EventSource(url);

    const onMessage = (evt: MessageEvent<string>) => {
      const parsed = mapEventStream(evt);
      if (parsed && handlersRef.current?.onEvent) {
        handlersRef.current.onEvent(parsed);
      }
    };
    const onError = (evt: Event) => {
      handlersRef.current?.onError?.(evt);
    };

    source.addEventListener("message", onMessage as EventListener);
    source.addEventListener("error", onError);

    return () => {
      source.removeEventListener("message", onMessage as EventListener);
      source.removeEventListener("error", onError);
      source.close();
    };
  }, [episodeId]);
}

// Shows hooks
export function useShows() {
  return useQuery<Show[], ApiError>({
    queryKey: ["shows"],
    queryFn: fetchShows,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

export function useCreateShow() {
  const client = useQueryClient();
  return useMutation<Show, ApiError, ShowCreateRequest>({
    mutationFn: createShow,
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["shows"] });
    },
  });
}

// S3 Videos hooks
export function useS3Videos(options?: { enabled?: boolean }) {
  return useQuery<S3VideoItem[], ApiError>({
    queryKey: ["s3-videos"],
    queryFn: async () => {
      const response = await fetchS3Videos();
      return response.items || [];
    },
    enabled: options?.enabled ?? true,
    staleTime: 30 * 1000, // 30 seconds
  });
}

// Jobs hooks
export function useJobs(episodeId?: string, options?: { enabled?: boolean; refetchInterval?: number }) {
  return useQuery<Job[], ApiError>({
    queryKey: ["jobs", episodeId],
    queryFn: () => fetchJobs(episodeId),
    enabled: options?.enabled ?? true,
    refetchInterval: options?.refetchInterval,
  });
}

export function useRunningJobs(options?: { refetchInterval?: number }) {
  return useQuery<Job[], ApiError>({
    queryKey: ["jobs", "running"],
    queryFn: fetchAllRunningJobs,
    refetchInterval: options?.refetchInterval ?? 2000, // Poll every 2 seconds
  });
}

export function useCancelJob() {
  const client = useQueryClient();
  return useMutation<void, ApiError, string>({
    mutationFn: cancelJob,
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

// Episode Detail hooks
export function useEpisodeDetail(episodeId?: string, options?: { enabled?: boolean }) {
  return useQuery<EpisodeDetail, ApiError>({
    queryKey: ["episode-detail", episodeId],
    queryFn: () => fetchEpisodeDetail(episodeId as string),
    enabled: Boolean(episodeId) && (options?.enabled ?? true),
  });
}

export function useDeleteEpisode() {
  const client = useQueryClient();
  return useMutation<void, ApiError, string>({
    mutationFn: deleteEpisode,
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["episodes"] });
      client.invalidateQueries({ queryKey: ["s3-videos"] });
    },
  });
}

// Mirror from S3
export function useMirrorFromS3() {
  const client = useQueryClient();
  return useMutation<{ local_video_path: string; bytes?: number }, ApiError, string>({
    mutationFn: mirrorEpisodeFromS3,
    onSuccess: (_data, episodeId) => {
      client.invalidateQueries({ queryKey: ["episode-detail", episodeId] });
      client.invalidateQueries({ queryKey: ["episode-status", episodeId] });
    },
  });
}

// Audio Pipeline
export function useTriggerAudioPipeline() {
  const client = useQueryClient();
  return useMutation<{ job_id?: string }, ApiError, AudioPipelineRequest>({
    mutationFn: triggerAudioPipeline,
    onSuccess: (_data, variables) => {
      client.invalidateQueries({ queryKey: ["jobs"] });
      client.invalidateQueries({ queryKey: ["episode-status", variables.ep_id] });
    },
  });
}

// Upsert Episode (for S3 browser)
export function useUpsertEpisode() {
  const client = useQueryClient();
  return useMutation<
    { ep_id: string; created: boolean },
    ApiError,
    { ep_id: string; show_slug: string; season: number; episode: number }
  >({
    mutationFn: upsertEpisodeById,
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["s3-videos"] });
      client.invalidateQueries({ queryKey: ["episodes"] });
    },
  });
}

// Episodes List
export function useEpisodes(options?: { enabled?: boolean }) {
  return useQuery<EpisodeSummary[], ApiError>({
    queryKey: ["episodes"],
    queryFn: fetchEpisodes,
    enabled: options?.enabled ?? true,
    staleTime: 30 * 1000, // 30 seconds
  });
}

// Bulk delete episodes
export function useBulkDeleteEpisodes() {
  const client = useQueryClient();
  return useMutation<
    { deleted: number; errors: string[] },
    ApiError,
    { episodeIds: string[]; includeS3?: boolean }
  >({
    mutationFn: ({ episodeIds, includeS3 }) => bulkDeleteEpisodes(episodeIds, includeS3),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["episodes"] });
      client.invalidateQueries({ queryKey: ["s3-videos"] });
    },
  });
}

// Timestamp preview for featured thumbnail
export function useTimestampPreview(
  episodeId: string,
  timestampS: number,
  options?: { enabled?: boolean }
) {
  return useQuery<TimestampPreviewResponse, ApiError>({
    queryKey: ["timestamp-preview", episodeId, timestampS],
    queryFn: () => fetchTimestampPreview(episodeId, timestampS),
    enabled: options?.enabled ?? true,
    staleTime: 60 * 1000, // 1 minute
  });
}

// Set featured thumbnail
export function useSetFeaturedThumbnail() {
  const client = useQueryClient();
  return useMutation<{ url: string }, ApiError, { episodeId: string; timestampS: number }>({
    mutationFn: ({ episodeId, timestampS }) => setFeaturedThumbnail(episodeId, timestampS),
    onSuccess: (_data, variables) => {
      client.invalidateQueries({ queryKey: ["episodes"] });
      client.invalidateQueries({ queryKey: ["episode-detail", variables.episodeId] });
    },
  });
}

// Batch fetch episode statuses
export function useEpisodeStatuses(
  episodeIds: string[],
  options?: { enabled?: boolean; refetchInterval?: number }
) {
  return useQuery<Map<string, EpisodeStatus>, ApiError>({
    queryKey: ["episode-statuses", episodeIds.sort().join(",")],
    queryFn: async () => {
      const results = await Promise.allSettled(
        episodeIds.map((epId) => fetchEpisodeStatus(epId))
      );
      const statusMap = new Map<string, EpisodeStatus>();
      results.forEach((result, idx) => {
        if (result.status === "fulfilled") {
          statusMap.set(episodeIds[idx], result.value);
        }
      });
      return statusMap;
    },
    enabled: (options?.enabled ?? true) && episodeIds.length > 0,
    refetchInterval: options?.refetchInterval,
    staleTime: 10 * 1000, // 10 seconds
  });
}

// Local storage helpers for favorites and recent episodes
const FAVORITES_KEY = "screenalytics_favorites";
const RECENT_KEY = "screenalytics_recent";
const MAX_RECENT = 10;

export function useFavorites() {
  const getFavorites = useCallback((): string[] => {
    if (typeof window === "undefined") return [];
    try {
      const stored = localStorage.getItem(FAVORITES_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }, []);

  const addFavorite = useCallback((epId: string) => {
    if (typeof window === "undefined") return;
    const favorites = getFavorites();
    if (!favorites.includes(epId)) {
      localStorage.setItem(FAVORITES_KEY, JSON.stringify([epId, ...favorites]));
    }
  }, [getFavorites]);

  const removeFavorite = useCallback((epId: string) => {
    if (typeof window === "undefined") return;
    const favorites = getFavorites().filter((id) => id !== epId);
    localStorage.setItem(FAVORITES_KEY, JSON.stringify(favorites));
  }, [getFavorites]);

  const isFavorite = useCallback(
    (epId: string) => getFavorites().includes(epId),
    [getFavorites]
  );

  return { getFavorites, addFavorite, removeFavorite, isFavorite };
}

export function useRecentEpisodes() {
  const getRecent = useCallback((): string[] => {
    if (typeof window === "undefined") return [];
    try {
      const stored = localStorage.getItem(RECENT_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }, []);

  const addRecent = useCallback((epId: string) => {
    if (typeof window === "undefined") return;
    const recent = getRecent().filter((id) => id !== epId);
    const updated = [epId, ...recent].slice(0, MAX_RECENT);
    localStorage.setItem(RECENT_KEY, JSON.stringify(updated));
  }, [getRecent]);

  const clearRecent = useCallback(() => {
    if (typeof window === "undefined") return;
    localStorage.removeItem(RECENT_KEY);
  }, []);

  return { getRecent, addRecent, clearRecent };
}

// ============================================================================
// Episode Detail Hooks
// ============================================================================

// Extended episode details
export function useEpisodeDetails(
  episodeId: string,
  options?: { enabled?: boolean }
) {
  return useQuery<EpisodeDetailResponse, ApiError>({
    queryKey: ["episode-details", episodeId],
    queryFn: () => fetchEpisodeDetails(episodeId),
    enabled: (options?.enabled ?? true) && !!episodeId,
    staleTime: 30 * 1000,
  });
}

// Video metadata
export function useVideoMeta(
  episodeId: string,
  options?: { enabled?: boolean }
) {
  return useQuery<VideoMeta | null, ApiError>({
    queryKey: ["video-meta", episodeId],
    queryFn: () => fetchVideoMeta(episodeId),
    enabled: (options?.enabled ?? true) && !!episodeId,
    staleTime: 60 * 1000,
  });
}

// Job history for episode
export function useEpisodeJobHistory(
  episodeId: string,
  options?: { enabled?: boolean; limit?: number }
) {
  return useQuery<Job[], ApiError>({
    queryKey: ["episode-job-history", episodeId, options?.limit ?? 5],
    queryFn: () => fetchEpisodeJobHistory(episodeId, options?.limit ?? 5),
    enabled: (options?.enabled ?? true) && !!episodeId,
    staleTime: 10 * 1000,
  });
}

// Artifact status
export function useArtifactStatus(
  episodeId: string,
  options?: { enabled?: boolean }
) {
  return useQuery<EpisodeArtifactStatus | null, ApiError>({
    queryKey: ["artifact-status", episodeId],
    queryFn: () => fetchArtifactStatus(episodeId),
    enabled: (options?.enabled ?? true) && !!episodeId,
    staleTime: 30 * 1000,
  });
}

// Storage configuration
export function useStorageConfig(options?: { enabled?: boolean }) {
  return useQuery<StorageStatus | null, ApiError>({
    queryKey: ["storage-config"],
    queryFn: fetchStorageConfig,
    enabled: options?.enabled ?? true,
    staleTime: 60 * 1000,
  });
}

// Trigger detect/track with settings
export function useTriggerDetectTrack() {
  const client = useQueryClient();
  return useMutation<
    { job_id?: string; status?: string },
    ApiError,
    DetectTrackJobRequest
  >({
    mutationFn: triggerDetectTrack,
    onSuccess: (_data, variables) => {
      client.invalidateQueries({ queryKey: ["episode-status", variables.ep_id] });
      client.invalidateQueries({ queryKey: ["episode-job-history", variables.ep_id] });
    },
  });
}

// Trigger faces harvest with settings
export function useTriggerFacesEmbed() {
  const client = useQueryClient();
  return useMutation<
    { job_id?: string; status?: string },
    ApiError,
    FacesJobRequest
  >({
    mutationFn: triggerFacesEmbed,
    onSuccess: (_data, variables) => {
      client.invalidateQueries({ queryKey: ["episode-status", variables.ep_id] });
      client.invalidateQueries({ queryKey: ["episode-job-history", variables.ep_id] });
    },
  });
}

// Trigger clustering with settings
export function useTriggerCluster() {
  const client = useQueryClient();
  return useMutation<
    { job_id?: string; status?: string },
    ApiError,
    ClusterJobRequest
  >({
    mutationFn: triggerCluster,
    onSuccess: (_data, variables) => {
      client.invalidateQueries({ queryKey: ["episode-status", variables.ep_id] });
      client.invalidateQueries({ queryKey: ["episode-job-history", variables.ep_id] });
    },
  });
}

// Pipeline settings (localStorage)
const PIPELINE_SETTINGS_KEY = "screenalytics_pipeline_settings";

export function usePipelineSettings() {
  const getSettings = useCallback((): PipelineSettings => {
    if (typeof window === "undefined") return DEFAULT_PIPELINE_SETTINGS;
    try {
      const stored = localStorage.getItem(PIPELINE_SETTINGS_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        return { ...DEFAULT_PIPELINE_SETTINGS, ...parsed };
      }
    } catch {
      // Ignore parse errors
    }
    return DEFAULT_PIPELINE_SETTINGS;
  }, []);

  const saveSettings = useCallback((settings: Partial<PipelineSettings>) => {
    if (typeof window === "undefined") return;
    const current = getSettings();
    const updated = { ...current, ...settings };
    localStorage.setItem(PIPELINE_SETTINGS_KEY, JSON.stringify(updated));
  }, [getSettings]);

  const resetSettings = useCallback(() => {
    if (typeof window === "undefined") return;
    localStorage.removeItem(PIPELINE_SETTINGS_KEY);
  }, []);

  return { getSettings, saveSettings, resetSettings };
}

// ============================================================================
// Faces Review Hooks
// ============================================================================

// Show cast members
export function useShowCast(showSlug?: string, seasonLabel?: string) {
  return useQuery<CastMember[], ApiError>({
    queryKey: ["show-cast", showSlug, seasonLabel],
    queryFn: () => fetchShowCast(showSlug!, seasonLabel),
    enabled: !!showSlug,
    staleTime: 15 * 1000, // 15 seconds (may be mutated by assignments)
  });
}

// Show people (cast with clusters)
export function useShowPeople(showSlug?: string) {
  return useQuery<PeopleResponse, ApiError>({
    queryKey: ["show-people", showSlug],
    queryFn: () => fetchShowPeople(showSlug!),
    enabled: !!showSlug,
    staleTime: 15 * 1000,
  });
}

// Episode identities (all clusters)
export function useEpisodeIdentities(episodeId: string, options?: { enabled?: boolean }) {
  return useQuery<IdentitiesResponse, ApiError>({
    queryKey: ["episode-identities", episodeId],
    queryFn: () => fetchEpisodeIdentities(episodeId),
    enabled: (options?.enabled ?? true) && !!episodeId,
    staleTime: 15 * 1000,
  });
}

// Unlinked entities (needs assignment)
export function useUnlinkedEntities(episodeId: string, options?: { enabled?: boolean }) {
  return useQuery<UnlinkedEntitiesResponse, ApiError>({
    queryKey: ["unlinked-entities", episodeId],
    queryFn: () => fetchUnlinkedEntities(episodeId),
    enabled: (options?.enabled ?? true) && !!episodeId,
    staleTime: 15 * 1000,
  });
}

// Cluster track representatives
export function useClusterTrackReps(
  episodeId: string,
  clusterId?: string,
  framesPerTrack: number = 0
) {
  return useQuery<ClusterTrackRepsResponse, ApiError>({
    queryKey: ["cluster-track-reps", episodeId, clusterId, framesPerTrack],
    queryFn: () => fetchClusterTrackReps(episodeId, clusterId!, framesPerTrack),
    enabled: !!episodeId && !!clusterId,
    staleTime: 60 * 1000,
  });
}

// Cluster metrics
export function useClusterMetrics(episodeId: string, clusterId?: string) {
  return useQuery<ClusterMetrics, ApiError>({
    queryKey: ["cluster-metrics", episodeId, clusterId],
    queryFn: () => fetchClusterMetrics(episodeId, clusterId!),
    enabled: !!episodeId && !!clusterId,
    staleTime: 60 * 1000,
  });
}

// Track detail
export function useTrackDetail(episodeId: string, trackId?: number) {
  return useQuery<Track, ApiError>({
    queryKey: ["track-detail", episodeId, trackId],
    queryFn: () => fetchTrackDetail(episodeId, trackId!),
    enabled: !!episodeId && trackId !== undefined,
    staleTime: 60 * 1000,
  });
}

// Track metrics
export function useTrackMetrics(episodeId: string, trackId?: number) {
  return useQuery<TrackMetrics, ApiError>({
    queryKey: ["track-metrics", episodeId, trackId],
    queryFn: () => fetchTrackMetrics(episodeId, trackId!),
    enabled: !!episodeId && trackId !== undefined,
    staleTime: 60 * 1000,
  });
}

// Track frames (paginated)
export function useTrackFrames(
  episodeId: string,
  trackId?: number,
  options?: {
    page?: number;
    pageSize?: number;
    sample?: number;
    includeSkipped?: boolean;
    enabled?: boolean;
  }
) {
  return useQuery<TrackFramesResponse, ApiError>({
    queryKey: [
      "track-frames",
      episodeId,
      trackId,
      options?.page,
      options?.pageSize,
      options?.sample,
      options?.includeSkipped,
    ],
    queryFn: () =>
      fetchTrackFrames(episodeId, trackId!, {
        page: options?.page,
        pageSize: options?.pageSize,
        sample: options?.sample,
        includeSkipped: options?.includeSkipped,
      }),
    enabled: (options?.enabled ?? true) && !!episodeId && trackId !== undefined,
    staleTime: 60 * 1000,
  });
}

// Cast suggestions
export function useCastSuggestions(episodeId: string, options?: { enabled?: boolean }) {
  return useQuery<CastSuggestionsResponse, ApiError>({
    queryKey: ["cast-suggestions", episodeId],
    queryFn: () => fetchCastSuggestions(episodeId),
    enabled: (options?.enabled ?? true) && !!episodeId,
    staleTime: 30 * 1000,
  });
}

// Review progress
export function useReviewProgress(episodeId: string, options?: { enabled?: boolean }) {
  return useQuery<ReviewProgress, ApiError>({
    queryKey: ["review-progress", episodeId],
    queryFn: () => fetchReviewProgress(episodeId),
    enabled: (options?.enabled ?? true) && !!episodeId,
    staleTime: 15 * 1000,
  });
}

// Roster names for autocomplete
export function useRosterNames(showSlug?: string) {
  return useQuery<string[], ApiError>({
    queryKey: ["roster-names", showSlug],
    queryFn: () => fetchRosterNames(showSlug!),
    enabled: !!showSlug,
    staleTime: 60 * 1000,
  });
}

// Cleanup preview
export function useCleanupPreview(episodeId: string, options?: { enabled?: boolean }) {
  return useQuery<CleanupPreviewResponse, ApiError>({
    queryKey: ["cleanup-preview", episodeId],
    queryFn: () => fetchCleanupPreview(episodeId),
    enabled: (options?.enabled ?? true) && !!episodeId,
    staleTime: 30 * 1000,
  });
}

// Backups list
export function useBackups(episodeId: string, options?: { enabled?: boolean }) {
  return useQuery<{ backups: Array<{ backup_id: string; created_at?: string }> }, ApiError>({
    queryKey: ["backups", episodeId],
    queryFn: () => fetchBackups(episodeId),
    enabled: (options?.enabled ?? true) && !!episodeId,
    staleTime: 30 * 1000,
  });
}

// Assign track mutation
export function useAssignTrack() {
  const client = useQueryClient();
  return useMutation<
    AssignmentResponse,
    ApiError,
    { episodeId: string; trackId: number; payload: AssignTrackRequest }
  >({
    mutationFn: ({ episodeId, trackId, payload }) => assignTrack(episodeId, trackId, payload),
    onSuccess: (_data, variables) => {
      client.invalidateQueries({ queryKey: ["episode-identities", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["unlinked-entities", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["cast-suggestions", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["review-progress", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["show-people"] });
    },
  });
}

// Bulk assign tracks mutation
export function useBulkAssignTracks() {
  const client = useQueryClient();
  return useMutation<
    BulkAssignmentResponse,
    ApiError,
    { episodeId: string; payload: BulkAssignRequest }
  >({
    mutationFn: ({ episodeId, payload }) => bulkAssignTracks(episodeId, payload),
    onSuccess: (_data, variables) => {
      client.invalidateQueries({ queryKey: ["episode-identities", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["unlinked-entities", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["cast-suggestions", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["review-progress", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["show-people"] });
    },
  });
}

// Save identity name mutation
export function useSaveIdentityName() {
  const client = useQueryClient();
  return useMutation<
    AssignmentResponse,
    ApiError,
    { episodeId: string; identityId: string; name: string; show?: string }
  >({
    mutationFn: ({ episodeId, identityId, name, show }) =>
      saveIdentityName(episodeId, identityId, name, show),
    onSuccess: (_data, variables) => {
      client.invalidateQueries({ queryKey: ["episode-identities", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["unlinked-entities", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["cast-suggestions", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["roster-names"] });
    },
  });
}

// Move frames mutation
export function useMoveFrames() {
  const client = useQueryClient();
  return useMutation<
    { moved: number; target_name?: string; target_identity_id?: string },
    ApiError,
    { episodeId: string; trackId: number; payload: MoveFramesRequest }
  >({
    mutationFn: ({ episodeId, trackId, payload }) => moveFrames(episodeId, trackId, payload),
    onSuccess: (_data, variables) => {
      client.invalidateQueries({ queryKey: ["track-frames", variables.episodeId, variables.trackId] });
      client.invalidateQueries({ queryKey: ["track-detail", variables.episodeId, variables.trackId] });
      client.invalidateQueries({ queryKey: ["episode-identities", variables.episodeId] });
    },
  });
}

// Delete frames mutation
export function useDeleteFrames() {
  const client = useQueryClient();
  return useMutation<
    { deleted: number },
    ApiError,
    { episodeId: string; trackId: number; payload: DeleteFramesRequest }
  >({
    mutationFn: ({ episodeId, trackId, payload }) => deleteFrames(episodeId, trackId, payload),
    onSuccess: (_data, variables) => {
      client.invalidateQueries({ queryKey: ["track-frames", variables.episodeId, variables.trackId] });
      client.invalidateQueries({ queryKey: ["track-detail", variables.episodeId, variables.trackId] });
      client.invalidateQueries({ queryKey: ["episode-identities", variables.episodeId] });
    },
  });
}

// Refresh similarity mutation
export function useRefreshSimilarity() {
  const client = useQueryClient();
  return useMutation<RefreshSimilarityResponse, ApiError, string>({
    mutationFn: refreshSimilarity,
    onSuccess: (_data, episodeId) => {
      client.invalidateQueries({ queryKey: ["episode-identities", episodeId] });
      client.invalidateQueries({ queryKey: ["cast-suggestions", episodeId] });
      client.invalidateQueries({ queryKey: ["cluster-metrics"] });
      client.invalidateQueries({ queryKey: ["track-metrics"] });
    },
  });
}

// Auto-link cast mutation
export function useAutoLinkCast() {
  const client = useQueryClient();
  return useMutation<AutoLinkCastResponse, ApiError, string>({
    mutationFn: autoLinkCast,
    onSuccess: (_data, episodeId) => {
      client.invalidateQueries({ queryKey: ["episode-identities", episodeId] });
      client.invalidateQueries({ queryKey: ["unlinked-entities", episodeId] });
      client.invalidateQueries({ queryKey: ["cast-suggestions", episodeId] });
      client.invalidateQueries({ queryKey: ["review-progress", episodeId] });
    },
  });
}

// Run cleanup mutation
export function useRunCleanup() {
  const client = useQueryClient();
  return useMutation<
    CleanupResponse,
    ApiError,
    { episodeId: string; actions: CleanupAction[]; protectedIds?: string[] }
  >({
    mutationFn: ({ episodeId, actions, protectedIds }) =>
      runCleanup(episodeId, { actions, protected_identity_ids: protectedIds }),
    onSuccess: (_data, variables) => {
      client.invalidateQueries({ queryKey: ["episode-identities", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["unlinked-entities", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["cleanup-preview", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["backups", variables.episodeId] });
    },
  });
}

// Create backup mutation
export function useCreateBackup() {
  const client = useQueryClient();
  return useMutation<{ backup_id: string }, ApiError, string>({
    mutationFn: createBackup,
    onSuccess: (_data, episodeId) => {
      client.invalidateQueries({ queryKey: ["backups", episodeId] });
    },
  });
}

// Restore backup mutation
export function useRestoreBackup() {
  const client = useQueryClient();
  return useMutation<
    { files_restored: number },
    ApiError,
    { episodeId: string; backupId: string }
  >({
    mutationFn: ({ episodeId, backupId }) => restoreBackup(episodeId, backupId),
    onSuccess: (_data, variables) => {
      // Invalidate everything
      client.invalidateQueries({ queryKey: ["episode-identities", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["unlinked-entities", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["cast-suggestions", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["cluster-metrics"] });
      client.invalidateQueries({ queryKey: ["track-metrics"] });
      client.invalidateQueries({ queryKey: ["review-progress", variables.episodeId] });
    },
  });
}

// Save assignments mutation
export function useSaveAssignments() {
  return useMutation<{ saved_count: number }, ApiError, string>({
    mutationFn: saveAssignments,
  });
}

// Create cast member mutation
export function useCreateCastMember() {
  const client = useQueryClient();
  return useMutation<{ cast_id: string }, ApiError, { showSlug: string; name: string }>({
    mutationFn: ({ showSlug, name }) => createCastMember(showSlug, name),
    onSuccess: (_data, variables) => {
      client.invalidateQueries({ queryKey: ["show-cast", variables.showSlug] });
      client.invalidateQueries({ queryKey: ["roster-names", variables.showSlug] });
    },
  });
}

// Faces Review view state (stored in URL and localStorage)
const FACES_VIEW_KEY = "screenalytics_faces_view";

export function useFacesReviewState(episodeId: string) {
  const [view, setViewState] = useState<FacesReviewView>("main");
  const [selectedCastId, setSelectedCastId] = useState<string | null>(null);
  const [selectedClusterId, setSelectedClusterId] = useState<string | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);
  const [selectedFrameIds, setSelectedFrameIds] = useState<Set<number>>(new Set());

  // Load state from localStorage on mount
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const stored = localStorage.getItem(`${FACES_VIEW_KEY}:${episodeId}`);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (parsed.view) setViewState(parsed.view);
        if (parsed.castId) setSelectedCastId(parsed.castId);
        if (parsed.clusterId) setSelectedClusterId(parsed.clusterId);
        if (parsed.trackId) setSelectedTrackId(parsed.trackId);
      }
    } catch {
      // Ignore
    }
  }, [episodeId]);

  // Save state to localStorage
  const saveState = useCallback(() => {
    if (typeof window === "undefined") return;
    localStorage.setItem(
      `${FACES_VIEW_KEY}:${episodeId}`,
      JSON.stringify({
        view,
        castId: selectedCastId,
        clusterId: selectedClusterId,
        trackId: selectedTrackId,
      })
    );
  }, [episodeId, view, selectedCastId, selectedClusterId, selectedTrackId]);

  // Navigation helpers
  const goToMain = useCallback(() => {
    setViewState("main");
    setSelectedCastId(null);
    setSelectedClusterId(null);
    setSelectedTrackId(null);
    setSelectedFrameIds(new Set());
  }, []);

  const goToCastMember = useCallback((castId: string) => {
    setViewState("cast_member");
    setSelectedCastId(castId);
    setSelectedClusterId(null);
    setSelectedTrackId(null);
    setSelectedFrameIds(new Set());
  }, []);

  const goToCluster = useCallback((clusterId: string) => {
    setViewState("cluster");
    setSelectedClusterId(clusterId);
    setSelectedTrackId(null);
    setSelectedFrameIds(new Set());
  }, []);

  const goToTrack = useCallback((trackId: number) => {
    setViewState("track");
    setSelectedTrackId(trackId);
    setSelectedFrameIds(new Set());
  }, []);

  const goBack = useCallback(() => {
    if (view === "track") {
      setViewState("cluster");
      setSelectedTrackId(null);
      setSelectedFrameIds(new Set());
    } else if (view === "cluster") {
      if (selectedCastId) {
        setViewState("cast_member");
      } else {
        setViewState("main");
      }
      setSelectedClusterId(null);
    } else if (view === "cast_member") {
      setViewState("main");
      setSelectedCastId(null);
    }
  }, [view, selectedCastId]);

  // Frame selection helpers
  const toggleFrameSelection = useCallback((frameId: number) => {
    setSelectedFrameIds((prev) => {
      const next = new Set(prev);
      if (next.has(frameId)) {
        next.delete(frameId);
      } else {
        next.add(frameId);
      }
      return next;
    });
  }, []);

  const selectAllFrames = useCallback((frameIds: number[]) => {
    setSelectedFrameIds(new Set(frameIds));
  }, []);

  const clearFrameSelection = useCallback(() => {
    setSelectedFrameIds(new Set());
  }, []);

  // Save state when it changes
  useEffect(() => {
    saveState();
  }, [saveState]);

  return {
    view,
    selectedCastId,
    selectedClusterId,
    selectedTrackId,
    selectedFrameIds,
    goToMain,
    goToCastMember,
    goToCluster,
    goToTrack,
    goBack,
    toggleFrameSelection,
    selectAllFrames,
    clearFrameSelection,
  };
}

// Bulk track selection for assignments
export function useBulkTrackSelection() {
  const [selectedTracks, setSelectedTracks] = useState<Set<number>>(new Set());

  const toggleTrack = useCallback((trackId: number) => {
    setSelectedTracks((prev) => {
      const next = new Set(prev);
      if (next.has(trackId)) {
        next.delete(trackId);
      } else {
        next.add(trackId);
      }
      return next;
    });
  }, []);

  const selectAllTracks = useCallback((trackIds: number[]) => {
    setSelectedTracks(new Set(trackIds));
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedTracks(new Set());
  }, []);

  const isSelected = useCallback((trackId: number) => selectedTracks.has(trackId), [selectedTracks]);

  return {
    selectedTracks,
    toggleTrack,
    selectAllTracks,
    clearSelection,
    isSelected,
    count: selectedTracks.size,
  };
}

// Undo stack for tracking actions
export function useUndoStack(episodeId: string) {
  const [stack, setStack] = useState<Array<{ backup_id: string; action: string; timestamp: string }>>([]);
  const restoreBackupMutation = useRestoreBackup();

  const pushAction = useCallback((backupId: string, action: string) => {
    setStack((prev) => [
      { backup_id: backupId, action, timestamp: new Date().toISOString() },
      ...prev.slice(0, 9), // Keep last 10
    ]);
  }, []);

  const undo = useCallback(async () => {
    if (stack.length === 0) return;
    const [latest, ...rest] = stack;
    await restoreBackupMutation.mutateAsync({ episodeId, backupId: latest.backup_id });
    setStack(rest);
  }, [stack, episodeId, restoreBackupMutation]);

  const canUndo = stack.length > 0;

  return { stack, pushAction, undo, canUndo, isUndoing: restoreBackupMutation.isPending };
}

// ============================================================================
// Improve Faces Hooks (Post-Cluster Modal)
// ============================================================================

import {
  fetchImproveFacesSuggestions,
  markInitialPassDone,
  submitFaceReviewDecision,
  type ImproveFacesSuggestionsResponse,
  type FaceReviewDecisionRequest,
} from "./client";

// Query hook for fetching improve faces suggestions
export function useImproveFacesSuggestions(
  episodeId: string,
  options?: { enabled?: boolean }
) {
  return useQuery<ImproveFacesSuggestionsResponse, ApiError>({
    queryKey: ["improve-faces-suggestions", episodeId],
    queryFn: () => fetchImproveFacesSuggestions(episodeId),
    enabled: options?.enabled ?? true,
    staleTime: 60 * 1000, // 1 minute
    retry: false,
  });
}

// Mark initial pass done mutation
export function useMarkInitialPassDone() {
  const client = useQueryClient();
  return useMutation<void, ApiError, string>({
    mutationFn: markInitialPassDone,
    onSuccess: (_data, episodeId) => {
      client.invalidateQueries({ queryKey: ["improve-faces-suggestions", episodeId] });
    },
  });
}

// Submit face review decision mutation (merge or reject)
export function useSubmitFaceReviewDecision() {
  const client = useQueryClient();
  return useMutation<
    { status: "queued" | "success"; job_id?: string },
    ApiError,
    { episodeId: string; payload: FaceReviewDecisionRequest }
  >({
    mutationFn: ({ episodeId, payload }) => submitFaceReviewDecision(episodeId, payload),
    onSuccess: (_data, variables) => {
      client.invalidateQueries({ queryKey: ["episode-identities", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["unlinked-entities", variables.episodeId] });
      client.invalidateQueries({ queryKey: ["episode-status", variables.episodeId] });
    },
  });
}
