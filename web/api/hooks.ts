"use client";

import { useEffect, useRef, useCallback } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  bulkDeleteEpisodes,
  cancelJob,
  createEpisode,
  createShow,
  deleteEpisode,
  eventsUrl,
  fetchAllRunningJobs,
  fetchArtifactStatus,
  fetchEpisodeDetail,
  fetchEpisodeDetails,
  fetchEpisodeJobHistory,
  fetchEpisodes,
  fetchEpisodeStatus,
  fetchJobs,
  fetchS3Videos,
  fetchShows,
  fetchStorageConfig,
  fetchTimestampPreview,
  fetchVideoMeta,
  mapEventStream,
  mirrorEpisodeFromS3,
  presignEpisodeAssets,
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
  AssetUploadResponse,
  AudioPipelineRequest,
  ClusterJobRequest,
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
  Job,
  PipelineSettings,
  S3VideoItem,
  Show,
  ShowCreateRequest,
  StorageStatus,
  TimestampPreviewResponse,
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
