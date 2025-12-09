"use client";

import { useEffect, useRef } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  cancelJob,
  createEpisode,
  createShow,
  deleteEpisode,
  eventsUrl,
  fetchAllRunningJobs,
  fetchEpisodeDetail,
  fetchEpisodes,
  fetchEpisodeStatus,
  fetchJobs,
  fetchS3Videos,
  fetchShows,
  mapEventStream,
  mirrorEpisodeFromS3,
  presignEpisodeAssets,
  triggerAudioPipeline,
  triggerJob,
  upsertEpisodeById,
} from "./client";
import type {
  ApiError,
  AssetUploadResponse,
  AudioPipelineRequest,
  EpisodeCreateRequest,
  EpisodeDetail,
  EpisodeEvent,
  EpisodePhase,
  EpisodeStatus,
  EpisodeSummary,
  Job,
  S3VideoItem,
  Show,
  ShowCreateRequest,
} from "./types";

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
