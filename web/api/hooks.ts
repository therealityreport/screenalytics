"use client";

import { useEffect, useRef } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createEpisode, eventsUrl, fetchEpisodeStatus, mapEventStream, presignEpisodeAssets, triggerJob } from "./client";
import type { ApiError, AssetUploadResponse, EpisodeCreateRequest, EpisodeEvent, EpisodePhase, EpisodeStatus } from "./types";

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
