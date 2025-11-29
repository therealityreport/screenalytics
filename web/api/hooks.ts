"use client";

import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createEpisode, eventsUrl, fetchEpisodeStatus, mapEventStream, presignEpisodeAssets, triggerJob } from "./client";
import type {
  ApiError,
  AssetUploadResponse,
  EpisodeCreateRequest,
  EpisodeEvent,
  EpisodePhase,
  EpisodeStatus,
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

export type EventConnectionState = "idle" | "connecting" | "connected" | "error";

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
  const queryClient = useQueryClient();
  const [events, setEvents] = useState<EpisodeEvent[]>([]);
  const [lastEvent, setLastEvent] = useState<EpisodeEvent | null>(null);
  const [state, setState] = useState<EventConnectionState>("idle");
  const [manifestMtimes, setManifestMtimes] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!episodeId) return undefined;

    if (process.env.NEXT_PUBLIC_MSW === "1") {
      setState("connected");
      const timer = setInterval(() => {
        const mockEvent: EpisodeEvent = {
          episode_id: episodeId,
          phase: "detect-track",
          event: "progress",
          message: "mock event",
          progress: Math.random(),
        };
        handlersRef.current?.onEvent?.(mockEvent);
        setLastEvent(mockEvent);
        setEvents((prev) => [mockEvent, ...prev].slice(0, 50));
      }, 3000);
      return () => clearInterval(timer);
    }

    const url = eventsUrl(episodeId);
    const source = new EventSource(url);
    setState("connecting");

    const onMessage = (evt: MessageEvent<string>) => {
      const parsed = mapEventStream(evt);
      if (!parsed) return;
      handlersRef.current?.onEvent?.(parsed);
      setLastEvent(parsed);
      setEvents((prev) => [parsed, ...prev].slice(0, 50));
      if (parsed.manifest_mtime && parsed.phase) {
        const manifestKey = parsed.manifest_type || parsed.phase;
        setManifestMtimes((prev) => ({
          ...prev,
          [manifestKey]: parsed.manifest_mtime as string,
        }));
        queryClient.invalidateQueries({
          predicate: (query) => {
            const key = query.queryKey;
            return Array.isArray(key) && key[0] === "episode-status" && key[1] === episodeId;
          },
        });
        queryClient.invalidateQueries({ queryKey: ["episode-manifest", episodeId, manifestKey] });
      }
    };
    const onError = (evt: Event) => {
      setState("error");
      handlersRef.current?.onError?.(evt);
    };
    const onOpen = () => setState("connected");

    source.addEventListener("message", onMessage as EventListener);
    source.addEventListener("error", onError);
    source.addEventListener("open", onOpen);

    return () => {
      source.removeEventListener("message", onMessage as EventListener);
      source.removeEventListener("error", onError);
      source.removeEventListener("open", onOpen);
      source.close();
      setState("idle");
    };
  }, [episodeId, queryClient]);

  return { events, lastEvent, state, manifestMtimes };
}
