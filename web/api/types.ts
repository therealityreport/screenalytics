import type { components } from "./schema";

export type ApiError = {
  code: string;
  message: string;
  details?: unknown;
};

export type PhaseStatus = components["schemas"]["PhaseStatus"];
export type EpisodeStatus = components["schemas"]["EpisodeStatusResponse"];
export type EpisodeCreateRequest = components["schemas"]["EpisodeCreateRequest"];
export type EpisodeCreateResponse = components["schemas"]["EpisodeCreateResponse"];
export type AssetUploadResponse = components["schemas"]["AssetUploadResponse"];
export type DetectTrackRequest = components["schemas"]["DetectTrackRequest"];

export type EpisodePhase = "detect" | "track" | "faces" | "cluster" | "screentime" | "detect-track";

export type EpisodeEvent = {
  episode_id: string;
  phase: EpisodePhase | "detect_track";
  event: "start" | "finish" | "error" | "progress";
  message?: string;
  progress?: number;
  flags?: Pick<EpisodeStatus, "tracks_only_fallback" | "faces_manifest_fallback">;
  manifest_mtime?: string;
  metrics?: Record<string, number | string>;
};
