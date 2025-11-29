// Types derived from the generated OpenAPI schema. Keep schema.ts in sync via `npm run api:gen`.
import type { components } from "./schema";

export type ApiErrorEnvelope = {
  code: string;
  message: string;
  details?: unknown;
};

export type ApiError = ApiErrorEnvelope;

export type PhaseStatus = components["schemas"]["PhaseStatus"];
export type EpisodeStatus = components["schemas"]["EpisodeStatusResponse"];
export type EpisodeCreateRequest = components["schemas"]["EpisodeCreateRequest"];
export type EpisodeCreateResponse = components["schemas"]["EpisodeCreateResponse"];
export type AssetUploadResponse = components["schemas"]["AssetUploadResponse"];
export type DetectTrackRequest = components["schemas"]["DetectTrackRequest"];
export type S3Show = components["schemas"]["S3Show"];
export type S3EpisodeForShow = components["schemas"]["S3EpisodeForShow"];
export type S3ShowsResponse = components["schemas"]["S3ShowsResponse"];
export type S3EpisodesForShowResponse = components["schemas"]["S3EpisodesForShowResponse"];

export type EpisodePhase = "detect" | "track" | "faces" | "cluster" | "screentime" | "detect-track";

export type EpisodeEvent = {
  episode_id: string;
  phase: EpisodePhase | "detect_track";
  event: "start" | "finish" | "error" | "progress" | "manifest_updated";
  message?: string;
  progress?: number;
  flags?: Pick<EpisodeStatus, "tracks_only_fallback" | "faces_manifest_fallback">;
  manifest_mtime?: string;
  manifest_type?: string;
  metrics?: Record<string, number | string>;
};
