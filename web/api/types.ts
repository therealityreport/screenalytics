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

export type EpisodePhase = "detect" | "track" | "faces" | "cluster" | "screentime" | "detect-track" | "audio";

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

// Show types
export type Show = {
  id: string;
  slug: string;
  name?: string;
  network?: string;
  created_at?: string;
};

export type ShowCreateRequest = {
  slug: string;
  name?: string;
  network?: string;
};

// S3 Video types
export type S3VideoItem = {
  ep_id: string;
  key: string;
  size?: number;
  last_modified?: string;
  exists_in_store: boolean;
};

export type S3VideosResponse = {
  items: S3VideoItem[];
};

// Job types
export type JobState = "pending" | "running" | "succeeded" | "failed" | "canceled";

export type JobProgress = {
  frames_done?: number;
  frames_total?: number;
  percent?: number;
  fps_infer?: number;
  device?: string;
  secs_done?: number;
  elapsed_sec?: number;
};

export type Job = {
  job_id: string;
  ep_id: string;
  phase: string;
  state: JobState;
  progress?: JobProgress;
  created_at?: string;
  started_at?: string;
  finished_at?: string;
  error?: string;
  summary?: Record<string, unknown>;
};

export type JobsResponse = {
  jobs: Job[];
};

// Audio pipeline types
export type ASRProvider = "openai_whisper" | "gemini";

export type AudioPipelineRequest = {
  ep_id: string;
  overwrite?: boolean;
  asr_provider?: ASRProvider;
  skip_diarization?: boolean;
};

// Video validation types
export type VideoValidation = {
  codec: string;
  container: string;
  duration: number;
  resolution: { width: number; height: number };
  fps: number;
  fileSize: number;
  isValid: boolean;
  warnings: string[];
  errors: string[];
};

// Episode detail for replace mode
export type EpisodeDetail = {
  ep_id: string;
  show_slug: string;
  season: number;
  episode: number;
  title?: string;
  air_date?: string;
  video_meta?: {
    fps_detected?: number;
    duration?: number;
    resolution?: string;
    codec?: string;
  };
  tracks_count?: number;
  faces_count?: number;
  processed_at?: string;
};

// Episode list types
export type EpisodeSummary = {
  ep_id: string;
  show_slug: string;
  season_number: number;
  episode_number: number;
  title?: string | null;
  air_date?: string | null;
};

export type EpisodeListResponse = {
  episodes: EpisodeSummary[];
};
