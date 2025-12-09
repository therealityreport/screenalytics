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

// Extended episode with status for list view
export type EpisodeWithStatus = EpisodeSummary & {
  status?: EpisodeStatus;
  thumbnail_url?: string | null;
  featured_timestamp?: number | null;
  processing_progress?: number | null;
};

// Episode sort options
export type EpisodeSortOption =
  | "show-season-episode"
  | "newest-first"
  | "oldest-first"
  | "most-tracks"
  | "alphabetical";

// Episode view modes
export type EpisodeViewMode = "card" | "table" | "timeline";

// Bulk operation types
export type BulkOperation = "delete" | "rerun-detect" | "rerun-cluster";

// Featured thumbnail request
export type SetFeaturedThumbnailRequest = {
  ep_id: string;
  timestamp_s: number;
};

// Timestamp preview response
export type TimestampPreviewResponse = {
  url: string;
  timestamp_s: number;
  frame_idx: number;
  faces_count: number;
  faces?: Array<{
    track_id: number;
    identity_id?: string;
    cast_name?: string;
    bbox: [number, number, number, number];
  }>;
};

// Recently accessed episode (for localStorage)
export type RecentEpisode = {
  ep_id: string;
  show_slug: string;
  accessed_at: string;
};

// Favorite episode (for localStorage)
export type FavoriteEpisode = {
  ep_id: string;
  added_at: string;
};

// ============================================================================
// Episode Detail Page Types
// ============================================================================

// Extended video metadata
export type VideoMeta = {
  fps_detected?: number;
  duration_sec?: number;
  resolution?: string;
  width?: number;
  height?: number;
  codec?: string;
  container?: string;
  file_size?: number;
  frames?: number;
};

// Extended episode detail response
export type EpisodeDetailResponse = {
  ep_id: string;
  show_slug: string;
  season_number: number;
  episode_number: number;
  title?: string | null;
  air_date?: string | null;
  created_at?: string;
  video_meta?: VideoMeta;
  s3?: {
    v1_exists?: boolean;
    v2_exists?: boolean;
    key?: string;
    size?: number;
  };
  local?: {
    exists?: boolean;
    path?: string;
  };
};

// Pipeline settings (persisted to localStorage)
export type PipelineSettings = {
  // Detect/Track settings
  device: string;
  detector: string;
  tracker: string;
  stride: number;
  det_thresh: number;
  save_frames: boolean;
  save_crops: boolean;
  max_gap: number;
  // Scene detection
  scene_detector: string;
  scene_threshold: number;
  scene_min_len: number;
  // Faces harvest
  faces_device: string;
  min_frames_between_crops: number;
  thumb_size: number;
  faces_jpeg_quality: number;
  // Cluster settings
  cluster_device: string;
  cluster_thresh: number;
  min_cluster_size: number;
  min_identity_sim: number;
};

// Default pipeline settings
export const DEFAULT_PIPELINE_SETTINGS: PipelineSettings = {
  device: "auto",
  detector: "retinaface",
  tracker: "bytetrack",
  stride: 3,
  det_thresh: 0.5,
  save_frames: true,
  save_crops: true,
  max_gap: 90,
  scene_detector: "pyscenedetect",
  scene_threshold: 27.0,
  scene_min_len: 12,
  faces_device: "auto",
  min_frames_between_crops: 32,
  thumb_size: 256,
  faces_jpeg_quality: 72,
  cluster_device: "auto",
  cluster_thresh: 0.58,
  min_cluster_size: 1,
  min_identity_sim: 0.5,
};

// Phase-specific job trigger request
export type DetectTrackJobRequest = {
  ep_id: string;
  device?: string;
  detector?: string;
  tracker?: string;
  stride?: number;
  det_thresh?: number;
  save_frames?: boolean;
  save_crops?: boolean;
  max_gap?: number;
  scene_detector?: string;
  scene_threshold?: number;
  scene_min_len?: number;
};

export type FacesJobRequest = {
  ep_id: string;
  device?: string;
  min_frames_between_crops?: number;
  thumb_size?: number;
  jpeg_quality?: number;
};

export type ClusterJobRequest = {
  ep_id: string;
  device?: string;
  cluster_thresh?: number;
  min_cluster_size?: number;
  min_identity_sim?: number;
};

// Job history entry with more details
export type JobHistoryEntry = Job & {
  requested?: Record<string, unknown>;
  runtime_sec?: number;
};

// Storage status
export type StorageStatus = {
  s3_enabled: boolean;
  local_path?: string;
  bucket?: string;
};

// Artifact counts
export type ArtifactCounts = {
  frames: number;
  crops: number;
  thumbs_tracks: number;
  manifests: number;
};

// Episode artifact status
export type EpisodeArtifactStatus = {
  sync_status: "synced" | "partial" | "pending" | "empty" | "s3_disabled";
  local: ArtifactCounts;
  s3: ArtifactCounts;
};

// Quick stats for episode
export type EpisodeQuickStats = {
  tracks_count: number;
  identities_count: number;
  assigned_count: number;
  unassigned_count: number;
  singleton_before: number;
  singleton_after: number;
  screen_time_calculated: boolean;
};
