"use client";

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  useCreateEpisode,
  useCreateShow,
  useEpisodeDetail,
  useEpisodeStatus,
  useMirrorFromS3,
  usePresignAssets,
  useRunningJobs,
  useS3Videos,
  useShows,
  useTriggerAudioPipeline,
  useTriggerPhase,
  useUpsertEpisode,
  useCancelJob,
} from "@/api/hooks";
import { normalizeError } from "@/api/client";
import { deleteEpisode } from "@/api/client";
import { uploadFileWithProgress } from "@/api/upload";
import { createInitialState, uploadReducer } from "@/lib/state/uploadMachine";
import { useToast } from "@/components/toast";
import type { ASRProvider, S3VideoItem } from "@/api/types";
import styles from "./upload.module.css";

// Constants
const MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024; // 5GB
const MIN_FILE_SIZE = 1 * 1024 * 1024; // 1MB
const SUPPORTED_CODECS = ["h264", "h.264", "avc", "h265", "h.265", "hevc", "vp9", "av1"];
const DRAFT_STORAGE_KEY = "screenalytics-upload-draft";
const NAV_COUNTDOWN_SECONDS = 3;

// Utilities
function formatBytes(bytes?: number): string {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(1)} ${units[unit]}`;
}

function formatSpeed(bytesPerSecond?: number): string {
  if (!bytesPerSecond) return "--";
  if (bytesPerSecond < 1024 * 1024) {
    return `${(bytesPerSecond / 1024).toFixed(1)} KB/s`;
  }
  return `${(bytesPerSecond / (1024 * 1024)).toFixed(1)} MB/s`;
}

function formatETA(remainingBytes: number, speedBps?: number): string {
  if (!speedBps || speedBps <= 0) return "--";
  const seconds = remainingBytes / speedBps;
  if (seconds < 60) return `${Math.ceil(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const secs = Math.ceil(seconds % 60);
  return `${minutes}:${secs.toString().padStart(2, "0")}`;
}

function formatDate(dateStr?: string): string {
  if (!dateStr) return "Unknown";
  try {
    return new Date(dateStr).toLocaleDateString();
  } catch {
    return dateStr.slice(0, 10);
  }
}

function parseEpIdFromKey(key: string): { show: string; season: number; episode: number } | null {
  const match = key.match(/raw\/videos\/([^/]+)\/s(\d{2})\/e(\d{2})/i);
  if (!match) return null;
  return {
    show: match[1],
    season: parseInt(match[2], 10),
    episode: parseInt(match[3], 10),
  };
}

// Draft state for auto-save
interface DraftState {
  showSlug: string;
  season: string;
  episode: string;
  title: string;
  runAudio: boolean;
  asrProvider: ASRProvider;
  savedAt: number;
}

function loadDraft(): DraftState | null {
  if (typeof window === "undefined") return null;
  try {
    const stored = localStorage.getItem(DRAFT_STORAGE_KEY);
    if (!stored) return null;
    const draft = JSON.parse(stored) as DraftState;
    // Only restore drafts less than 1 hour old
    if (Date.now() - draft.savedAt > 60 * 60 * 1000) {
      localStorage.removeItem(DRAFT_STORAGE_KEY);
      return null;
    }
    return draft;
  } catch {
    return null;
  }
}

function saveDraft(draft: Omit<DraftState, "savedAt">): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify({ ...draft, savedAt: Date.now() }));
  } catch {
    // Ignore storage errors
  }
}

function clearDraft(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(DRAFT_STORAGE_KEY);
}

// Mode type
type UploadMode = "select" | "upload" | "browse";

export default function UploadPage() {
  const router = useRouter();
  const params = useSearchParams();
  const toast = useToast();

  // Replace mode from URL
  const lockedEpisodeId = params.get("ep_id") || undefined;

  // View mode state
  const [mode, setMode] = useState<UploadMode>(lockedEpisodeId ? "upload" : "select");

  // Upload state machine
  const [state, dispatch] = useReducer(uploadReducer, lockedEpisodeId, createInitialState);

  // Form state
  const [file, setFile] = useState<File | undefined>();
  const [showSlug, setShowSlug] = useState("");
  const [season, setSeason] = useState("");
  const [episode, setEpisode] = useState("");
  const [title, setTitle] = useState("");
  const [runAudio, setRunAudio] = useState(false);
  const [asrProvider, setAsrProvider] = useState<ASRProvider>("openai_whisper");

  // S3 browser state
  const [s3Search, setS3Search] = useState("");
  const [selectedS3Item, setSelectedS3Item] = useState<S3VideoItem | null>(null);

  // Modals
  const [showCreateShowModal, setShowCreateShowModal] = useState(false);
  const [newShowSlug, setNewShowSlug] = useState("");
  const [newShowName, setNewShowName] = useState("");
  const [showReplaceConfirm, setShowReplaceConfirm] = useState(false);

  // Upload control
  const [aborter, setAborter] = useState<AbortController | null>(null);
  const [dragOver, setDragOver] = useState(false);

  // Navigation countdown
  const [navCountdown, setNavCountdown] = useState<number | null>(null);
  const navTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // File input ref
  const fileInputRef = useRef<HTMLInputElement>(null);

  // API hooks
  const createEpisodeMutation = useCreateEpisode();
  const presignMutation = usePresignAssets();
  const triggerPhase = useTriggerPhase();
  const triggerAudioMutation = useTriggerAudioPipeline();
  const showsQuery = useShows();
  const createShowMutation = useCreateShow();
  const s3VideosQuery = useS3Videos({ enabled: mode === "browse" });
  const runningJobsQuery = useRunningJobs({ refetchInterval: 2000 });
  const mirrorMutation = useMirrorFromS3();
  const upsertMutation = useUpsertEpisode();
  const cancelJobMutation = useCancelJob();

  // Replace mode episode detail
  const replaceDetailQuery = useEpisodeDetail(lockedEpisodeId, { enabled: Boolean(lockedEpisodeId) });

  // Episode status polling
  const episodeId = useMemo(() => state.episodeId || lockedEpisodeId, [state.episodeId, lockedEpisodeId]);
  const statusQuery = useEpisodeStatus(episodeId, {
    enabled: Boolean(episodeId) && (state.step === "processing" || state.step === "success"),
    refetchInterval: state.step === "processing" ? 1500 : false,
  });

  // Load draft on mount
  useEffect(() => {
    if (lockedEpisodeId) return; // Don't load draft in replace mode
    const draft = loadDraft();
    if (draft) {
      setShowSlug(draft.showSlug);
      setSeason(draft.season);
      setEpisode(draft.episode);
      setTitle(draft.title);
      setRunAudio(draft.runAudio);
      setAsrProvider(draft.asrProvider);
      toast.notify({ title: "Draft restored", description: "Continuing from your previous session" });
    }
  }, [lockedEpisodeId, toast]);

  // Auto-save draft
  useEffect(() => {
    if (lockedEpisodeId) return;
    if (!showSlug && !season && !episode && !title) return;
    const timeout = setTimeout(() => {
      saveDraft({ showSlug, season, episode, title, runAudio, asrProvider });
    }, 1000);
    return () => clearTimeout(timeout);
  }, [showSlug, season, episode, title, runAudio, asrProvider, lockedEpisodeId]);

  // Handle replace mode initialization
  useEffect(() => {
    if (!lockedEpisodeId) return;
    dispatch({ type: "SET_MODE", mode: "replace", episodeId: lockedEpisodeId });
    setMode("upload");
  }, [lockedEpisodeId]);

  // Handle upload completion
  useEffect(() => {
    if (!statusQuery.data || state.step !== "processing") return;
    const flags = {
      tracks_only_fallback: statusQuery.data.tracks_only_fallback,
      faces_manifest_fallback: statusQuery.data.faces_manifest_fallback,
    };
    if (statusQuery.data.tracks_ready && statusQuery.data.detect_track?.status) {
      dispatch({ type: "SET_STEP", step: "success", flags, message: "Detect + Track complete" });
      toast.notify({ title: "Processing complete", description: `Episode ${statusQuery.data.ep_id} is ready` });
      clearDraft();
      // Start navigation countdown
      setNavCountdown(NAV_COUNTDOWN_SECONDS);
    }
  }, [statusQuery.data, state.step, toast]);

  // Navigation countdown timer
  useEffect(() => {
    if (navCountdown === null) return;
    if (navCountdown <= 0) {
      router.push(`/screenalytics/episodes/${episodeId}`);
      return;
    }
    navTimeoutRef.current = setTimeout(() => {
      setNavCountdown(navCountdown - 1);
    }, 1000);
    return () => {
      if (navTimeoutRef.current) clearTimeout(navTimeoutRef.current);
    };
  }, [navCountdown, episodeId, router]);

  // Cancel navigation countdown
  const cancelNavigation = useCallback(() => {
    setNavCountdown(null);
    if (navTimeoutRef.current) clearTimeout(navTimeoutRef.current);
  }, []);

  // File validation
  const validateFile = useCallback((f: File): { valid: boolean; warnings: string[]; errors: string[] } => {
    const warnings: string[] = [];
    const errors: string[] = [];

    if (f.size === 0) {
      errors.push("File is empty");
    } else if (f.size < MIN_FILE_SIZE) {
      warnings.push(`File is small (${formatBytes(f.size)})`);
    } else if (f.size > MAX_FILE_SIZE) {
      errors.push(`File exceeds 5GB limit (${formatBytes(f.size)})`);
    }

    const ext = f.name.split(".").pop()?.toLowerCase();
    if (!["mp4", "mov", "mkv", "avi", "webm"].includes(ext || "")) {
      warnings.push(`Unusual extension: .${ext}`);
    }

    return { valid: errors.length === 0, warnings, errors };
  }, []);

  // Handle file selection
  const handleFileChange = useCallback((f?: File | null) => {
    if (!f) {
      setFile(undefined);
      dispatch({ type: "SET_FILE", file: undefined });
      return;
    }

    const validation = validateFile(f);
    if (!validation.valid) {
      toast.notify({ title: "Invalid file", description: validation.errors.join(", "), variant: "error" });
      return;
    }
    if (validation.warnings.length > 0) {
      toast.notify({ title: "Warning", description: validation.warnings.join(", "), variant: "error" });
    }

    setFile(f);
    dispatch({
      type: "SET_FILE",
      file: { name: f.name, size: f.size, type: f.type },
    });
  }, [validateFile, toast]);

  // Drag and drop handlers
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFileChange(f);
  }, [handleFileChange]);

  // Ensure episode ID exists
  const ensureEpisodeId = async (): Promise<string> => {
    if (episodeId) return episodeId;
    if (!showSlug || !season || !episode) {
      throw normalizeError({ code: "missing_metadata", message: "Provide show, season, and episode" });
    }
    const seasonNum = Number(season);
    const episodeNum = Number(episode);
    if (Number.isNaN(seasonNum) || Number.isNaN(episodeNum)) {
      throw normalizeError({ code: "invalid_numbers", message: "Season and episode must be numbers" });
    }
    const created = await createEpisodeMutation.mutateAsync({
      show_slug_or_id: showSlug.toUpperCase(),
      season_number: seasonNum,
      episode_number: episodeNum,
      title: title || null,
      air_date: null,
    });
    dispatch({ type: "SET_MODE", mode: "replace", episodeId: created.ep_id });
    return created.ep_id;
  };

  // Trigger detect/track
  const triggerDetectTrack = async (epId: string) => {
    const job = await triggerPhase.mutateAsync({ episodeId: epId, phase: "detect-track" });
    dispatch({
      type: "SET_STEP",
      step: "processing",
      jobId: (job as { job_id?: string }).job_id,
      message: "Detect/track running",
    });
  };

  // Handle upload
  const handleUpload = async () => {
    if (!file) {
      toast.notify({ title: "No file selected", description: "Select a video file first", variant: "error" });
      return;
    }

    // Confirm replace mode
    if (lockedEpisodeId && !showReplaceConfirm) {
      setShowReplaceConfirm(true);
      return;
    }
    setShowReplaceConfirm(false);

    try {
      dispatch({ type: "SET_STEP", step: "preparing", message: "Creating episode..." });
      const epId = await ensureEpisodeId();

      dispatch({ type: "SET_STEP", step: "preparing", message: "Getting upload URL..." });
      const presign = await presignMutation.mutateAsync({ epId });
      dispatch({ type: "SET_MODE", mode: "replace", episodeId: epId });

      // Local-only mode
      if (presign.method === "FILE" || !presign.upload_url) {
        dispatch({ type: "SET_STEP", step: "verifying", message: "Local upload ready" });
        await triggerDetectTrack(epId);
        if (runAudio) {
          await triggerAudioMutation.mutateAsync({ ep_id: epId, asr_provider: asrProvider });
        }
        toast.notify({ title: "Upload complete", description: `Video staged locally` });
        return;
      }

      // S3 upload
      dispatch({ type: "SET_STEP", step: "uploading", message: "Uploading video..." });
      const controller = new AbortController();
      setAborter(controller);

      const uploadResponse = await uploadFileWithProgress(presign, file, {
        signal: controller.signal,
        onProgress: (p) => dispatch({
          type: "SET_PROGRESS",
          progress: Math.round(p.percent * 100),
          speedBps: p.speedBps,
        }),
      });

      if (!uploadResponse.ok) {
        // Rollback episode on failure
        try {
          await deleteEpisode(epId);
        } catch {
          // Ignore rollback errors
        }
        throw normalizeError({ code: "upload_failed", message: `Upload failed: ${uploadResponse.status}` });
      }

      dispatch({ type: "SET_STEP", step: "verifying", message: "Starting processing..." });
      await triggerDetectTrack(epId);

      if (runAudio) {
        await triggerAudioMutation.mutateAsync({ ep_id: epId, asr_provider: asrProvider });
        toast.notify({ title: "Audio pipeline started", description: `ASR: ${asrProvider}` });
      }

      setAborter(null);
    } catch (err) {
      const error = normalizeError(err);
      dispatch({ type: "ERROR", error });
      toast.notify({ title: "Upload failed", description: error.message, variant: "error" });
      setAborter(null);
    }
  };

  // Handle reset
  const handleReset = useCallback(() => {
    if (aborter) aborter.abort();
    setAborter(null);
    setFile(undefined);
    dispatch({ type: "RESET" });
    setNavCountdown(null);
  }, [aborter]);

  // Cancel replace mode
  const cancelReplaceMode = useCallback(() => {
    router.push("/screenalytics/upload");
  }, [router]);

  // Create show handler
  const handleCreateShow = async () => {
    if (!newShowSlug) {
      toast.notify({ title: "Error", description: "Show slug is required", variant: "error" });
      return;
    }
    try {
      await createShowMutation.mutateAsync({
        slug: newShowSlug.toUpperCase(),
        name: newShowName || undefined,
      });
      setShowSlug(newShowSlug.toUpperCase());
      setShowCreateShowModal(false);
      setNewShowSlug("");
      setNewShowName("");
      toast.notify({ title: "Show created", description: `${newShowSlug.toUpperCase()} added` });
    } catch (err) {
      toast.notify({ title: "Failed to create show", description: normalizeError(err).message, variant: "error" });
    }
  };

  // S3 browser handlers
  const filteredS3Items = useMemo(() => {
    if (!s3VideosQuery.data) return [];
    const search = s3Search.toLowerCase();
    return s3VideosQuery.data
      .filter((item) => {
        if (!search) return true;
        return (
          item.ep_id.toLowerCase().includes(search) ||
          item.key.toLowerCase().includes(search)
        );
      })
      .sort((a, b) => {
        // Sort by show, then season desc, then episode desc
        const parseA = parseEpIdFromKey(a.key);
        const parseB = parseEpIdFromKey(b.key);
        if (!parseA || !parseB) return 0;
        if (parseA.show !== parseB.show) return parseA.show.localeCompare(parseB.show);
        if (parseA.season !== parseB.season) return parseB.season - parseA.season;
        return parseB.episode - parseA.episode;
      });
  }, [s3VideosQuery.data, s3Search]);

  const handleS3ItemClick = useCallback((item: S3VideoItem) => {
    setSelectedS3Item(item);
  }, []);

  const handleMirrorFromS3 = async () => {
    if (!selectedS3Item) return;
    try {
      const result = await mirrorMutation.mutateAsync(selectedS3Item.ep_id);
      toast.notify({
        title: "Mirror complete",
        description: `${formatBytes(result.bytes)} downloaded`,
      });
    } catch (err) {
      toast.notify({ title: "Mirror failed", description: normalizeError(err).message, variant: "error" });
    }
  };

  const handleCreateFromS3 = async () => {
    if (!selectedS3Item) return;
    const parsed = parseEpIdFromKey(selectedS3Item.key);
    if (!parsed) {
      toast.notify({ title: "Error", description: "Could not parse S3 key", variant: "error" });
      return;
    }
    try {
      const result = await upsertMutation.mutateAsync({
        ep_id: selectedS3Item.ep_id,
        show_slug: parsed.show,
        season: parsed.season,
        episode: parsed.episode,
      });
      toast.notify({
        title: result.created ? "Episode created" : "Episode updated",
        description: result.ep_id,
      });
    } catch (err) {
      toast.notify({ title: "Failed", description: normalizeError(err).message, variant: "error" });
    }
  };

  const handleProcessS3Item = async () => {
    if (!selectedS3Item) return;
    try {
      await triggerPhase.mutateAsync({ episodeId: selectedS3Item.ep_id, phase: "detect-track" });
      toast.notify({ title: "Processing started", description: selectedS3Item.ep_id });
    } catch (err) {
      toast.notify({ title: "Failed", description: normalizeError(err).message, variant: "error" });
    }
  };

  const disabled = state.step === "uploading" || state.step === "processing";
  const runningJobs = runningJobsQuery.data || [];

  return (
    <div className={styles.page}>
      {/* Replace Mode Banner */}
      {lockedEpisodeId && (
        <div className={styles.replaceBanner}>
          <div className={styles.replaceBannerTitle}>Replace Mode: {lockedEpisodeId}</div>
          <div className={styles.replaceBannerInfo}>
            Uploading will replace the existing video and require full reprocessing.
          </div>
          {replaceDetailQuery.data && (
            <div className={styles.replaceBannerMeta}>
              {replaceDetailQuery.data.video_meta?.fps_detected && (
                <span className={styles.replaceBannerMetaItem}>
                  FPS: {replaceDetailQuery.data.video_meta.fps_detected.toFixed(2)}
                </span>
              )}
              {replaceDetailQuery.data.tracks_count !== undefined && (
                <span className={styles.replaceBannerMetaItem}>
                  Tracks: {replaceDetailQuery.data.tracks_count.toLocaleString()}
                </span>
              )}
              {replaceDetailQuery.data.faces_count !== undefined && (
                <span className={styles.replaceBannerMetaItem}>
                  Faces: {replaceDetailQuery.data.faces_count.toLocaleString()}
                </span>
              )}
            </div>
          )}
          <button className={styles.buttonDanger} onClick={cancelReplaceMode}>
            Cancel Replace Mode
          </button>
        </div>
      )}

      {/* Replace Confirmation Modal */}
      {showReplaceConfirm && (
        <div className={styles.modalOverlay} onClick={() => setShowReplaceConfirm(false)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalTitle}>Confirm Replace</div>
            <p>This will:</p>
            <ul style={{ margin: "12px 0", paddingLeft: 20 }}>
              <li>Overwrite the existing video file</li>
              <li>Delete all existing detections, tracks, and faces</li>
              <li>Require full reprocessing</li>
            </ul>
            <div className={styles.modalActions}>
              <button className={styles.buttonSecondary} onClick={() => setShowReplaceConfirm(false)}>
                Cancel
              </button>
              <button className={styles.buttonPrimary} onClick={handleUpload}>
                I understand, proceed
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Mode Selector */}
      {!lockedEpisodeId && mode === "select" && (
        <div className="card">
          <h2 className={styles.sectionTitle}>Upload & Process</h2>
          <div className={styles.modeSelector}>
            <div
              className={styles.modeCard}
              onClick={() => setMode("upload")}
            >
              <div className={styles.modeIcon}>+</div>
              <div className={styles.modeTitle}>Upload New</div>
              <div className={styles.modeDescription}>
                Create a new episode from local video
              </div>
            </div>
            <div
              className={styles.modeCard}
              onClick={() => setMode("browse")}
            >
              <div className={styles.modeIcon}>=</div>
              <div className={styles.modeTitle}>Select Existing</div>
              <div className={styles.modeDescription}>
                Browse S3 videos, mirror, or reprocess
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Upload Form */}
      {(mode === "upload" || lockedEpisodeId) && (
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h2 className={styles.sectionTitle}>
              {lockedEpisodeId ? "Replace Video" : "Upload New Episode"}
            </h2>
            {!lockedEpisodeId && (
              <button
                className={`${styles.buttonSecondary} ${styles.buttonSmall}`}
                onClick={() => setMode("select")}
              >
                Back
              </button>
            )}
          </div>

          {/* Episode metadata */}
          {!lockedEpisodeId && (
            <div className={styles.grid}>
              <div className={styles.field}>
                <div className={styles.labelRow}>
                  <span>Show</span>
                  <button
                    type="button"
                    style={{ fontSize: 11, color: "#0ea5e9", background: "none", border: "none", cursor: "pointer" }}
                    onClick={() => setShowCreateShowModal(true)}
                  >
                    + New
                  </button>
                </div>
                <select
                  className={styles.select}
                  value={showSlug}
                  onChange={(e) => setShowSlug(e.target.value)}
                  disabled={disabled}
                >
                  <option value="">Select show...</option>
                  {showsQuery.data?.map((show) => (
                    <option key={show.id} value={show.slug}>
                      {show.name || show.slug}
                    </option>
                  ))}
                </select>
              </div>
              <div className={styles.field}>
                <div className={styles.labelRow}>Season</div>
                <input
                  type="number"
                  className={styles.input}
                  placeholder="e.g. 5"
                  value={season}
                  onChange={(e) => setSeason(e.target.value)}
                  disabled={disabled}
                  min={1}
                  max={99}
                />
              </div>
              <div className={styles.field}>
                <div className={styles.labelRow}>Episode</div>
                <input
                  type="number"
                  className={styles.input}
                  placeholder="e.g. 12"
                  value={episode}
                  onChange={(e) => setEpisode(e.target.value)}
                  disabled={disabled}
                  min={1}
                  max={99}
                />
              </div>
              <div className={`${styles.field} ${styles.gridFull}`}>
                <div className={styles.labelRow}>Title (optional)</div>
                <input
                  type="text"
                  className={styles.input}
                  placeholder="e.g. The Reunion Part 1"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  disabled={disabled}
                />
              </div>
            </div>
          )}

          {/* File drop zone */}
          <div style={{ marginTop: 16 }}>
            <label
              htmlFor="file-input"
              className={`${styles.dropzone} ${dragOver ? styles.dropzoneDragOver : ""} ${disabled ? styles.dropzoneDisabled : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <strong>Drop or click to select video</strong>
              <div style={{ fontSize: 13, marginTop: 6, color: "#64748b" }}>
                Supported: MP4, MOV, MKV (max 5GB) - H.264, H.265, VP9
              </div>
              <input
                ref={fileInputRef}
                id="file-input"
                type="file"
                accept="video/*"
                style={{ display: "none" }}
                onChange={(e) => handleFileChange(e.target.files?.[0])}
                disabled={disabled}
              />
            </label>

            {/* File preview */}
            {file && (
              <div className={styles.filePreview}>
                <div className={styles.fileIcon}>*</div>
                <div className={styles.fileInfo}>
                  <div className={styles.fileName}>{file.name}</div>
                  <div className={styles.fileMeta}>{formatBytes(file.size)}</div>
                </div>
                <button
                  className={styles.fileRemove}
                  onClick={() => handleFileChange(null)}
                  disabled={disabled}
                >
                  Remove
                </button>
              </div>
            )}
          </div>

          {/* Audio pipeline options */}
          <div className={styles.audioOptions}>
            <div className={styles.audioOptionsTitle}>Audio Pipeline</div>
            <label className={styles.checkbox}>
              <input
                type="checkbox"
                checked={runAudio}
                onChange={(e) => setRunAudio(e.target.checked)}
                disabled={disabled}
              />
              Run audio pipeline after upload
            </label>
            {runAudio && (
              <div className={styles.audioProviderRow}>
                <span style={{ fontSize: 13, color: "#64748b" }}>ASR Provider:</span>
                <select
                  className={styles.select}
                  value={asrProvider}
                  onChange={(e) => setAsrProvider(e.target.value as ASRProvider)}
                  disabled={disabled}
                  style={{ flex: 1 }}
                >
                  <option value="openai_whisper">OpenAI Whisper (faster, more accurate)</option>
                  <option value="gemini">Gemini (cheaper for long videos)</option>
                </select>
              </div>
            )}
          </div>

          {/* Progress */}
          {state.progress !== undefined && state.step === "uploading" && (
            <div className={styles.progressContainer}>
              <div className={styles.progressShell}>
                <div className={styles.progressBar} style={{ width: `${state.progress}%` }} />
              </div>
              <div className={styles.progressInfo}>
                <span>
                  {formatBytes((state.file?.size || 0) * (state.progress / 100))} / {formatBytes(state.file?.size)}
                </span>
                <span>{formatSpeed(state.speedBps)}</span>
                <span>
                  ETA: {formatETA((state.file?.size || 0) * (1 - state.progress / 100), state.speedBps)}
                </span>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className={styles.actions}>
            <button
              className={styles.buttonPrimary}
              onClick={handleUpload}
              disabled={disabled || !file}
            >
              {state.step === "uploading"
                ? `Uploading... ${state.progress || 0}%`
                : state.step === "processing"
                  ? "Processing..."
                  : state.step === "preparing"
                    ? "Preparing..."
                    : "Upload & Process"}
            </button>
            <button
              className={styles.buttonSecondary}
              onClick={handleReset}
              disabled={state.step === "idle"}
            >
              Reset
            </button>
          </div>

          {/* Status messages */}
          {state.error && <div className={styles.alert}>{state.error.message}</div>}
          {state.step === "success" && (
            <>
              <div className={styles.success}>
                Upload complete! Episode {episodeId} is being processed.
              </div>
              {navCountdown !== null && (
                <div className={styles.navigationCountdown}>
                  <span className={styles.navigationCountdownText}>
                    Opening Episode Detail in {navCountdown}...
                  </span>
                  <button className={styles.navigationCountdownCancel} onClick={cancelNavigation}>
                    Stay here
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* S3 Browser */}
      {mode === "browse" && (
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h2 className={styles.sectionTitle}>Browse S3 Videos</h2>
            <button
              className={`${styles.buttonSecondary} ${styles.buttonSmall}`}
              onClick={() => setMode("select")}
            >
              Back
            </button>
          </div>

          <div className={styles.s3Browser}>
            <div className={styles.s3SearchRow}>
              <input
                type="text"
                className={styles.s3SearchInput}
                placeholder="Search by episode ID or S3 key..."
                value={s3Search}
                onChange={(e) => setS3Search(e.target.value)}
              />
            </div>

            {s3VideosQuery.isLoading ? (
              <div className={styles.emptyState}>
                <div className={styles.spinner} />
                <div className={styles.emptyStateText}>Loading S3 videos...</div>
              </div>
            ) : filteredS3Items.length === 0 ? (
              <div className={styles.emptyState}>
                <div className={styles.emptyStateIcon}>?</div>
                <div className={styles.emptyStateText}>
                  {s3Search ? "No videos match your search" : "No videos found in S3"}
                </div>
              </div>
            ) : (
              <div className={styles.s3List}>
                {filteredS3Items.map((item) => (
                  <div
                    key={item.key}
                    className={`${styles.s3Item} ${selectedS3Item?.key === item.key ? styles.s3ItemSelected : ""}`}
                    onClick={() => handleS3ItemClick(item)}
                  >
                    <div className={styles.s3ItemInfo}>
                      <div className={styles.s3ItemId}>{item.ep_id}</div>
                      <div className={styles.s3ItemMeta}>
                        {formatBytes(item.size)} | {formatDate(item.last_modified)}
                      </div>
                    </div>
                    <span
                      className={`${styles.s3ItemBadge} ${item.exists_in_store ? styles.s3ItemBadgeTracked : styles.s3ItemBadgeUntracked}`}
                    >
                      {item.exists_in_store ? "Tracked" : "Not tracked"}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {/* S3 Actions */}
            {selectedS3Item && (
              <div className={styles.s3Actions}>
                {selectedS3Item.exists_in_store ? (
                  <>
                    <button
                      className={styles.buttonSecondary}
                      onClick={() => router.push(`/screenalytics/episodes/${selectedS3Item.ep_id}`)}
                    >
                      Open Detail
                    </button>
                    <button
                      className={styles.buttonSecondary}
                      onClick={handleMirrorFromS3}
                      disabled={mirrorMutation.isPending}
                    >
                      {mirrorMutation.isPending ? "Mirroring..." : "Mirror from S3"}
                    </button>
                    <button
                      className={styles.buttonPrimary}
                      onClick={handleProcessS3Item}
                      disabled={triggerPhase.isPending}
                    >
                      {triggerPhase.isPending ? "Starting..." : "Process (Detect/Track)"}
                    </button>
                  </>
                ) : (
                  <button
                    className={styles.buttonPrimary}
                    onClick={handleCreateFromS3}
                    disabled={upsertMutation.isPending}
                  >
                    {upsertMutation.isPending ? "Creating..." : "Create Episode in Store"}
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Job Status Panel */}
      {(state.step === "processing" || runningJobs.length > 0) && (
        <div className="card">
          <h3 className={styles.sectionTitle}>Active Jobs ({runningJobs.length})</h3>
          <div className={styles.jobStatus}>
            {state.jobId && (
              <div style={{ marginBottom: 12 }}>
                <div className={styles.tagRow}>
                  <span className={styles.tag}>Episode: {episodeId}</span>
                  <span className={styles.tag}>Job: {state.jobId}</span>
                  {statusQuery.data?.tracks_only_fallback && (
                    <span className={`${styles.tag} ${styles.tagWarning}`}>tracks_only_fallback</span>
                  )}
                </div>
                <div style={{ marginTop: 10, fontSize: 13, color: "#64748b" }}>
                  <div>Detect+Track: {statusQuery.data?.detect_track?.status || "pending"}</div>
                  <div>Faces: {statusQuery.data?.faces_embed?.status || "pending"}</div>
                  <div>Cluster: {statusQuery.data?.cluster?.status || "pending"}</div>
                </div>
              </div>
            )}

            {runningJobs.filter(j => j.job_id !== state.jobId).map((job) => (
              <div key={job.job_id} className={styles.jobCard}>
                <div className={styles.jobCardHeader}>
                  <span className={styles.jobCardEpId}>{job.ep_id}</span>
                  <span className={styles.jobCardPhase}>{job.phase}</span>
                </div>
                <div className={styles.jobCardProgress}>
                  <div
                    className={styles.jobCardProgressFill}
                    style={{ width: `${(job.progress?.percent || 0) * 100}%` }}
                  />
                </div>
                <div className={styles.jobCardInfo}>
                  <span>{Math.round((job.progress?.percent || 0) * 100)}%</span>
                  <button
                    style={{ fontSize: 11, color: "#b91c1c", background: "none", border: "none", cursor: "pointer" }}
                    onClick={() => cancelJobMutation.mutate(job.job_id)}
                    disabled={cancelJobMutation.isPending}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Create Show Modal */}
      {showCreateShowModal && (
        <div className={styles.modalOverlay} onClick={() => setShowCreateShowModal(false)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalTitle}>Create New Show</div>
            <div className={styles.field} style={{ marginBottom: 12 }}>
              <div className={styles.labelRow}>Show Slug (required)</div>
              <input
                type="text"
                className={styles.input}
                placeholder="e.g. RHOBH"
                value={newShowSlug}
                onChange={(e) => setNewShowSlug(e.target.value.toUpperCase())}
                style={{ textTransform: "uppercase" }}
              />
            </div>
            <div className={styles.field}>
              <div className={styles.labelRow}>Display Name (optional)</div>
              <input
                type="text"
                className={styles.input}
                placeholder="e.g. Real Housewives of Beverly Hills"
                value={newShowName}
                onChange={(e) => setNewShowName(e.target.value)}
              />
            </div>
            <div className={styles.modalActions}>
              <button className={styles.buttonSecondary} onClick={() => setShowCreateShowModal(false)}>
                Cancel
              </button>
              <button
                className={styles.buttonPrimary}
                onClick={handleCreateShow}
                disabled={!newShowSlug || createShowMutation.isPending}
              >
                {createShowMutation.isPending ? "Creating..." : "Create Show"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
