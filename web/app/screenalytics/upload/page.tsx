"use client";

import { Suspense, useEffect, useMemo, useReducer, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useCreateEpisode, useEpisodeStatus, usePresignAssets, useTriggerPhase } from "@/api/hooks";
import { normalizeError } from "@/api/client";
import { uploadFileWithProgress } from "@/api/upload";
import { createInitialState, uploadReducer } from "@/lib/state/uploadMachine";
import { useToast } from "@/components/toast";
import styles from "./upload.module.css";

function formatBytes(bytes?: number) {
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

function UploadPageContent() {
  const params = useSearchParams();
  const toast = useToast();
  const lockedEpisodeId = params.get("ep_id") || undefined;
  const [state, dispatch] = useReducer(uploadReducer, lockedEpisodeId, createInitialState);
  const [file, setFile] = useState<File | undefined>();
  const [aborter, setAborter] = useState<AbortController | null>(null);
  const [showName, setShowName] = useState("");
  const [season, setSeason] = useState("");
  const [episodeName, setEpisodeName] = useState("");

  const createEpisodeMutation = useCreateEpisode();
  const presignMutation = usePresignAssets();
  const triggerPhase = useTriggerPhase();

  const episodeId = useMemo(() => state.episodeId || lockedEpisodeId, [state.episodeId, lockedEpisodeId]);

  const statusQuery = useEpisodeStatus(episodeId, {
    enabled: Boolean(episodeId) && (state.step === "processing" || state.step === "success"),
    refetchInterval: state.step === "processing" ? 1200 : undefined,
  });

  useEffect(() => {
    if (!lockedEpisodeId) return;
    dispatch({ type: "SET_MODE", mode: "replace", episodeId: lockedEpisodeId });
  }, [lockedEpisodeId]);

  useEffect(() => {
    if (!statusQuery.data || state.step !== "processing") return;
    const flags = {
      tracks_only_fallback: statusQuery.data.tracks_only_fallback,
      faces_manifest_fallback: statusQuery.data.faces_manifest_fallback,
    };
    if (statusQuery.data.tracks_ready && statusQuery.data.detect_track?.status) {
      dispatch({ type: "SET_STEP", step: "success", flags, message: "Detect + Track complete" });
      toast.notify({ title: "Detect + track ready", description: `Episode ${statusQuery.data.ep_id}` });
    }
  }, [statusQuery.data, state.step, toast]);

  const handleFileChange = (next?: File | null) => {
    setFile(next || undefined);
    dispatch({
      type: "SET_FILE",
      file: next
        ? {
            name: next.name,
            size: next.size,
            type: next.type,
          }
        : undefined,
    });
  };

  const ensureEpisodeId = async (): Promise<string> => {
    if (episodeId) return episodeId;
    if (!showName || !season || !episodeName) {
      throw normalizeError({ code: "missing_metadata", message: "Provide show, season, and episode" });
    }
    const seasonNumber = Number(season);
    const episodeNumber = Number(episodeName);
    if (Number.isNaN(seasonNumber) || Number.isNaN(episodeNumber)) {
      throw normalizeError({ code: "invalid_numbers", message: "Season and episode must be numbers" });
    }
    const created = await createEpisodeMutation.mutateAsync({
      show_slug_or_id: showName,
      season_number: seasonNumber,
      episode_number: episodeNumber,
      title: null,
      air_date: null,
    });
    dispatch({ type: "SET_MODE", mode: "replace", episodeId: created.ep_id });
    return created.ep_id;
  };

  const triggerDetectTrack = async (epId: string) => {
    const job = await triggerPhase.mutateAsync({ episodeId: epId, phase: "detect-track" });
    dispatch({ type: "SET_STEP", step: "processing", jobId: (job as { job_id?: string }).job_id, message: "Detect/track running" });
  };

  const handleUpload = async () => {
    if (!file) {
      const err = normalizeError({ code: "missing_file", message: "Select a video first" });
      dispatch({ type: "ERROR", error: err });
      toast.notify({ title: "Upload blocked", description: err.message, variant: "error" });
      return;
    }

    try {
      dispatch({ type: "SET_STEP", step: "preparing", message: "Requesting upload slot" });
      const epId = await ensureEpisodeId();
      const presign = await presignMutation.mutateAsync({ epId });
      dispatch({ type: "SET_MODE", mode: "replace", episodeId: epId });

      if (presign.method === "FILE" || !presign.upload_url) {
        // Local-only path: treat as immediate success and kick off detect/track.
        dispatch({ type: "SET_STEP", step: "verifying", message: "Local upload ready" });
        await triggerDetectTrack(epId);
        toast.notify({ title: "Local upload", description: `Video staged at ${presign.path || "local"}` });
        return;
      }

      dispatch({ type: "SET_STEP", step: "uploading", message: "Uploading video" });
      const controller = new AbortController();
      setAborter(controller);

      const uploadResponse = await uploadFileWithProgress(presign, file, {
        signal: controller.signal,
        onProgress: (p) => dispatch({ type: "SET_PROGRESS", progress: Math.round(p.percent * 100), speedBps: p.speedBps }),
      });
      if (!uploadResponse.ok) {
        throw normalizeError({ code: "upload_failed", message: `Upload failed with status ${uploadResponse.status}` });
      }

      dispatch({ type: "SET_STEP", step: "verifying", message: "Verifying upload" });
      await triggerDetectTrack(epId);
      setAborter(null);
    } catch (err) {
      const error = normalizeError(err);
      dispatch({ type: "ERROR", error });
      toast.notify({ title: "Upload failed", description: error.message, variant: "error" });
      setAborter(null);
    }
  };

  const handleReset = () => {
    if (aborter) {
      aborter.abort();
    }
    setAborter(null);
    setFile(undefined);
    dispatch({ type: "RESET" });
  };

  const disabled = state.step === "uploading" || state.step === "processing";

  return (
    <div className={styles.page}>
      <div className="card">
        <h2 className={styles.sectionTitle}>Upload</h2>
        {lockedEpisodeId ? (
          <p className={styles.labelRow}>
            Replace mode locked to episode <strong>{lockedEpisodeId}</strong>
          </p>
        ) : (
          <p className={styles.labelRow}>
            Create a new episode by selecting show/season/episode and dropping a file.
          </p>
        )}

        <div className={styles.grid}>
          <div className={styles.field}>
            <div className={styles.labelRow}>
              <span>Show</span>
            </div>
            <input
              className={styles.input}
              placeholder="e.g. The Reality Report"
              value={showName}
              onChange={(e) => setShowName(e.target.value)}
              disabled={Boolean(lockedEpisodeId) || disabled}
            />
          </div>
          <div className={styles.field}>
            <div className={styles.labelRow}>
              <span>Season</span>
            </div>
            <input
              className={styles.input}
              placeholder="e.g. 3"
              value={season}
              onChange={(e) => setSeason(e.target.value)}
              disabled={Boolean(lockedEpisodeId) || disabled}
            />
          </div>
          <div className={styles.field}>
            <div className={styles.labelRow}>
              <span>Episode</span>
            </div>
            <input
              className={styles.input}
              placeholder="e.g. 5"
              value={episodeName}
              onChange={(e) => setEpisodeName(e.target.value)}
              disabled={Boolean(lockedEpisodeId) || disabled}
            />
          </div>
        </div>

        <div style={{ marginTop: 14 }}>
          <label htmlFor="file" className={styles.dropzone}>
            <strong>Drop or pick a video file</strong>
            <div style={{ fontSize: 13, marginTop: 6 }}>
              {state.file ? `${state.file.name} (${formatBytes(state.file.size)})` : "Supported: mp4, mov, mkv"}
            </div>
            <input
              id="file"
              type="file"
              accept="video/*"
              style={{ display: "none" }}
              onChange={(e) => handleFileChange(e.target.files?.[0])}
              disabled={disabled}
            />
          </label>
          <div className={styles.statsRow}>
            <span className={styles.statChip}>Mode: {lockedEpisodeId ? "Replace" : "New"}</span>
            {state.file && <span className={styles.statChip}>Size: {formatBytes(state.file.size)}</span>}
            {state.progress !== undefined && (
              <span className={styles.statChip}>Progress: {state.progress}%</span>
            )}
          </div>
        </div>

        <div className={styles.actions}>
          <button className={styles.buttonPrimary} onClick={handleUpload} disabled={disabled || !state.file}>
            {state.step === "uploading" ? "Uploading..." : state.step === "processing" ? "Processing..." : "Start upload"}
          </button>
          <button className={styles.buttonSecondary} onClick={handleReset} disabled={state.step === "uploading" && !aborter}>
            Reset
          </button>
        </div>

        {state.progress !== undefined && (
          <div style={{ marginTop: 12 }}>
            <div className={styles.progressShell}>
              <div className={styles.progressBar} style={{ width: `${state.progress}%` }} />
            </div>
          </div>
        )}

        {state.error && <div className={styles.alert}>{state.error.message}</div>}
        {state.step === "success" && (
          <div className={styles.success}>
            Upload verified. Detect/track complete for episode {episodeId}.
          </div>
        )}
      </div>

      <div className="card">
        <h3 className={styles.sectionTitle}>Job status</h3>
        <div className={styles.jobStatus}>
          <div className={styles.tagRow}>
            <span className={styles.tag}>Episode: {episodeId || "pending"}</span>
            {state.jobId && <span className={styles.tag}>Job: {state.jobId}</span>}
            {statusQuery.data?.tracks_only_fallback && <span className={styles.tag}>tracks_only_fallback</span>}
            {statusQuery.data?.faces_manifest_fallback && <span className={styles.tag}>faces_manifest_fallback</span>}
          </div>
          <div style={{ marginTop: 10 }}>
            <div>Detect+Track: {statusQuery.data?.detect_track?.status || "pending"}</div>
            <div>Faces: {statusQuery.data?.faces_embed?.status || "pending"}</div>
            <div>Cluster: {statusQuery.data?.cluster?.status || "pending"}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function UploadPage() {
  return (
    <Suspense fallback={<div className="card">Loading...</div>}>
      <UploadPageContent />
    </Suspense>
  );
}
