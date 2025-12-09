"use client";

import { use, useMemo, useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  useEpisodeStatus,
  useEpisodeEvents,
  useEpisodeDetails,
  useVideoMeta,
  useEpisodeJobHistory,
  useArtifactStatus,
  useTriggerDetectTrack,
  useTriggerFacesEmbed,
  useTriggerCluster,
  useTriggerPhase,
  useDeleteEpisode,
  usePipelineSettings,
  useRecentEpisodes,
  useImproveFacesSuggestions,
  useMarkInitialPassDone,
  useSubmitFaceReviewDecision,
} from "@/api/hooks";
import type { ImproveFacesSuggestion } from "@/api/client";
import type {
  EpisodePhase,
  PipelineSettings,
  Job,
  EpisodeEvent,
} from "@/api/types";
import { DEFAULT_PIPELINE_SETTINGS } from "@/api/types";
import styles from "./episode-detail.module.css";

// Helper to format bytes
function formatBytes(bytes?: number): string {
  if (!bytes) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

// Helper to format duration
function formatDuration(seconds?: number): string {
  if (!seconds) return "—";
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  if (hrs > 0) return `${hrs}:${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

// Helper to format relative time
function formatRelativeTime(dateStr?: string): string {
  if (!dateStr) return "—";
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHrs = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHrs / 24);

  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHrs < 24) return `${diffHrs}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

// Helper to calculate ETA
function calculateETA(progress?: number, elapsed?: number): string {
  if (!progress || !elapsed || progress <= 0 || progress >= 1) return "—";
  const remaining = (elapsed / progress) * (1 - progress);
  if (remaining < 60) return `~${Math.ceil(remaining)}s`;
  if (remaining < 3600) return `~${Math.ceil(remaining / 60)}m`;
  return `~${(remaining / 3600).toFixed(1)}h`;
}

// Status badge component
function StatusBadge({ status }: { status?: string }) {
  const text = status || "pending";
  const className = `${styles.statusBadge} ${
    text === "running" ? styles.statusRunning :
    text === "success" || text === "complete" || text === "done" ? styles.statusSuccess :
    text === "error" || text === "failed" ? styles.statusError :
    text === "missing" ? styles.statusMissing :
    styles.statusPending
  }`;
  return <span className={className}>{text}</span>;
}

// Phase card component
function PhaseCard({
  title,
  icon,
  status,
  metrics,
  progress,
  warning,
  error,
  onRun,
  isRunning,
  disabled,
  extraAction,
}: {
  title: string;
  icon: string;
  status?: string;
  metrics?: { label: string; value: string | number }[];
  progress?: { percent: number; framesDone?: number; framesTotal?: number; eta?: string };
  warning?: string;
  error?: string;
  onRun: () => void;
  isRunning?: boolean;
  disabled?: boolean;
  extraAction?: { label: string; onClick: () => void };
}) {
  return (
    <div className={styles.phaseCard}>
      <div className={styles.phaseCardHeader}>
        <div className={styles.phaseTitle}>
          <span className={styles.phaseIcon}>{icon}</span>
          {title}
        </div>
        <StatusBadge status={status} />
      </div>

      {metrics && metrics.length > 0 && (
        <div className={styles.phaseMetrics}>
          {metrics.map((m) => (
            <div key={m.label} className={styles.metric}>
              <span className={styles.metricLabel}>{m.label}</span>
              <span className={styles.metricValue}>{m.value}</span>
            </div>
          ))}
        </div>
      )}

      {progress && status === "running" && (
        <div className={styles.phaseProgress}>
          <div className={styles.progressBar}>
            <div
              className={styles.progressFill}
              style={{ width: `${Math.min(100, progress.percent * 100)}%` }}
            />
          </div>
          <div className={styles.progressText}>
            <span>
              {progress.framesDone !== undefined && progress.framesTotal !== undefined
                ? `${progress.framesDone.toLocaleString()} / ${progress.framesTotal.toLocaleString()} frames`
                : `${(progress.percent * 100).toFixed(1)}%`}
            </span>
            {progress.eta && <span>ETA: {progress.eta}</span>}
          </div>
        </div>
      )}

      {warning && <div className={styles.phaseWarning}>{warning}</div>}
      {error && <div className={styles.phaseError}>{error}</div>}

      <div className={styles.phaseActions}>
        <button
          className={`${styles.phaseBtn} ${styles.phaseBtnPrimary}`}
          onClick={onRun}
          disabled={disabled || isRunning}
        >
          {isRunning ? "Running..." : status === "success" || status === "done" ? "Rerun" : "Run"}
        </button>
      </div>

      {/* Extra action button (e.g., Improve Faces after cluster) */}
      {extraAction && (
        <button
          className={styles.phaseImproveFacesBtn}
          onClick={extraAction.onClick}
        >
          🎯 {extraAction.label}
        </button>
      )}
    </div>
  );
}

// Settings modal component
function SettingsModal({
  isOpen,
  onClose,
  settings,
  onSave,
  onReset,
}: {
  isOpen: boolean;
  onClose: () => void;
  settings: PipelineSettings;
  onSave: (settings: PipelineSettings) => void;
  onReset: () => void;
}) {
  const [activeTab, setActiveTab] = useState<"detect" | "faces" | "cluster">("detect");
  const [localSettings, setLocalSettings] = useState(settings);

  useEffect(() => {
    setLocalSettings(settings);
  }, [settings]);

  if (!isOpen) return null;

  const handleChange = (key: keyof PipelineSettings, value: string | number | boolean) => {
    setLocalSettings((prev) => ({ ...prev, [key]: value }));
  };

  const handleSave = () => {
    onSave(localSettings);
    onClose();
  };

  const handleReset = () => {
    setLocalSettings(DEFAULT_PIPELINE_SETTINGS);
    onReset();
  };

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h3 className={styles.modalTitle}>Pipeline Settings</h3>
          <button className={styles.modalClose} onClick={onClose}>×</button>
        </div>

        <div className={styles.modalBody}>
          <div className={styles.settingsTabs}>
            <button
              className={`${styles.settingsTab} ${activeTab === "detect" ? styles.settingsTabActive : ""}`}
              onClick={() => setActiveTab("detect")}
            >
              Detect/Track
            </button>
            <button
              className={`${styles.settingsTab} ${activeTab === "faces" ? styles.settingsTabActive : ""}`}
              onClick={() => setActiveTab("faces")}
            >
              Faces
            </button>
            <button
              className={`${styles.settingsTab} ${activeTab === "cluster" ? styles.settingsTabActive : ""}`}
              onClick={() => setActiveTab("cluster")}
            >
              Cluster
            </button>
          </div>

          {activeTab === "detect" && (
            <>
              <div className={styles.settingsSection}>
                <h4 className={styles.settingsSectionTitle}>Detection Settings</h4>
                <div className={styles.settingsGrid}>
                  <div className={styles.settingsField}>
                    <label className={styles.settingsLabel}>Device</label>
                    <select
                      className={styles.settingsSelect}
                      value={localSettings.device}
                      onChange={(e) => handleChange("device", e.target.value)}
                    >
                      <option value="auto">Auto</option>
                      <option value="cuda">CUDA (GPU)</option>
                      <option value="mps">MPS (Apple)</option>
                      <option value="cpu">CPU</option>
                    </select>
                  </div>
                  <div className={styles.settingsField}>
                    <label className={styles.settingsLabel}>Detector</label>
                    <select
                      className={styles.settingsSelect}
                      value={localSettings.detector}
                      onChange={(e) => handleChange("detector", e.target.value)}
                    >
                      <option value="retinaface">RetinaFace</option>
                      <option value="mtcnn">MTCNN</option>
                    </select>
                  </div>
                  <div className={styles.settingsField}>
                    <label className={styles.settingsLabel}>Detection Threshold</label>
                    <input
                      type="number"
                      className={styles.settingsInput}
                      value={localSettings.det_thresh}
                      onChange={(e) => handleChange("det_thresh", parseFloat(e.target.value))}
                      min={0.1}
                      max={1}
                      step={0.05}
                    />
                    <span className={styles.settingsHint}>0.1 - 1.0 (higher = stricter)</span>
                  </div>
                  <div className={styles.settingsField}>
                    <label className={styles.settingsLabel}>Frame Stride</label>
                    <input
                      type="number"
                      className={styles.settingsInput}
                      value={localSettings.stride}
                      onChange={(e) => handleChange("stride", parseInt(e.target.value))}
                      min={1}
                      max={10}
                    />
                    <span className={styles.settingsHint}>Process every Nth frame</span>
                  </div>
                </div>
              </div>

              <div className={styles.settingsSection}>
                <h4 className={styles.settingsSectionTitle}>Tracking Settings</h4>
                <div className={styles.settingsGrid}>
                  <div className={styles.settingsField}>
                    <label className={styles.settingsLabel}>Tracker</label>
                    <select
                      className={styles.settingsSelect}
                      value={localSettings.tracker}
                      onChange={(e) => handleChange("tracker", e.target.value)}
                    >
                      <option value="bytetrack">ByteTrack</option>
                      <option value="sort">SORT</option>
                    </select>
                  </div>
                  <div className={styles.settingsField}>
                    <label className={styles.settingsLabel}>Max Gap (frames)</label>
                    <input
                      type="number"
                      className={styles.settingsInput}
                      value={localSettings.max_gap}
                      onChange={(e) => handleChange("max_gap", parseInt(e.target.value))}
                      min={1}
                      max={300}
                    />
                    <span className={styles.settingsHint}>Max frames to bridge track gaps</span>
                  </div>
                </div>
              </div>

              <div className={styles.settingsSection}>
                <h4 className={styles.settingsSectionTitle}>Scene Detection</h4>
                <div className={styles.settingsGrid}>
                  <div className={styles.settingsField}>
                    <label className={styles.settingsLabel}>Scene Detector</label>
                    <select
                      className={styles.settingsSelect}
                      value={localSettings.scene_detector}
                      onChange={(e) => handleChange("scene_detector", e.target.value)}
                    >
                      <option value="pyscenedetect">PySceneDetect</option>
                      <option value="none">Disabled</option>
                    </select>
                  </div>
                  <div className={styles.settingsField}>
                    <label className={styles.settingsLabel}>Scene Threshold</label>
                    <input
                      type="number"
                      className={styles.settingsInput}
                      value={localSettings.scene_threshold}
                      onChange={(e) => handleChange("scene_threshold", parseFloat(e.target.value))}
                      min={10}
                      max={50}
                      step={1}
                    />
                  </div>
                </div>
              </div>

              <div className={styles.settingsSection}>
                <h4 className={styles.settingsSectionTitle}>Output Options</h4>
                <div className={styles.settingsGrid}>
                  <label className={styles.settingsCheckbox}>
                    <input
                      type="checkbox"
                      checked={localSettings.save_frames}
                      onChange={(e) => handleChange("save_frames", e.target.checked)}
                    />
                    <span className={styles.settingsLabel}>Save annotated frames</span>
                  </label>
                  <label className={styles.settingsCheckbox}>
                    <input
                      type="checkbox"
                      checked={localSettings.save_crops}
                      onChange={(e) => handleChange("save_crops", e.target.checked)}
                    />
                    <span className={styles.settingsLabel}>Save face crops</span>
                  </label>
                </div>
              </div>
            </>
          )}

          {activeTab === "faces" && (
            <div className={styles.settingsSection}>
              <h4 className={styles.settingsSectionTitle}>Face Embedding Settings</h4>
              <div className={styles.settingsGrid}>
                <div className={styles.settingsField}>
                  <label className={styles.settingsLabel}>Device</label>
                  <select
                    className={styles.settingsSelect}
                    value={localSettings.faces_device}
                    onChange={(e) => handleChange("faces_device", e.target.value)}
                  >
                    <option value="auto">Auto</option>
                    <option value="cuda">CUDA (GPU)</option>
                    <option value="mps">MPS (Apple)</option>
                    <option value="cpu">CPU</option>
                  </select>
                </div>
                <div className={styles.settingsField}>
                  <label className={styles.settingsLabel}>Min Frames Between Crops</label>
                  <input
                    type="number"
                    className={styles.settingsInput}
                    value={localSettings.min_frames_between_crops}
                    onChange={(e) => handleChange("min_frames_between_crops", parseInt(e.target.value))}
                    min={1}
                    max={120}
                  />
                  <span className={styles.settingsHint}>Frames between face samples</span>
                </div>
                <div className={styles.settingsField}>
                  <label className={styles.settingsLabel}>Thumbnail Size</label>
                  <input
                    type="number"
                    className={styles.settingsInput}
                    value={localSettings.thumb_size}
                    onChange={(e) => handleChange("thumb_size", parseInt(e.target.value))}
                    min={64}
                    max={512}
                  />
                  <span className={styles.settingsHint}>Face crop dimensions (px)</span>
                </div>
                <div className={styles.settingsField}>
                  <label className={styles.settingsLabel}>JPEG Quality</label>
                  <input
                    type="number"
                    className={styles.settingsInput}
                    value={localSettings.faces_jpeg_quality}
                    onChange={(e) => handleChange("faces_jpeg_quality", parseInt(e.target.value))}
                    min={30}
                    max={100}
                  />
                  <span className={styles.settingsHint}>Higher = better quality, larger files</span>
                </div>
              </div>
            </div>
          )}

          {activeTab === "cluster" && (
            <div className={styles.settingsSection}>
              <h4 className={styles.settingsSectionTitle}>Clustering Settings</h4>
              <div className={styles.settingsGrid}>
                <div className={styles.settingsField}>
                  <label className={styles.settingsLabel}>Device</label>
                  <select
                    className={styles.settingsSelect}
                    value={localSettings.cluster_device}
                    onChange={(e) => handleChange("cluster_device", e.target.value)}
                  >
                    <option value="auto">Auto</option>
                    <option value="cuda">CUDA (GPU)</option>
                    <option value="mps">MPS (Apple)</option>
                    <option value="cpu">CPU</option>
                  </select>
                </div>
                <div className={styles.settingsField}>
                  <label className={styles.settingsLabel}>Cluster Threshold</label>
                  <input
                    type="number"
                    className={styles.settingsInput}
                    value={localSettings.cluster_thresh}
                    onChange={(e) => handleChange("cluster_thresh", parseFloat(e.target.value))}
                    min={0.3}
                    max={0.9}
                    step={0.02}
                  />
                  <span className={styles.settingsHint}>Lower = more clusters (stricter)</span>
                </div>
                <div className={styles.settingsField}>
                  <label className={styles.settingsLabel}>Min Cluster Size</label>
                  <input
                    type="number"
                    className={styles.settingsInput}
                    value={localSettings.min_cluster_size}
                    onChange={(e) => handleChange("min_cluster_size", parseInt(e.target.value))}
                    min={1}
                    max={10}
                  />
                  <span className={styles.settingsHint}>Minimum tracks per cluster</span>
                </div>
                <div className={styles.settingsField}>
                  <label className={styles.settingsLabel}>Min Identity Similarity</label>
                  <input
                    type="number"
                    className={styles.settingsInput}
                    value={localSettings.min_identity_sim}
                    onChange={(e) => handleChange("min_identity_sim", parseFloat(e.target.value))}
                    min={0.3}
                    max={0.9}
                    step={0.05}
                  />
                  <span className={styles.settingsHint}>Min similarity for cast matching</span>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className={styles.modalFooter}>
          <button className={`${styles.modalBtn} ${styles.modalBtnSecondary}`} onClick={handleReset}>
            Reset to Defaults
          </button>
          <button className={`${styles.modalBtn} ${styles.modalBtnSecondary}`} onClick={onClose}>
            Cancel
          </button>
          <button className={`${styles.modalBtn} ${styles.modalBtnPrimary}`} onClick={handleSave}>
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
}

// Delete confirmation modal
function DeleteModal({
  isOpen,
  episodeId,
  onClose,
  onConfirm,
  isDeleting,
}: {
  isOpen: boolean;
  episodeId: string;
  onClose: () => void;
  onConfirm: () => void;
  isDeleting: boolean;
}) {
  if (!isOpen) return null;

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h3 className={styles.modalTitle}>Delete Episode</h3>
          <button className={styles.modalClose} onClick={onClose}>×</button>
        </div>

        <div className={styles.modalBody}>
          <div className={styles.deleteModalContent}>
            <div className={styles.deleteModalIcon}>⚠️</div>
            <p className={styles.deleteModalText}>
              Are you sure you want to delete <strong>{episodeId}</strong>?
            </p>
            <p className={styles.deleteModalWarning}>
              This will permanently delete all episode data, including tracks, faces, and manifests.
              This action cannot be undone.
            </p>
          </div>
        </div>

        <div className={styles.modalFooter}>
          <button
            className={`${styles.modalBtn} ${styles.modalBtnSecondary}`}
            onClick={onClose}
            disabled={isDeleting}
          >
            Cancel
          </button>
          <button
            className={`${styles.modalBtn}`}
            style={{ background: "#dc2626", color: "#fff", border: "none" }}
            onClick={onConfirm}
            disabled={isDeleting}
          >
            {isDeleting ? "Deleting..." : "Delete Episode"}
          </button>
        </div>
      </div>
    </div>
  );
}

// Auto-Run Pipeline phases
type AutoRunPhase = "detect" | "faces" | "cluster";

// Phase weights for progress calculation
const PHASE_WEIGHTS: Record<AutoRunPhase, number> = {
  detect: 0.5,
  faces: 0.3,
  cluster: 0.2,
};

// Auto-Run Pipeline Panel Component
function AutoRunPanel({
  isActive,
  currentPhase,
  completedPhases,
  progress,
  onStart,
  onStop,
  disabled,
  statusMessage,
  isComplete,
  error,
}: {
  isActive: boolean;
  currentPhase: AutoRunPhase | null;
  completedPhases: AutoRunPhase[];
  progress: number;
  onStart: () => void;
  onStop: () => void;
  disabled: boolean;
  statusMessage?: string;
  isComplete?: boolean;
  error?: string;
}) {
  const allPhases: AutoRunPhase[] = ["detect", "faces", "cluster"];
  const phaseLabels: Record<AutoRunPhase, string> = {
    detect: "Detect/Track",
    faces: "Faces Harvest",
    cluster: "Cluster",
  };

  // Calculate overall progress
  let overallProgress = 0;
  for (const phase of completedPhases) {
    overallProgress += PHASE_WEIGHTS[phase] * 100;
  }
  if (currentPhase) {
    overallProgress += PHASE_WEIGHTS[currentPhase] * progress;
  }

  return (
    <div className={styles.autoRunPanel}>
      <div className={styles.autoRunHeader}>
        <div className={styles.autoRunTitle}>
          <span>🚀</span>
          Auto-Run Pipeline
        </div>
      </div>

      {/* Phase Indicators */}
      <div className={styles.autoRunPhases}>
        {allPhases.map((phase, idx) => {
          const isComplete = completedPhases.includes(phase);
          const isRunning = currentPhase === phase;
          const isPending = !isComplete && !isRunning;

          return (
            <span key={phase}>
              <span
                className={`${styles.autoRunPhase} ${
                  isComplete ? styles.autoRunPhaseComplete :
                  isRunning ? styles.autoRunPhaseActive :
                  styles.autoRunPhasePending
                }`}
              >
                {isComplete ? "✓" : isRunning ? "⟳" : "○"} {phaseLabels[phase]}
              </span>
              {idx < allPhases.length - 1 && (
                <span className={styles.autoRunArrow}>→</span>
              )}
            </span>
          );
        })}
      </div>

      {/* Progress Bar */}
      {isActive && (
        <div className={styles.autoRunProgress}>
          <div className={styles.autoRunProgressBar}>
            <div
              className={styles.autoRunProgressFill}
              style={{ width: `${Math.min(100, overallProgress)}%` }}
            />
          </div>
          <div className={styles.autoRunProgressText}>
            <span>{overallProgress.toFixed(0)}% overall</span>
            {currentPhase && <span>Running: {phaseLabels[currentPhase]}</span>}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className={styles.autoRunActions}>
        {!isActive ? (
          <button
            className={`${styles.autoRunBtn} ${styles.autoRunBtnStart}`}
            onClick={onStart}
            disabled={disabled}
          >
            🚀 Start Auto-Run
          </button>
        ) : (
          <button
            className={`${styles.autoRunBtn} ${styles.autoRunBtnStop}`}
            onClick={onStop}
          >
            ⏹️ Stop Auto-Run
          </button>
        )}
      </div>

      {/* Status Messages */}
      {isComplete && (
        <div className={`${styles.autoRunStatus} ${styles.autoRunSuccess}`}>
          🎉 Auto-Run Pipeline Complete! All phases finished successfully.
        </div>
      )}
      {error && (
        <div className={`${styles.autoRunStatus} ${styles.autoRunError}`}>
          ❌ Error: {error}
        </div>
      )}
      {statusMessage && !isComplete && !error && (
        <div className={styles.autoRunStatus}>
          {statusMessage}
        </div>
      )}
    </div>
  );
}

// Improve Faces Modal Component
function ImproveFacesModal({
  isOpen,
  suggestions,
  currentIndex,
  onMerge,
  onReject,
  onSkipAll,
  onGoToFaces,
  onClose,
  isSubmitting,
  episodeId,
}: {
  isOpen: boolean;
  suggestions: ImproveFacesSuggestion[];
  currentIndex: number;
  onMerge: () => void;
  onReject: () => void;
  onSkipAll: () => void;
  onGoToFaces: () => void;
  onClose: () => void;
  isSubmitting: boolean;
  episodeId: string;
}) {
  if (!isOpen) return null;

  const isComplete = currentIndex >= suggestions.length;
  const current = suggestions[currentIndex];

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div
        className={`${styles.modal} ${styles.improveFacesModal}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.modalHeader}>
          <h3 className={styles.modalTitle}>🎯 Improve Face Clustering</h3>
          <button className={styles.modalClose} onClick={onClose}>×</button>
        </div>

        <div className={styles.modalBody}>
          {isComplete ? (
            <div className={styles.improveFacesComplete}>
              <div className={styles.improveFacesCompleteIcon}>✅</div>
              <h3 className={styles.improveFacesCompleteTitle}>All suggestions reviewed!</h3>
              <p className={styles.improveFacesCompleteText}>
                Click <strong>Go to Faces Review</strong> to continue assigning faces to cast members.
              </p>
              <div className={styles.improveFacesCompleteActions}>
                <Link
                  href={`/screenalytics/episodes/${episodeId}/faces`}
                  className={`${styles.modalBtn} ${styles.modalBtnPrimary}`}
                  onClick={onGoToFaces}
                >
                  Go to Faces Review
                </Link>
                <button
                  className={`${styles.modalBtn} ${styles.modalBtnSecondary}`}
                  onClick={onClose}
                >
                  Close
                </button>
              </div>
            </div>
          ) : current ? (
            <>
              {/* Progress */}
              <div className={styles.improveFacesProgress}>
                <div className={styles.improveFacesProgressBar}>
                  <div
                    className={styles.improveFacesProgressFill}
                    style={{ width: `${((currentIndex + 1) / suggestions.length) * 100}%` }}
                  />
                </div>
                <span className={styles.improveFacesProgressText}>
                  {currentIndex + 1} of {suggestions.length}
                </span>
              </div>

              {/* Question */}
              <h3 className={styles.improveFacesQuestion}>
                Are they the same person?
              </h3>

              {/* Compare clusters */}
              <div className={styles.improveFacesCompare}>
                <div className={styles.improveFacesCluster}>
                  {current.cluster_a.crop_url ? (
                    <img
                      src={`/api/proxy/thumb?url=${encodeURIComponent(current.cluster_a.crop_url)}`}
                      alt="Cluster A"
                      className={styles.improveFacesImage}
                    />
                  ) : (
                    <div className={styles.improveFacesNoImage}>No image</div>
                  )}
                  <div className={styles.improveFacesClusterInfo}>
                    <div className={styles.improveFacesClusterId}>
                      {current.cluster_a.id.slice(0, 12)}...
                    </div>
                    <div className={styles.improveFacesClusterStats}>
                      {current.cluster_a.tracks} tracks · {current.cluster_a.faces} faces
                    </div>
                  </div>
                </div>

                <div className={styles.improveFacesCluster}>
                  {current.cluster_b.crop_url ? (
                    <img
                      src={`/api/proxy/thumb?url=${encodeURIComponent(current.cluster_b.crop_url)}`}
                      alt="Cluster B"
                      className={styles.improveFacesImage}
                    />
                  ) : (
                    <div className={styles.improveFacesNoImage}>No image</div>
                  )}
                  <div className={styles.improveFacesClusterInfo}>
                    <div className={styles.improveFacesClusterId}>
                      {current.cluster_b.id.slice(0, 12)}...
                    </div>
                    <div className={styles.improveFacesClusterStats}>
                      {current.cluster_b.tracks} tracks · {current.cluster_b.faces} faces
                    </div>
                  </div>
                </div>
              </div>

              {/* Similarity */}
              <div className={styles.improveFacesSimilarity}>
                <div className={styles.improveFacesSimilarityLabel}>Similarity</div>
                <div className={styles.improveFacesSimilarityValue}>
                  {(current.similarity * 100).toFixed(1)}%
                </div>
              </div>

              {/* Actions */}
              <div className={styles.improveFacesActions}>
                <button
                  className={`${styles.improveFacesBtn} ${styles.improveFacesBtnYes}`}
                  onClick={onMerge}
                  disabled={isSubmitting}
                >
                  Yes, Merge
                </button>
                <button
                  className={`${styles.improveFacesBtn} ${styles.improveFacesBtnNo}`}
                  onClick={onReject}
                  disabled={isSubmitting}
                >
                  No
                </button>
                <button
                  className={`${styles.improveFacesBtn} ${styles.improveFacesBtnSkip}`}
                  onClick={onSkipAll}
                  disabled={isSubmitting}
                >
                  Skip All
                </button>
              </div>
            </>
          ) : (
            <div className={styles.loading}>Loading suggestions...</div>
          )}
        </div>
      </div>
    </div>
  );
}

// Job history item component
function JobHistoryItem({ job }: { job: Job }) {
  const duration = job.started_at && job.finished_at
    ? ((new Date(job.finished_at).getTime() - new Date(job.started_at).getTime()) / 1000)
    : undefined;

  return (
    <div className={styles.jobItem}>
      <span className={styles.jobPhase}>{job.phase}</span>
      <StatusBadge status={job.state} />
      <span className={styles.jobTime}>{formatRelativeTime(job.created_at)}</span>
      {duration && (
        <span className={styles.jobDuration}>{formatDuration(duration)}</span>
      )}
    </div>
  );
}

export default function EpisodeDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id: episodeId } = use(params);
  const router = useRouter();

  // State
  const [events, setEvents] = useState<string[]>([]);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [liveProgress, setLiveProgress] = useState<Record<string, { percent: number; frames?: number; total?: number; elapsed?: number }>>({});

  // Auto-Run Pipeline State
  const [autoRunActive, setAutoRunActive] = useState(false);
  const [autoRunPhase, setAutoRunPhase] = useState<AutoRunPhase | null>(null);
  const [autoRunCompletedPhases, setAutoRunCompletedPhases] = useState<AutoRunPhase[]>([]);
  const [autoRunComplete, setAutoRunComplete] = useState(false);
  const [autoRunError, setAutoRunError] = useState<string | undefined>();
  const [autoRunStatusMessage, setAutoRunStatusMessage] = useState<string | undefined>();

  // Improve Faces Modal State
  const [showImproveFacesModal, setShowImproveFacesModal] = useState(false);
  const [improveFacesSuggestions, setImproveFacesSuggestions] = useState<ImproveFacesSuggestion[]>([]);
  const [improveFacesIndex, setImproveFacesIndex] = useState(0);
  const [checkImproveFaces, setCheckImproveFaces] = useState(false);

  // Hooks
  const { getSettings, saveSettings, resetSettings } = usePipelineSettings();
  const { addRecent } = useRecentEpisodes();
  const [currentSettings, setCurrentSettings] = useState<PipelineSettings>(DEFAULT_PIPELINE_SETTINGS);

  // Load settings on mount
  useEffect(() => {
    setCurrentSettings(getSettings());
  }, [getSettings]);

  // Track recent episode
  useEffect(() => {
    addRecent(episodeId);
  }, [episodeId, addRecent]);

  // Queries
  const statusQuery = useEpisodeStatus(episodeId, { enabled: true, refetchInterval: 2000 });
  const detailsQuery = useEpisodeDetails(episodeId);
  const videoMetaQuery = useVideoMeta(episodeId);
  const jobHistoryQuery = useEpisodeJobHistory(episodeId, { limit: 5 });
  const artifactStatusQuery = useArtifactStatus(episodeId);

  // Mutations
  const triggerDetectTrack = useTriggerDetectTrack();
  const triggerFaces = useTriggerFacesEmbed();
  const triggerCluster = useTriggerCluster();
  const triggerScreentime = useTriggerPhase();
  const deleteEpisode = useDeleteEpisode();

  // Improve Faces mutations
  const markInitialPassDone = useMarkInitialPassDone();
  const submitFaceReviewDecision = useSubmitFaceReviewDecision();

  // Improve Faces suggestions query (only fetch when needed)
  const improveFacesSuggestionsQuery = useImproveFacesSuggestions(episodeId, {
    enabled: checkImproveFaces && !showImproveFacesModal,
  });

  // SSE events
  useEpisodeEvents(episodeId, {
    onEvent: (evt: EpisodeEvent) => {
      // Add to log
      const logEntry = `${evt.phase}:${evt.event}${evt.message ? ` - ${evt.message}` : ""}`;
      setEvents((prev) => [logEntry, ...prev].slice(0, 10));

      // Update progress
      if (evt.progress !== undefined) {
        const phase = evt.phase === "detect_track" ? "detect-track" : evt.phase;
        setLiveProgress((prev) => ({
          ...prev,
          [phase]: {
            percent: evt.progress ?? 0,
            frames: evt.metrics?.frames_done as number | undefined,
            total: evt.metrics?.frames_total as number | undefined,
            elapsed: evt.metrics?.elapsed_sec as number | undefined,
          },
        }));
      }

      // Clear progress on finish
      if (evt.event === "finish" || evt.event === "error") {
        const phase = evt.phase === "detect_track" ? "detect-track" : evt.phase;
        setLiveProgress((prev) => {
          const next = { ...prev };
          delete next[phase];
          return next;
        });
      }
    },
  });

  // Handlers
  const handleRunDetectTrack = useCallback(() => {
    const settings = getSettings();
    triggerDetectTrack.mutate({
      ep_id: episodeId,
      device: settings.device,
      detector: settings.detector,
      tracker: settings.tracker,
      stride: settings.stride,
      det_thresh: settings.det_thresh,
      save_frames: settings.save_frames,
      save_crops: settings.save_crops,
      max_gap: settings.max_gap,
      scene_detector: settings.scene_detector,
      scene_threshold: settings.scene_threshold,
      scene_min_len: settings.scene_min_len,
    });
  }, [episodeId, getSettings, triggerDetectTrack]);

  const handleRunFaces = useCallback(() => {
    const settings = getSettings();
    triggerFaces.mutate({
      ep_id: episodeId,
      device: settings.faces_device,
      min_frames_between_crops: settings.min_frames_between_crops,
      thumb_size: settings.thumb_size,
      jpeg_quality: settings.faces_jpeg_quality,
    });
  }, [episodeId, getSettings, triggerFaces]);

  const handleRunCluster = useCallback(() => {
    const settings = getSettings();
    triggerCluster.mutate({
      ep_id: episodeId,
      device: settings.cluster_device,
      cluster_thresh: settings.cluster_thresh,
      min_cluster_size: settings.min_cluster_size,
      min_identity_sim: settings.min_identity_sim,
    });
  }, [episodeId, getSettings, triggerCluster]);

  const handleRunScreentime = useCallback(() => {
    triggerScreentime.mutate({ episodeId, phase: "screentime" });
  }, [episodeId, triggerScreentime]);

  const handleSaveSettings = useCallback((settings: PipelineSettings) => {
    saveSettings(settings);
    setCurrentSettings(settings);
  }, [saveSettings]);

  const handleResetSettings = useCallback(() => {
    resetSettings();
    setCurrentSettings(DEFAULT_PIPELINE_SETTINGS);
  }, [resetSettings]);

  const handleDelete = useCallback(() => {
    deleteEpisode.mutate(episodeId, {
      onSuccess: () => {
        router.push("/screenalytics/episodes");
      },
    });
  }, [episodeId, deleteEpisode, router]);

  // Auto-Run Pipeline handlers
  const handleStartAutoRun = useCallback(() => {
    setAutoRunActive(true);
    setAutoRunComplete(false);
    setAutoRunError(undefined);
    setAutoRunCompletedPhases([]);
    setAutoRunPhase("detect");
    setAutoRunStatusMessage("Starting Detect/Track phase...");
    // Trigger the first phase
    handleRunDetectTrack();
  }, [handleRunDetectTrack]);

  const handleStopAutoRun = useCallback(() => {
    setAutoRunActive(false);
    setAutoRunPhase(null);
    setAutoRunStatusMessage("Auto-run stopped");
  }, []);

  // Improve Faces handlers
  const handleOpenImproveFaces = useCallback(() => {
    setCheckImproveFaces(true);
  }, []);

  const handleImproveFacesMerge = useCallback(() => {
    const current = improveFacesSuggestions[improveFacesIndex];
    if (!current) return;

    submitFaceReviewDecision.mutate({
      episodeId,
      payload: {
        pair_type: "unassigned_unassigned",
        cluster_a_id: current.cluster_a.id,
        cluster_b_id: current.cluster_b.id,
        decision: "merge",
        execution_mode: "local",
      },
    }, {
      onSuccess: () => {
        setImproveFacesIndex((prev) => prev + 1);
      },
    });
  }, [episodeId, improveFacesSuggestions, improveFacesIndex, submitFaceReviewDecision]);

  const handleImproveFacesReject = useCallback(() => {
    const current = improveFacesSuggestions[improveFacesIndex];
    if (!current) return;

    submitFaceReviewDecision.mutate({
      episodeId,
      payload: {
        pair_type: "unassigned_unassigned",
        cluster_a_id: current.cluster_a.id,
        cluster_b_id: current.cluster_b.id,
        decision: "reject",
        execution_mode: "local",
      },
    }, {
      onSuccess: () => {
        setImproveFacesIndex((prev) => prev + 1);
      },
    });
  }, [episodeId, improveFacesSuggestions, improveFacesIndex, submitFaceReviewDecision]);

  const handleImproveFacesSkipAll = useCallback(() => {
    markInitialPassDone.mutate(episodeId, {
      onSuccess: () => {
        setShowImproveFacesModal(false);
        setImproveFacesSuggestions([]);
        setImproveFacesIndex(0);
        setCheckImproveFaces(false);
      },
    });
  }, [episodeId, markInitialPassDone]);

  const handleImproveFacesClose = useCallback(() => {
    markInitialPassDone.mutate(episodeId, {
      onSettled: () => {
        setShowImproveFacesModal(false);
        setImproveFacesSuggestions([]);
        setImproveFacesIndex(0);
        setCheckImproveFaces(false);
      },
    });
  }, [episodeId, markInitialPassDone]);

  const handleImproveFacesGoToFaces = useCallback(() => {
    markInitialPassDone.mutate(episodeId);
  }, [episodeId, markInitialPassDone]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      switch (e.key) {
        case "Escape":
          setShowSettingsModal(false);
          setShowDeleteModal(false);
          setShowImproveFacesModal(false);
          break;
        case "s":
          if (!e.metaKey && !e.ctrlKey) {
            setShowSettingsModal(true);
          }
          break;
        case "f":
          router.push(`/screenalytics/episodes/${episodeId}/faces`);
          break;
        case "c":
          router.push(`/screenalytics/cast`);
          break;
        case "Backspace":
          router.push("/screenalytics/episodes");
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [episodeId, router]);

  // Effect: Handle improve faces suggestions query result
  useEffect(() => {
    if (improveFacesSuggestionsQuery.data && checkImproveFaces) {
      const { suggestions, initial_pass_done } = improveFacesSuggestionsQuery.data;
      if (suggestions.length > 0 && !initial_pass_done) {
        setImproveFacesSuggestions(suggestions);
        setImproveFacesIndex(0);
        setShowImproveFacesModal(true);
        setCheckImproveFaces(false);
      } else {
        setCheckImproveFaces(false);
      }
    }
  }, [improveFacesSuggestionsQuery.data, checkImproveFaces]);

  // Extract data (moved before effects that depend on it)
  const details = detailsQuery.data;
  const videoMeta = videoMetaQuery.data;
  const status = statusQuery.data;
  const detectTrack = status?.detect_track;
  const faces = status?.faces_embed;
  const cluster = status?.cluster;
  const artifactStatus = artifactStatusQuery.data;
  const jobHistory = jobHistoryQuery.data || [];

  // Effect: Auto-run phase advancement based on status changes
  useEffect(() => {
    if (!autoRunActive || !autoRunPhase) return;

    const detectComplete = detectTrack?.status === "success" || detectTrack?.status === "done";
    const facesComplete = faces?.status === "success" || faces?.status === "done";
    const clusterComplete = cluster?.status === "success" || cluster?.status === "done";

    // Handle phase transitions
    if (autoRunPhase === "detect" && detectComplete && !autoRunCompletedPhases.includes("detect")) {
      // Detect finished, advance to faces
      setAutoRunCompletedPhases((prev) => [...prev, "detect"]);
      setAutoRunPhase("faces");
      setAutoRunStatusMessage("Detect/Track complete - starting Faces Harvest...");
      handleRunFaces();
    } else if (autoRunPhase === "faces" && facesComplete && !autoRunCompletedPhases.includes("faces")) {
      // Faces finished, advance to cluster
      setAutoRunCompletedPhases((prev) => [...prev, "faces"]);
      setAutoRunPhase("cluster");
      setAutoRunStatusMessage("Faces Harvest complete - starting Clustering...");
      handleRunCluster();
    } else if (autoRunPhase === "cluster" && clusterComplete && !autoRunCompletedPhases.includes("cluster")) {
      // Cluster finished, pipeline complete
      setAutoRunCompletedPhases((prev) => [...prev, "cluster"]);
      setAutoRunActive(false);
      setAutoRunPhase(null);
      setAutoRunComplete(true);
      setAutoRunStatusMessage(undefined);
      // Trigger improve faces check after cluster completes
      setCheckImproveFaces(true);
    }

    // Handle errors during auto-run
    if (detectTrack?.status === "error" && autoRunPhase === "detect") {
      setAutoRunError("Detect/Track phase failed");
      setAutoRunActive(false);
      setAutoRunPhase(null);
    } else if (faces?.status === "error" && autoRunPhase === "faces") {
      setAutoRunError("Faces Harvest phase failed");
      setAutoRunActive(false);
      setAutoRunPhase(null);
    } else if (cluster?.status === "error" && autoRunPhase === "cluster") {
      setAutoRunError("Cluster phase failed");
      setAutoRunActive(false);
      setAutoRunPhase(null);
    }
  }, [
    autoRunActive,
    autoRunPhase,
    autoRunCompletedPhases,
    detectTrack?.status,
    faces?.status,
    cluster?.status,
    handleRunFaces,
    handleRunCluster,
  ]);

  // Check if any phase is running
  const isAnyRunning =
    triggerDetectTrack.isPending ||
    triggerFaces.isPending ||
    triggerCluster.isPending ||
    triggerScreentime.isPending ||
    detectTrack?.status === "running" ||
    faces?.status === "running" ||
    cluster?.status === "running";

  // Derive phase dependencies status
  const phases = useMemo(() => [
    {
      id: "detect-track",
      name: "Detect/Track",
      status: detectTrack?.status,
      complete: detectTrack?.status === "success" || detectTrack?.status === "done",
    },
    {
      id: "faces",
      name: "Faces",
      status: faces?.status,
      complete: faces?.status === "success" || faces?.status === "done",
    },
    {
      id: "cluster",
      name: "Cluster",
      status: cluster?.status,
      complete: cluster?.status === "success" || cluster?.status === "done",
    },
  ], [detectTrack?.status, faces?.status, cluster?.status]);

  if (detailsQuery.isLoading) {
    return <div className={styles.loading}>Loading episode details...</div>;
  }

  if (detailsQuery.error) {
    return (
      <div className={styles.error}>
        <p>Failed to load episode: {detailsQuery.error.message}</p>
        <Link href="/screenalytics/episodes">← Back to Episodes</Link>
      </div>
    );
  }

  return (
    <div className={styles.page}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <Link href="/screenalytics/episodes" className={styles.backLink}>
            ← Episodes
          </Link>
          <h1 className={styles.title}>{episodeId}</h1>
          <p className={styles.subtitle}>
            {details?.show_slug && <span>{details.show_slug}</span>}
            {details?.season_number && details?.episode_number && (
              <span>S{details.season_number.toString().padStart(2, "0")}E{details.episode_number.toString().padStart(2, "0")}</span>
            )}
            {details?.title && <span>{details.title}</span>}
          </p>
        </div>
        <div className={styles.headerActions}>
          <button
            className={styles.settingsBtn}
            onClick={() => setShowSettingsModal(true)}
            title="Pipeline Settings (S)"
          >
            ⚙️
          </button>
          <button
            className={styles.deleteBtn}
            onClick={() => setShowDeleteModal(true)}
          >
            Delete Episode
          </button>
        </div>
      </div>

      {/* Quick Actions */}
      <div className={styles.quickActions}>
        <Link
          href={`/screenalytics/episodes/${episodeId}/faces`}
          className={styles.quickActionBtn}
        >
          👤 Faces Review
        </Link>
        <Link
          href={`/screenalytics/cast`}
          className={`${styles.quickActionBtn} ${styles.quickActionBtnSecondary}`}
        >
          🎭 Cast Assignments
        </Link>
      </div>

      {/* Auto-Run Pipeline Panel */}
      <AutoRunPanel
        isActive={autoRunActive}
        currentPhase={autoRunPhase}
        completedPhases={autoRunCompletedPhases}
        progress={
          autoRunPhase === "detect" ? (liveProgress["detect-track"]?.percent ?? 0) * 100 :
          autoRunPhase === "faces" ? (liveProgress["faces"]?.percent ?? 0) * 100 :
          autoRunPhase === "cluster" ? (liveProgress["cluster"]?.percent ?? 0) * 100 :
          0
        }
        onStart={handleStartAutoRun}
        onStop={handleStopAutoRun}
        disabled={isAnyRunning && !autoRunActive}
        statusMessage={autoRunStatusMessage}
        isComplete={autoRunComplete}
        error={autoRunError}
      />

      {/* Video Metadata */}
      {videoMeta && (
        <div className={styles.metadataPanel}>
          <div className={styles.metaItem}>
            <span className={styles.metaLabel}>Duration</span>
            <span className={styles.metaValue}>{formatDuration(videoMeta.duration_sec)}</span>
          </div>
          <div className={styles.metaItem}>
            <span className={styles.metaLabel}>Resolution</span>
            <span className={styles.metaValue}>{videoMeta.resolution || `${videoMeta.width}×${videoMeta.height}` || "—"}</span>
          </div>
          <div className={styles.metaItem}>
            <span className={styles.metaLabel}>FPS</span>
            <span className={`${styles.metaValue} ${styles.metaValueMono}`}>
              {videoMeta.fps_detected?.toFixed(2) || "—"}
            </span>
          </div>
          <div className={styles.metaItem}>
            <span className={styles.metaLabel}>Codec</span>
            <span className={styles.metaValue}>{videoMeta.codec || "—"}</span>
          </div>
          <div className={styles.metaItem}>
            <span className={styles.metaLabel}>File Size</span>
            <span className={styles.metaValue}>{formatBytes(videoMeta.file_size)}</span>
          </div>
          {videoMeta.frames && (
            <div className={styles.metaItem}>
              <span className={styles.metaLabel}>Frames</span>
              <span className={`${styles.metaValue} ${styles.metaValueMono}`}>
                {videoMeta.frames.toLocaleString()}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Phase Dependencies */}
      <div className={styles.dependencies}>
        {phases.map((phase, idx) => (
          <span key={phase.id}>
            <span
              className={`${styles.depPhase} ${
                phase.complete ? styles.depPhaseComplete :
                phase.status === "running" ? styles.depPhaseActive :
                ""
              }`}
            >
              {phase.complete ? "✓" : phase.status === "running" ? "⟳" : "○"} {phase.name}
            </span>
            {idx < phases.length - 1 && <span className={styles.depArrow}>→</span>}
          </span>
        ))}
      </div>

      {/* Phase Status Cards */}
      <div className={styles.phasesGrid}>
        <PhaseCard
          title="Detect + Track"
          icon="🔍"
          status={detectTrack?.status}
          metrics={[
            { label: "Tracks", value: detectTrack?.tracks ?? 0 },
            { label: "Detections", value: detectTrack?.detections ?? 0 },
          ]}
          progress={
            liveProgress["detect-track"]
              ? {
                  percent: liveProgress["detect-track"].percent,
                  framesDone: liveProgress["detect-track"].frames,
                  framesTotal: liveProgress["detect-track"].total,
                  eta: calculateETA(
                    liveProgress["detect-track"].percent,
                    liveProgress["detect-track"].elapsed
                  ),
                }
              : undefined
          }
          warning={status?.tracks_only_fallback ? "Tracks only - no detections available" : undefined}
          onRun={handleRunDetectTrack}
          isRunning={triggerDetectTrack.isPending || detectTrack?.status === "running"}
          disabled={isAnyRunning}
        />

        <PhaseCard
          title="Faces Embed"
          icon="👤"
          status={faces?.status}
          metrics={[
            { label: "Faces", value: faces?.faces ?? 0 },
          ]}
          progress={
            liveProgress["faces"]
              ? {
                  percent: liveProgress["faces"].percent,
                  framesDone: liveProgress["faces"].frames,
                  framesTotal: liveProgress["faces"].total,
                  eta: calculateETA(
                    liveProgress["faces"].percent,
                    liveProgress["faces"].elapsed
                  ),
                }
              : undefined
          }
          warning={status?.faces_manifest_fallback ? "Using manifest fallback - status may be stale" : undefined}
          onRun={handleRunFaces}
          isRunning={triggerFaces.isPending || faces?.status === "running"}
          disabled={isAnyRunning || !detectTrack?.tracks}
        />

        <PhaseCard
          title="Cluster"
          icon="🔗"
          status={cluster?.status}
          metrics={[
            { label: "Identities", value: cluster?.identities ?? 0 },
            {
              label: "Singletons",
              value: cluster?.singleton_fraction_before != null
                ? `${(cluster.singleton_fraction_before * 100).toFixed(0)}% → ${((cluster.singleton_fraction_after ?? 0) * 100).toFixed(0)}%`
                : "—",
            },
          ]}
          progress={
            liveProgress["cluster"]
              ? {
                  percent: liveProgress["cluster"].percent,
                  eta: calculateETA(
                    liveProgress["cluster"].percent,
                    liveProgress["cluster"].elapsed
                  ),
                }
              : undefined
          }
          onRun={handleRunCluster}
          isRunning={triggerCluster.isPending || cluster?.status === "running"}
          disabled={isAnyRunning || !faces?.faces}
          extraAction={
            (cluster?.status === "success" || cluster?.status === "done") && (cluster?.identities ?? 0) > 0
              ? { label: "Improve Faces", onClick: handleOpenImproveFaces }
              : undefined
          }
        />

        {/* Screen Time phase - commented out as it's not in current API schema
        <PhaseCard
          title="Screen Time"
          icon="⏱️"
          status={(status as any)?.screentime?.status}
          metrics={
            (status as any)?.screentime?.total_screen_time !== undefined
              ? [
                  {
                    label: "Total",
                    value: formatDuration((status as any).screentime.total_screen_time),
                  },
                ]
              : []
          }
          onRun={handleRunScreentime}
          isRunning={triggerScreentime.isPending || (status as any)?.screentime?.status === "running"}
          disabled={isAnyRunning || !cluster?.identities}
        />
        */}
      </div>

      {/* Quick Stats */}
      <div className={styles.statsGrid}>
        <div className={styles.statCard}>
          <div className={styles.statValue}>{detectTrack?.tracks ?? 0}</div>
          <div className={styles.statLabel}>Tracks</div>
        </div>
        <div className={styles.statCard}>
          <div className={styles.statValue}>{faces?.faces ?? 0}</div>
          <div className={styles.statLabel}>Faces</div>
        </div>
        <div className={styles.statCard}>
          <div className={styles.statValue}>{cluster?.identities ?? 0}</div>
          <div className={styles.statLabel}>Identities</div>
        </div>
        <div className={styles.statCard}>
          <div className={styles.statValue}>{detectTrack?.detections ?? 0}</div>
          <div className={styles.statLabel}>Detections</div>
        </div>
      </div>

      {/* Job History */}
      <div className={styles.jobHistory}>
        <div className={styles.jobHistoryHeader}>
          <h3 className={styles.jobHistoryTitle}>Recent Jobs</h3>
        </div>
        <div className={styles.jobList}>
          {jobHistory.length > 0 ? (
            jobHistory.map((job) => <JobHistoryItem key={job.job_id} job={job} />)
          ) : (
            <p style={{ color: "#64748b", fontSize: 14 }}>No recent jobs</p>
          )}
        </div>
      </div>

      {/* Storage Status */}
      {artifactStatus && (
        <div className={styles.storagePanel}>
          <div className={styles.storagePanelHeader}>
            <h3 className={styles.storagePanelTitle}>Storage</h3>
            <span
              className={`${styles.syncBadge} ${
                artifactStatus.sync_status === "synced" ? styles.syncSynced :
                artifactStatus.sync_status === "partial" ? styles.syncPartial :
                styles.syncPending
              }`}
            >
              {artifactStatus.sync_status}
            </span>
          </div>
          <div className={styles.storageGrid}>
            <div className={styles.storageColumn}>
              <div className={styles.storageColumnTitle}>Local</div>
              <div className={styles.storageItem}>
                <span className={styles.storageItemLabel}>Frames</span>
                <span className={styles.storageItemValue}>{artifactStatus.local.frames}</span>
              </div>
              <div className={styles.storageItem}>
                <span className={styles.storageItemLabel}>Crops</span>
                <span className={styles.storageItemValue}>{artifactStatus.local.crops}</span>
              </div>
              <div className={styles.storageItem}>
                <span className={styles.storageItemLabel}>Thumbnails</span>
                <span className={styles.storageItemValue}>{artifactStatus.local.thumbs_tracks}</span>
              </div>
            </div>
            <div className={styles.storageColumn}>
              <div className={styles.storageColumnTitle}>S3</div>
              <div className={styles.storageItem}>
                <span className={styles.storageItemLabel}>Frames</span>
                <span className={styles.storageItemValue}>{artifactStatus.s3.frames}</span>
              </div>
              <div className={styles.storageItem}>
                <span className={styles.storageItemLabel}>Crops</span>
                <span className={styles.storageItemValue}>{artifactStatus.s3.crops}</span>
              </div>
              <div className={styles.storageItem}>
                <span className={styles.storageItemLabel}>Thumbnails</span>
                <span className={styles.storageItemValue}>{artifactStatus.s3.thumbs_tracks}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Events Log */}
      {events.length > 0 && (
        <div className={styles.eventsLog}>
          <div className={styles.eventsLogTitle}>Live Events (SSE)</div>
          <ul className={styles.eventsList}>
            {events.map((evt, idx) => (
              <li key={`${idx}-${evt}`}>{evt}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Keyboard Hints */}
      <div className={styles.keyboardHint}>
        <kbd>S</kbd> Settings • <kbd>F</kbd> Faces • <kbd>C</kbd> Cast • <kbd>⌫</kbd> Back
      </div>

      {/* Modals */}
      <SettingsModal
        isOpen={showSettingsModal}
        onClose={() => setShowSettingsModal(false)}
        settings={currentSettings}
        onSave={handleSaveSettings}
        onReset={handleResetSettings}
      />

      <DeleteModal
        isOpen={showDeleteModal}
        episodeId={episodeId}
        onClose={() => setShowDeleteModal(false)}
        onConfirm={handleDelete}
        isDeleting={deleteEpisode.isPending}
      />

      <ImproveFacesModal
        isOpen={showImproveFacesModal}
        suggestions={improveFacesSuggestions}
        currentIndex={improveFacesIndex}
        onMerge={handleImproveFacesMerge}
        onReject={handleImproveFacesReject}
        onSkipAll={handleImproveFacesSkipAll}
        onGoToFaces={handleImproveFacesGoToFaces}
        onClose={handleImproveFacesClose}
        isSubmitting={submitFaceReviewDecision.isPending}
        episodeId={episodeId}
      />
    </div>
  );
}
