"use client";

import { use, useMemo, useState } from "react";
import Link from "next/link";
import { useEpisodeEvents, useEpisodeStatus, useTriggerPhase } from "@/api/hooks";
import type { EpisodePhase } from "@/api/types";
import styles from "../episode.module.css";

function StatusBadge({ state }: { state?: string }) {
  const text = state || "pending";
  const cls = `${styles.statusBadge} ${styles[text as keyof typeof styles] || styles.pending}`;
  return <span className={cls}>{text}</span>;
}

export default function EpisodeDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id: episodeId } = use(params);
  const [events, setEvents] = useState<string[]>([]);

  const statusQuery = useEpisodeStatus(episodeId, { enabled: true, refetchInterval: 1800 });
  const triggerPhase = useTriggerPhase();

  useEpisodeEvents(episodeId, {
    onEvent: (evt) => {
      setEvents((prev) => [`${evt.phase}:${evt.event}${evt.message ? ` - ${evt.message}` : ""}`, ...prev].slice(0, 6));
    },
  });

  const flags = useMemo(
    () => ({
      tracks_only_fallback: statusQuery.data?.tracks_only_fallback,
      faces_manifest_fallback: statusQuery.data?.faces_manifest_fallback,
    }),
    [statusQuery.data?.tracks_only_fallback, statusQuery.data?.faces_manifest_fallback],
  );

  const rerun = (phase: EpisodePhase) => {
    triggerPhase.mutate({ episodeId, phase });
  };

  const detectTrack = statusQuery.data?.detect_track;
  const faces = statusQuery.data?.faces_embed;
  const cluster = statusQuery.data?.cluster;

  return (
    <div className="card">
      <div className={styles.headerRow}>
        <div>
          <Link href="/screenalytics/episodes" style={{ color: "#64748b", fontSize: 13, textDecoration: "none" }}>
            ← Episodes
          </Link>
          <h2 style={{ margin: "4px 0 0" }}>{episodeId}</h2>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            style={{ border: "1px solid #cbd5e1", background: "#fff", padding: "10px 12px", borderRadius: 10, cursor: "pointer" }}
            onClick={() => rerun("detect-track")}
            disabled={triggerPhase.isPending}
          >
            Rerun Detect/Track
          </button>
          <button
            style={{ border: "1px solid #cbd5e1", background: "#fff", padding: "10px 12px", borderRadius: 10, cursor: "pointer" }}
            onClick={() => rerun("cluster")}
            disabled={triggerPhase.isPending}
          >
            Rerun Cluster
          </button>
        </div>
      </div>

      <p style={{ color: "#475569" }}>
        Polls status every ~2s and listens to SSE events when available. Flags mirror Streamlit rules: tracks-only disables harvest,
        faces manifest fallback shows a warning, cluster metrics differentiate pre/post merge.
      </p>

      <div className={styles.statusGrid}>
        <div className={styles.statusCard}>
          <div className={styles.statusTitle}>Detect+Track</div>
          <StatusBadge state={detectTrack?.status} />
          <div style={{ fontSize: 12, color: "#475569", marginTop: 4 }}>
            tracks: {detectTrack?.tracks ?? 0} | detections: {detectTrack?.detections ?? 0}
          </div>
        </div>
        <div className={styles.statusCard}>
          <div className={styles.statusTitle}>Faces</div>
          <StatusBadge state={faces?.status} />
          {flags.faces_manifest_fallback && (
            <div style={{ marginTop: 6, color: "#b45309", fontSize: 12 }}>status stale, using manifest</div>
          )}
        </div>
        <div className={styles.statusCard}>
          <div className={styles.statusTitle}>Cluster</div>
          <StatusBadge state={cluster?.status} />
          {cluster?.singleton_fraction_before !== undefined && cluster?.singleton_fraction_after !== undefined && (
            <div style={{ fontSize: 12, color: "#475569", marginTop: 4 }}>
              singleton: {(cluster.singleton_fraction_before * 100).toFixed(1)}% → {(cluster.singleton_fraction_after * 100).toFixed(1)}%
            </div>
          )}
        </div>
      </div>

      <div style={{ marginTop: 16 }}>
        {flags.tracks_only_fallback && (
          <div style={{ color: "#b91c1c", marginBottom: 8 }}>
            tracks_only_fallback: harvest disabled until detections arrive.
          </div>
        )}
        <div>
          <strong>Events (SSE):</strong>
          <ul className={styles.logList}>
            {events.map((evt, index) => (
              <li key={`${index}-${evt}`}>{evt}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
