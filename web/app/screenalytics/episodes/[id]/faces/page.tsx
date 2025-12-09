"use client";

import { use, useMemo, useState, useCallback, useEffect } from "react";
import Link from "next/link";
import {
  useEpisodeDetails,
  useEpisodeIdentities,
  useUnlinkedEntities,
  useShowPeople,
  useClusterTrackReps,
  useTrackFrames,
  useCastSuggestions,
  useReviewProgress,
  useRosterNames,
  useFacesReviewState,
  useBulkTrackSelection,
  useAssignTrack,
  useBulkAssignTracks,
  useRefreshSimilarity,
  useAutoLinkCast,
  useCreateBackup,
  useUndoStack,
  useSaveAssignments,
  useMoveFrames,
  useDeleteFrames,
} from "@/api/hooks";
import type {
  Identity,
  Track,
  Frame,
  CastSuggestion,
  FacesReviewView,
} from "@/api/types";
import styles from "./faces-review.module.css";

// Helper to format percentage
function formatPercent(value?: number): string {
  if (value === undefined || value === null) return "—";
  return `${Math.round(value * 100)}%`;
}

// Helper to get badge class based on score
function getBadgeStrength(score?: number, threshold = 0.7): string {
  if (score === undefined) return "";
  if (score >= threshold) return styles.badgeStrong;
  if (score < threshold - 0.15) return styles.badgeWeak;
  return "";
}

// Similarity badge component
function SimilarityBadge({
  type,
  value,
  label,
}: {
  type: "identity" | "cast" | "track" | "cluster" | "temporal" | "ambiguity" | "isolation" | "quality";
  value?: number;
  label?: string;
}) {
  if (value === undefined) return null;

  const typeClasses: Record<string, string> = {
    identity: styles.badgeIdentity,
    cast: styles.badgeCast,
    track: styles.badgeTrack,
    cluster: styles.badgeCluster,
    temporal: styles.badgeTemporal,
    ambiguity: styles.badgeAmbiguity,
    isolation: styles.badgeIsolation,
    quality: styles.badgeQuality,
  };

  const labels: Record<string, string> = {
    identity: "ID",
    cast: "CAST",
    track: "TRK",
    cluster: "CLU",
    temporal: "TEMP",
    ambiguity: "AMB",
    isolation: "ISO",
    quality: "Q",
  };

  return (
    <span className={`${styles.badge} ${typeClasses[type]} ${getBadgeStrength(value)}`}>
      {label || labels[type]}: {formatPercent(value)}
    </span>
  );
}

// Cast member card component
function CastMemberCard({
  person,
  onClick,
  isSelected,
}: {
  person: {
    person_id: string;
    name: string;
    thumbnail_url?: string;
    cluster_ids: string[];
    track_count: number;
    face_count: number;
    cohesion?: number;
  };
  onClick: () => void;
  isSelected?: boolean;
}) {
  return (
    <div
      className={`${styles.card} ${isSelected ? styles.cardSelected : ""}`}
      onClick={onClick}
    >
      <div className={styles.cardThumbnail}>
        {person.thumbnail_url ? (
          <img src={person.thumbnail_url} alt={person.name} />
        ) : (
          <div className={styles.cardPlaceholder}>👤</div>
        )}
      </div>
      <div className={styles.cardName}>{person.name}</div>
      <div className={styles.cardMeta}>
        <span>{person.cluster_ids.length} cluster{person.cluster_ids.length !== 1 ? "s" : ""}</span>
        <span>{person.track_count} track{person.track_count !== 1 ? "s" : ""}</span>
      </div>
      {person.cohesion !== undefined && (
        <div className={styles.badges}>
          <SimilarityBadge type="identity" value={person.cohesion} />
        </div>
      )}
    </div>
  );
}

// Cluster card component
function ClusterCard({
  identity,
  onClick,
  isSelected,
  onCheckboxClick,
  showCheckbox,
  suggestions,
}: {
  identity: Identity;
  onClick: () => void;
  isSelected?: boolean;
  onCheckboxClick?: () => void;
  showCheckbox?: boolean;
  suggestions?: CastSuggestion[];
}) {
  const topSuggestion = suggestions?.[0];

  return (
    <div
      className={`${styles.card} ${isSelected ? styles.cardSelected : ""}`}
      onClick={onClick}
    >
      <div className={styles.cardThumbnail}>
        {identity.thumbnail_url ? (
          <img src={identity.thumbnail_url} alt={identity.name || identity.identity_id} />
        ) : (
          <div className={styles.cardPlaceholder}>👤</div>
        )}
        {showCheckbox && (
          <div
            className={`${styles.cardCheckbox} ${isSelected ? styles.cardCheckboxSelected : ""}`}
            onClick={(e) => {
              e.stopPropagation();
              onCheckboxClick?.();
            }}
          >
            {isSelected && "✓"}
          </div>
        )}
      </div>
      <div className={styles.cardName}>
        {identity.name || identity.cast_name || `Cluster ${identity.identity_id.slice(-6)}`}
      </div>
      <div className={styles.cardMeta}>
        <span>{identity.track_count} track{identity.track_count !== 1 ? "s" : ""}</span>
        <span>{identity.face_count} face{identity.face_count !== 1 ? "s" : ""}</span>
      </div>
      <div className={styles.badges}>
        {identity.cohesion !== undefined && (
          <SimilarityBadge type="cluster" value={identity.cohesion} />
        )}
        {identity.cast_similarity !== undefined && (
          <SimilarityBadge type="cast" value={identity.cast_similarity} />
        )}
        {identity.ambiguity !== undefined && identity.ambiguity < 0.15 && (
          <SimilarityBadge type="ambiguity" value={identity.ambiguity} />
        )}
      </div>
      {topSuggestion && !identity.is_assigned && (
        <div className={styles.cardMeta} style={{ marginTop: 4, color: "#7c3aed" }}>
          → {topSuggestion.cast_name} ({formatPercent(topSuggestion.similarity)})
        </div>
      )}
    </div>
  );
}

// Track card component
function TrackCard({
  track,
  onClick,
  isSelected,
  onCheckboxClick,
  showCheckbox,
}: {
  track: Track;
  onClick: () => void;
  isSelected?: boolean;
  onCheckboxClick?: () => void;
  showCheckbox?: boolean;
}) {
  return (
    <div
      className={`${styles.card} ${styles.trackCard} ${isSelected ? styles.cardSelected : ""}`}
      onClick={onClick}
    >
      <div className={styles.cardThumbnail}>
        {track.thumbnail_url || track.crop_url ? (
          <img src={track.thumbnail_url || track.crop_url} alt={`Track ${track.track_id}`} />
        ) : (
          <div className={styles.cardPlaceholder}>🖼️</div>
        )}
        {showCheckbox && (
          <div
            className={`${styles.cardCheckbox} ${isSelected ? styles.cardCheckboxSelected : ""}`}
            onClick={(e) => {
              e.stopPropagation();
              onCheckboxClick?.();
            }}
          >
            {isSelected && "✓"}
          </div>
        )}
        <div className={styles.trackFrameCount}>{track.frame_count} frames</div>
      </div>
      <div className={styles.cardName}>Track {track.track_id}</div>
      <div className={styles.cardMeta}>
        <span>{track.face_count} face{track.face_count !== 1 ? "s" : ""}</span>
        {track.start_frame !== undefined && track.end_frame !== undefined && (
          <span>F{track.start_frame}-{track.end_frame}</span>
        )}
      </div>
      <div className={styles.badges}>
        {track.similarity !== undefined && (
          <SimilarityBadge type="track" value={track.similarity} />
        )}
        {track.quality?.score !== undefined && (
          <SimilarityBadge type="quality" value={track.quality.score} />
        )}
        {track.excluded_frames !== undefined && track.excluded_frames > 0 && (
          <span className={styles.badge} style={{ background: "#fef3c7", color: "#b45309" }}>
            {track.excluded_frames} excl
          </span>
        )}
      </div>
    </div>
  );
}

// Frame card component
function FrameCard({
  frame,
  isSelected,
  onClick,
}: {
  frame: Frame;
  isSelected?: boolean;
  onClick: () => void;
}) {
  return (
    <div
      className={`${styles.card} ${styles.frameCard} ${isSelected ? styles.cardSelected : ""} ${frame.is_outlier ? styles.frameOutlier : ""} ${frame.is_skipped ? styles.frameSkipped : ""}`}
      onClick={onClick}
    >
      <div className={styles.cardThumbnail}>
        <img src={frame.crop_url || frame.thumbnail_url} alt={`Frame ${frame.frame_idx}`} />
        <div className={styles.frameIdx}>#{frame.frame_idx}</div>
        {frame.is_outlier && (
          <div className={styles.frameOutlierBadge}>
            OUTLIER {formatPercent(frame.similarity)}
          </div>
        )}
        <div
          className={`${styles.cardCheckbox} ${isSelected ? styles.cardCheckboxSelected : ""}`}
        >
          {isSelected && "✓"}
        </div>
      </div>
      <div className={styles.badges}>
        {frame.quality?.score !== undefined && (
          <SimilarityBadge type="quality" value={frame.quality.score} />
        )}
        {frame.similarity !== undefined && (
          <SimilarityBadge type="track" value={frame.similarity} label="SIM" />
        )}
      </div>
    </div>
  );
}

// Quick assign modal component
function QuickAssignModal({
  isOpen,
  onClose,
  suggestions,
  rosterNames,
  onAssign,
  isAssigning,
}: {
  isOpen: boolean;
  onClose: () => void;
  suggestions: CastSuggestion[];
  rosterNames: string[];
  onAssign: (name: string, castId?: string) => void;
  isAssigning: boolean;
}) {
  const [customName, setCustomName] = useState("");
  const [selectedRoster, setSelectedRoster] = useState("");

  if (!isOpen) return null;

  const handleSubmit = () => {
    const name = selectedRoster || customName.trim();
    if (!name) return;
    onAssign(name);
  };

  return (
    <>
      <div className={styles.modalOverlay} onClick={onClose} />
      <div className={styles.quickAssignModal}>
        <div className={styles.modalHeader}>
          <div className={styles.modalTitle}>Assign to Cast</div>
          <button className={styles.modalClose} onClick={onClose}>×</button>
        </div>
        <div className={styles.modalBody}>
          {suggestions.length > 0 && (
            <>
              <div className={styles.inputLabel}>Suggested Matches</div>
              {suggestions.slice(0, 5).map((sugg) => (
                <div
                  key={sugg.cast_id}
                  className={styles.suggestionItem}
                  onClick={() => onAssign(sugg.cast_name, sugg.cast_id)}
                >
                  <div className={styles.suggestionThumb}>
                    {sugg.thumbnail_url ? (
                      <img src={sugg.thumbnail_url} alt={sugg.cast_name} />
                    ) : (
                      <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", background: "#f1f5f9" }}>👤</div>
                    )}
                  </div>
                  <div className={styles.suggestionInfo}>
                    <div className={styles.suggestionName}>{sugg.cast_name}</div>
                    <div className={styles.suggestionScore}>
                      {formatPercent(sugg.similarity)} match • #{sugg.rank}
                    </div>
                  </div>
                </div>
              ))}
              <div style={{ margin: "16px 0", borderTop: "1px solid #e2e8f0" }} />
            </>
          )}

          <div className={styles.inputGroup}>
            <label className={styles.inputLabel}>Select from roster</label>
            <select
              className={styles.select}
              value={selectedRoster}
              onChange={(e) => {
                setSelectedRoster(e.target.value);
                if (e.target.value) setCustomName("");
              }}
            >
              <option value="">Choose a cast member...</option>
              {rosterNames.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          </div>

          <div className={styles.inputGroup}>
            <label className={styles.inputLabel}>Or enter new name</label>
            <input
              type="text"
              className={styles.input}
              placeholder="Enter name..."
              value={customName}
              onChange={(e) => {
                setCustomName(e.target.value);
                if (e.target.value) setSelectedRoster("");
              }}
            />
          </div>
        </div>
        <div className={styles.modalFooter}>
          <button className={`${styles.btn} ${styles.btnSecondary}`} onClick={onClose}>
            Cancel
          </button>
          <button
            className={`${styles.btn} ${styles.btnPrimary}`}
            onClick={handleSubmit}
            disabled={isAssigning || (!selectedRoster && !customName.trim())}
          >
            {isAssigning ? "Assigning..." : "Assign"}
          </button>
        </div>
      </div>
    </>
  );
}

// Main view component
function MainView({
  episodeId,
  showSlug,
  onSelectCastMember,
  onSelectCluster,
}: {
  episodeId: string;
  showSlug?: string;
  onSelectCastMember: (castId: string) => void;
  onSelectCluster: (clusterId: string) => void;
}) {
  const [activeTab, setActiveTab] = useState<"assigned" | "unassigned">("assigned");

  const peopleQuery = useShowPeople(showSlug);
  const unlinkedQuery = useUnlinkedEntities(episodeId);
  const suggestionsQuery = useCastSuggestions(episodeId);

  const people = peopleQuery.data?.people || [];
  const unassignedClusters = unlinkedQuery.data?.unassigned_clusters || [];
  const autoPeople = unlinkedQuery.data?.auto_people || [];

  // Build suggestions lookup
  const suggestionsMap = useMemo(() => {
    const map = new Map<string, CastSuggestion[]>();
    suggestionsQuery.data?.suggestions.forEach((s) => {
      map.set(s.cluster_id, s.cast_suggestions);
    });
    return map;
  }, [suggestionsQuery.data]);

  const assignedCount = people.length;
  const unassignedCount = unassignedClusters.length + autoPeople.reduce((sum, p) => sum + p.clusters.length, 0);

  return (
    <>
      <div className={styles.tabs}>
        <button
          className={`${styles.tab} ${activeTab === "assigned" ? styles.tabActive : ""}`}
          onClick={() => setActiveTab("assigned")}
        >
          Cast Members
          <span className={styles.tabBadge}>{assignedCount}</span>
        </button>
        <button
          className={`${styles.tab} ${activeTab === "unassigned" ? styles.tabActive : ""}`}
          onClick={() => setActiveTab("unassigned")}
        >
          Needs Assignment
          <span className={styles.tabBadge}>{unassignedCount}</span>
        </button>
      </div>

      {activeTab === "assigned" && (
        <>
          {people.length === 0 ? (
            <div className={styles.empty}>
              <div className={styles.emptyIcon}>🎭</div>
              <div className={styles.emptyTitle}>No Cast Members Yet</div>
              <div className={styles.emptyText}>
                Assign clusters to cast members to see them here.
              </div>
            </div>
          ) : (
            <div className={`${styles.grid} ${styles.gridLarge}`}>
              {people.map((person) => (
                <CastMemberCard
                  key={person.person_id}
                  person={person}
                  onClick={() => onSelectCastMember(person.person_id)}
                />
              ))}
            </div>
          )}
        </>
      )}

      {activeTab === "unassigned" && (
        <>
          {/* Auto-grouped people (unnamed) */}
          {autoPeople.length > 0 && (
            <div style={{ marginBottom: 24 }}>
              <div className={styles.sectionHeader}>
                <div className={styles.sectionTitle}>
                  Auto-Grouped
                  <span className={styles.sectionBadge}>{autoPeople.length} groups</span>
                </div>
              </div>
              <div className={`${styles.grid} ${styles.gridLarge}`}>
                {autoPeople.map((group) => (
                  <div key={group.person_id}>
                    <div style={{ marginBottom: 8, fontSize: 13, color: "#64748b" }}>
                      {group.name || `Group ${group.person_id.slice(-6)}`} ({group.clusters.length} clusters)
                    </div>
                    {group.clusters.map((cluster) => (
                      <ClusterCard
                        key={cluster.identity_id}
                        identity={cluster}
                        onClick={() => onSelectCluster(cluster.identity_id)}
                        suggestions={suggestionsMap.get(cluster.identity_id)}
                      />
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Standalone unassigned clusters */}
          {unassignedClusters.length > 0 && (
            <div>
              <div className={styles.sectionHeader}>
                <div className={styles.sectionTitle}>
                  Unassigned Clusters
                  <span className={styles.sectionBadge}>{unassignedClusters.length}</span>
                </div>
              </div>
              <div className={`${styles.grid} ${styles.gridLarge}`}>
                {unassignedClusters.map((cluster) => (
                  <ClusterCard
                    key={cluster.identity_id}
                    identity={cluster}
                    onClick={() => onSelectCluster(cluster.identity_id)}
                    suggestions={suggestionsMap.get(cluster.identity_id)}
                  />
                ))}
              </div>
            </div>
          )}

          {unassignedClusters.length === 0 && autoPeople.length === 0 && (
            <div className={styles.empty}>
              <div className={styles.emptyIcon}>✅</div>
              <div className={styles.emptyTitle}>All Assigned!</div>
              <div className={styles.emptyText}>
                All clusters have been assigned to cast members.
              </div>
            </div>
          )}
        </>
      )}
    </>
  );
}

// Cluster view component
function ClusterView({
  episodeId,
  clusterId,
  showSlug,
  onSelectTrack,
  onBack,
}: {
  episodeId: string;
  clusterId: string;
  showSlug?: string;
  onSelectTrack: (trackId: number) => void;
  onBack: () => void;
}) {
  const trackRepsQuery = useClusterTrackReps(episodeId, clusterId);
  const suggestionsQuery = useCastSuggestions(episodeId);
  const rosterQuery = useRosterNames(showSlug);
  const assignTrack = useAssignTrack();
  const bulkAssign = useBulkAssignTracks();
  const createBackup = useCreateBackup();

  const [showAssignModal, setShowAssignModal] = useState(false);
  const bulkSelection = useBulkTrackSelection();

  const tracks = trackRepsQuery.data?.tracks || [];
  const clusterInfo = trackRepsQuery.data;

  // Get suggestions for this cluster
  const suggestions = useMemo(() => {
    const clusterSuggestions = suggestionsQuery.data?.suggestions.find(
      (s) => s.cluster_id === clusterId
    );
    return clusterSuggestions?.cast_suggestions || [];
  }, [suggestionsQuery.data, clusterId]);

  const handleAssign = async (name: string, castId?: string) => {
    // Create backup first
    await createBackup.mutateAsync(episodeId);

    if (bulkSelection.count > 0) {
      // Bulk assign selected tracks
      await bulkAssign.mutateAsync({
        episodeId,
        payload: {
          track_ids: Array.from(bulkSelection.selectedTracks),
          name,
          show: showSlug,
          cast_id: castId,
        },
      });
      bulkSelection.clearSelection();
    } else if (tracks.length > 0) {
      // Assign first track (which assigns whole cluster)
      await assignTrack.mutateAsync({
        episodeId,
        trackId: tracks[0].track_id,
        payload: { name, show: showSlug, cast_id: castId },
      });
    }
    setShowAssignModal(false);
  };

  return (
    <>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionTitle}>
          Cluster {clusterId.slice(-8)}
          <span className={styles.sectionBadge}>{tracks.length} tracks</span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {bulkSelection.count > 0 && (
            <button
              className={`${styles.btn} ${styles.btnSecondary}`}
              onClick={bulkSelection.clearSelection}
            >
              Clear ({bulkSelection.count})
            </button>
          )}
          <button
            className={`${styles.btn} ${styles.btnPrimary}`}
            onClick={() => setShowAssignModal(true)}
          >
            Assign to Cast
          </button>
        </div>
      </div>

      {/* Cluster metrics */}
      {clusterInfo && (
        <div className={styles.badges} style={{ marginBottom: 16 }}>
          {clusterInfo.cohesion !== undefined && (
            <SimilarityBadge type="cluster" value={clusterInfo.cohesion} label="Cohesion" />
          )}
          {clusterInfo.isolation !== undefined && (
            <SimilarityBadge type="isolation" value={clusterInfo.isolation} label="Isolation" />
          )}
        </div>
      )}

      {/* Suggestions preview */}
      {suggestions.length > 0 && (
        <div style={{ marginBottom: 16, padding: 12, background: "#f8fafc", borderRadius: 8 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 8 }}>Top Suggestions:</div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {suggestions.slice(0, 3).map((sugg) => (
              <span
                key={sugg.cast_id}
                style={{
                  padding: "4px 10px",
                  background: "#f3e8ff",
                  borderRadius: 6,
                  fontSize: 12,
                  cursor: "pointer",
                }}
                onClick={() => handleAssign(sugg.cast_name, sugg.cast_id)}
              >
                {sugg.cast_name} ({formatPercent(sugg.similarity)})
              </span>
            ))}
          </div>
        </div>
      )}

      {tracks.length === 0 ? (
        <div className={styles.empty}>
          <div className={styles.emptyIcon}>🖼️</div>
          <div className={styles.emptyTitle}>No Tracks</div>
          <div className={styles.emptyText}>This cluster has no tracks.</div>
        </div>
      ) : (
        <div className={`${styles.grid} ${styles.gridLarge}`}>
          {tracks.map((track) => (
            <TrackCard
              key={track.track_id}
              track={track}
              onClick={() => onSelectTrack(track.track_id)}
              isSelected={bulkSelection.isSelected(track.track_id)}
              onCheckboxClick={() => bulkSelection.toggleTrack(track.track_id)}
              showCheckbox
            />
          ))}
        </div>
      )}

      <QuickAssignModal
        isOpen={showAssignModal}
        onClose={() => setShowAssignModal(false)}
        suggestions={suggestions}
        rosterNames={rosterQuery.data || []}
        onAssign={handleAssign}
        isAssigning={assignTrack.isPending || bulkAssign.isPending}
      />
    </>
  );
}

// Track view component (frame gallery)
function TrackView({
  episodeId,
  trackId,
  showSlug,
  onBack,
}: {
  episodeId: string;
  trackId: number;
  showSlug?: string;
  onBack: () => void;
}) {
  const [page, setPage] = useState(1);
  const [includeSkipped, setIncludeSkipped] = useState(false);
  const [selectedFrames, setSelectedFrames] = useState<Set<number>>(new Set());

  const framesQuery = useTrackFrames(episodeId, trackId, {
    page,
    pageSize: 50,
    includeSkipped,
  });
  const moveFrames = useMoveFrames();
  const deleteFrames = useDeleteFrames();

  const frames = framesQuery.data?.frames || [];
  const hasMore = framesQuery.data?.has_more || false;
  const totalFrames = framesQuery.data?.total_frames || 0;

  const toggleFrame = (frameIdx: number) => {
    setSelectedFrames((prev) => {
      const next = new Set(prev);
      if (next.has(frameIdx)) {
        next.delete(frameIdx);
      } else {
        next.add(frameIdx);
      }
      return next;
    });
  };

  const handleDeleteSelected = async () => {
    if (selectedFrames.size === 0) return;
    if (!confirm(`Delete ${selectedFrames.size} frame(s)? This cannot be undone.`)) return;

    await deleteFrames.mutateAsync({
      episodeId,
      trackId,
      payload: {
        frame_ids: Array.from(selectedFrames),
        delete_assets: true,
      },
    });
    setSelectedFrames(new Set());
  };

  return (
    <>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionTitle}>
          Track {trackId}
          <span className={styles.sectionBadge}>{totalFrames} frames</span>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, color: "#64748b" }}>
            <input
              type="checkbox"
              checked={includeSkipped}
              onChange={(e) => setIncludeSkipped(e.target.checked)}
            />
            Show skipped
          </label>
          {selectedFrames.size > 0 && (
            <>
              <button
                className={`${styles.btn} ${styles.btnSecondary}`}
                onClick={() => setSelectedFrames(new Set())}
              >
                Clear ({selectedFrames.size})
              </button>
              <button
                className={`${styles.btn} ${styles.btnDanger}`}
                onClick={handleDeleteSelected}
                disabled={deleteFrames.isPending}
              >
                {deleteFrames.isPending ? "Deleting..." : "Delete Selected"}
              </button>
            </>
          )}
        </div>
      </div>

      {frames.length === 0 ? (
        <div className={styles.empty}>
          <div className={styles.emptyIcon}>🖼️</div>
          <div className={styles.emptyTitle}>No Frames</div>
          <div className={styles.emptyText}>This track has no frames.</div>
        </div>
      ) : (
        <>
          <div className={`${styles.grid} ${styles.gridSmall}`}>
            {frames.map((frame) => (
              <FrameCard
                key={frame.frame_idx}
                frame={frame}
                isSelected={selectedFrames.has(frame.frame_idx)}
                onClick={() => toggleFrame(frame.frame_idx)}
              />
            ))}
          </div>

          {/* Pagination */}
          <div style={{ display: "flex", justifyContent: "center", gap: 8, marginTop: 24 }}>
            <button
              className={`${styles.btn} ${styles.btnSecondary}`}
              disabled={page === 1}
              onClick={() => setPage((p) => p - 1)}
            >
              Previous
            </button>
            <span style={{ display: "flex", alignItems: "center", padding: "0 12px", color: "#64748b", fontSize: 14 }}>
              Page {page}
            </span>
            <button
              className={`${styles.btn} ${styles.btnSecondary}`}
              disabled={!hasMore}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </button>
          </div>
        </>
      )}
    </>
  );
}

// Main page component
export default function FacesReviewPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id: episodeId } = use(params);

  // Queries
  const detailsQuery = useEpisodeDetails(episodeId);
  const progressQuery = useReviewProgress(episodeId);

  // Mutations
  const refreshSimilarity = useRefreshSimilarity();
  const autoLinkCast = useAutoLinkCast();
  const saveAssignments = useSaveAssignments();

  // View state
  const viewState = useFacesReviewState(episodeId);

  // Undo stack
  const undoStack = useUndoStack(episodeId);

  const details = detailsQuery.data;
  const showSlug = details?.show_slug;
  const progress = progressQuery.data;

  // Loading state
  if (detailsQuery.isLoading) {
    return <div className={styles.loading}>Loading episode...</div>;
  }

  if (detailsQuery.error) {
    return (
      <div className={styles.error}>
        <p>Failed to load episode: {detailsQuery.error.message}</p>
        <Link href="/screenalytics/episodes">← Back to Episodes</Link>
      </div>
    );
  }

  // Build breadcrumbs
  const breadcrumbs: Array<{ label: string; onClick?: () => void }> = [
    { label: "Faces Review", onClick: viewState.goToMain },
  ];

  if (viewState.view === "cast_member" && viewState.selectedCastId) {
    breadcrumbs.push({ label: `Cast Member` });
  } else if (viewState.view === "cluster" && viewState.selectedClusterId) {
    breadcrumbs.push({
      label: `Cluster ${viewState.selectedClusterId.slice(-8)}`,
    });
  } else if (viewState.view === "track" && viewState.selectedTrackId !== null) {
    if (viewState.selectedClusterId) {
      breadcrumbs.push({
        label: `Cluster ${viewState.selectedClusterId.slice(-8)}`,
        onClick: () => viewState.goToCluster(viewState.selectedClusterId!),
      });
    }
    breadcrumbs.push({ label: `Track ${viewState.selectedTrackId}` });
  }

  return (
    <div className={styles.page}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <Link href={`/screenalytics/episodes/${episodeId}`} className={styles.backLink}>
            ← Episode Detail
          </Link>
          <h1 className={styles.title}>Faces Review</h1>
          <p className={styles.subtitle}>
            <span>{episodeId}</span>
            {details?.show_slug && <span>{details.show_slug}</span>}
          </p>
        </div>
      </div>

      {/* Progress Bar */}
      {progress && (
        <div className={styles.progressBar}>
          <div className={styles.progressHeader}>
            <span className={styles.progressLabel}>Review Progress</span>
            <span className={styles.progressPercent}>
              {Math.round(progress.percent_complete)}%
            </span>
          </div>
          <div className={styles.progressTrack}>
            <div
              className={styles.progressFill}
              style={{ width: `${progress.percent_complete}%` }}
            />
          </div>
          <div className={styles.progressStats}>
            <span>{progress.assigned_clusters} assigned</span>
            <span>{progress.unassigned_clusters} unassigned</span>
            <span>{progress.singleton_count} singletons</span>
          </div>
        </div>
      )}

      {/* Action Bar */}
      <div className={styles.actionBar}>
        <button
          className={`${styles.actionBtn} ${styles.actionBtnPrimary}`}
          onClick={() => refreshSimilarity.mutate(episodeId)}
          disabled={refreshSimilarity.isPending}
        >
          {refreshSimilarity.isPending ? "Refreshing..." : "🔄 Refresh Suggestions"}
        </button>
        <button
          className={styles.actionBtn}
          onClick={() => autoLinkCast.mutate(episodeId)}
          disabled={autoLinkCast.isPending}
        >
          {autoLinkCast.isPending ? "Auto-linking..." : "🔗 Auto-Assign"}
        </button>
        <button
          className={styles.actionBtn}
          onClick={() => saveAssignments.mutate(episodeId)}
          disabled={saveAssignments.isPending}
        >
          {saveAssignments.isPending ? "Saving..." : "💾 Save Progress"}
        </button>
        {undoStack.canUndo && (
          <button
            className={styles.actionBtn}
            onClick={undoStack.undo}
            disabled={undoStack.isUndoing}
          >
            {undoStack.isUndoing ? "Undoing..." : `↩️ Undo (${undoStack.stack.length})`}
          </button>
        )}
      </div>

      {/* Breadcrumbs */}
      {breadcrumbs.length > 1 && (
        <div className={styles.breadcrumbs}>
          {breadcrumbs.map((crumb, idx) => (
            <span key={idx}>
              {idx > 0 && <span className={styles.breadcrumbSep}> / </span>}
              <span
                className={idx === breadcrumbs.length - 1 ? styles.breadcrumbActive : styles.breadcrumb}
                onClick={crumb.onClick}
              >
                {crumb.label}
              </span>
            </span>
          ))}
        </div>
      )}

      {/* View Content */}
      {viewState.view === "main" && (
        <MainView
          episodeId={episodeId}
          showSlug={showSlug}
          onSelectCastMember={viewState.goToCastMember}
          onSelectCluster={viewState.goToCluster}
        />
      )}

      {viewState.view === "cluster" && viewState.selectedClusterId && (
        <ClusterView
          episodeId={episodeId}
          clusterId={viewState.selectedClusterId}
          showSlug={showSlug}
          onSelectTrack={viewState.goToTrack}
          onBack={viewState.goBack}
        />
      )}

      {viewState.view === "track" && viewState.selectedTrackId !== null && (
        <TrackView
          episodeId={episodeId}
          trackId={viewState.selectedTrackId}
          showSlug={showSlug}
          onBack={viewState.goBack}
        />
      )}
    </div>
  );
}
