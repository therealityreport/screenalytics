"use client";

import { useEffect, useMemo, useState, useCallback, useRef } from "react";
import Link from "next/link";
import {
  useEpisodes,
  useDeleteEpisode,
  useBulkDeleteEpisodes,
  useEpisodeStatuses,
  useFavorites,
  useRecentEpisodes,
  useTriggerPhase,
} from "@/api/hooks";
import type { EpisodeSummary, EpisodeViewMode, EpisodeSortOption, EpisodeStatus } from "@/api/types";
import styles from "./episodes.module.css";

// Helper to derive status from phase data
function getEpisodeProcessingStatus(status?: EpisodeStatus): "pending" | "processing" | "complete" | "error" {
  if (!status) return "pending";

  const detectStatus = status.detect_track?.status;
  const facesStatus = status.faces_embed?.status;
  const clusterStatus = status.cluster?.status;

  // Check for any errors
  if (detectStatus === "error" || facesStatus === "error" || clusterStatus === "error") {
    return "error";
  }

  // Check if processing
  if (detectStatus === "running" || facesStatus === "running" || clusterStatus === "running") {
    return "processing";
  }

  // Check if complete (all phases done)
  if (
    (detectStatus === "complete" || detectStatus === "done") &&
    (facesStatus === "complete" || facesStatus === "done") &&
    (clusterStatus === "complete" || clusterStatus === "done")
  ) {
    return "complete";
  }

  // Partial progress
  if (detectStatus === "complete" || detectStatus === "done") {
    return "processing";
  }

  return "pending";
}

// Get track count from status
function getTrackCount(status?: EpisodeStatus): number | null {
  return status?.detect_track?.tracks ?? null;
}

// Get cluster count from status
function getClusterCount(status?: EpisodeStatus): number | null {
  return status?.cluster?.identities ?? null;
}

type GroupedEpisodes = {
  [show: string]: {
    [season: number]: EpisodeSummary[];
  };
};

function groupEpisodes(episodes: EpisodeSummary[]): GroupedEpisodes {
  const grouped: GroupedEpisodes = {};
  for (const ep of episodes) {
    const show = ep.show_slug || "Unknown";
    if (!grouped[show]) grouped[show] = {};
    const season = ep.season_number ?? 0;
    if (!grouped[show][season]) grouped[show][season] = [];
    grouped[show][season].push(ep);
  }
  // Sort episodes within each season
  for (const show of Object.keys(grouped)) {
    for (const season of Object.keys(grouped[show])) {
      grouped[show][Number(season)].sort((a, b) => a.episode_number - b.episode_number);
    }
  }
  return grouped;
}

function sortEpisodes(
  episodes: EpisodeSummary[],
  sortBy: EpisodeSortOption,
  statusMap?: Map<string, EpisodeStatus>
): EpisodeSummary[] {
  const sorted = [...episodes];

  switch (sortBy) {
    case "show-season-episode":
      sorted.sort((a, b) => {
        const showCmp = (a.show_slug || "").localeCompare(b.show_slug || "");
        if (showCmp !== 0) return showCmp;
        const seasonCmp = (a.season_number ?? 0) - (b.season_number ?? 0);
        if (seasonCmp !== 0) return seasonCmp;
        return (a.episode_number ?? 0) - (b.episode_number ?? 0);
      });
      break;
    case "newest-first":
      sorted.sort((a, b) => {
        const dateA = a.air_date ? new Date(a.air_date).getTime() : 0;
        const dateB = b.air_date ? new Date(b.air_date).getTime() : 0;
        return dateB - dateA;
      });
      break;
    case "oldest-first":
      sorted.sort((a, b) => {
        const dateA = a.air_date ? new Date(a.air_date).getTime() : 0;
        const dateB = b.air_date ? new Date(b.air_date).getTime() : 0;
        return dateA - dateB;
      });
      break;
    case "most-tracks":
      sorted.sort((a, b) => {
        const tracksA = statusMap ? getTrackCount(statusMap.get(a.ep_id)) ?? 0 : 0;
        const tracksB = statusMap ? getTrackCount(statusMap.get(b.ep_id)) ?? 0 : 0;
        return tracksB - tracksA;
      });
      break;
    case "alphabetical":
      sorted.sort((a, b) => a.ep_id.localeCompare(b.ep_id));
      break;
  }

  return sorted;
}

// Status badge component
function StatusBadge({ status }: { status: "pending" | "processing" | "complete" | "error" }) {
  const statusStyles: Record<string, string> = {
    pending: styles.statusPending,
    processing: styles.statusProcessing,
    complete: styles.statusComplete,
    error: styles.statusError,
  };

  const labels: Record<string, string> = {
    pending: "Pending",
    processing: "Processing",
    complete: "Complete",
    error: "Error",
  };

  return (
    <span className={`${styles.statusBadge} ${statusStyles[status]}`}>
      {labels[status]}
    </span>
  );
}

// Episode card component
function EpisodeCard({
  episode,
  status,
  isSelected,
  isFavorite,
  isFocused,
  onSelect,
  onDelete,
  onFavoriteToggle,
  onOpenFaces,
  onRerunCluster,
}: {
  episode: EpisodeSummary;
  status?: EpisodeStatus;
  isSelected: boolean;
  isFavorite: boolean;
  isFocused: boolean;
  onSelect: (epId: string, shiftKey: boolean) => void;
  onDelete: (epId: string) => void;
  onFavoriteToggle: (epId: string) => void;
  onOpenFaces: (epId: string) => void;
  onRerunCluster: (epId: string) => void;
}) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const processingStatus = getEpisodeProcessingStatus(status);
  const trackCount = getTrackCount(status);
  const clusterCount = getClusterCount(status);

  const cardClasses = [
    styles.episodeCard,
    isSelected && styles.episodeCardSelected,
    isFocused && styles.episodeCardFocused,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={cardClasses} data-ep-id={episode.ep_id}>
      {/* Selection checkbox */}
      <input
        type="checkbox"
        className={styles.cardCheckbox}
        checked={isSelected}
        onChange={(e) => {
          e.stopPropagation();
          onSelect(episode.ep_id, e.shiftKey);
        }}
      />

      {/* Thumbnail */}
      <div className={styles.cardThumbnail}>
        <div className={styles.thumbnailPlaceholder}>
          {episode.show_slug?.substring(0, 2).toUpperCase() || "EP"}
        </div>
        <div className={styles.thumbnailOverlay}>
          <StatusBadge status={processingStatus} />
        </div>
      </div>

      {/* Quick action buttons */}
      <div className={styles.cardActions}>
        <button
          className={`${styles.actionBtn} ${isFavorite ? styles.actionBtnFavorite : ""}`}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onFavoriteToggle(episode.ep_id);
          }}
          title={isFavorite ? "Remove from favorites" : "Add to favorites"}
        >
          {isFavorite ? "★" : "☆"}
        </button>
        <button
          className={styles.actionBtn}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onOpenFaces(episode.ep_id);
          }}
          title="Open Faces Review"
        >
          👤
        </button>
        <button
          className={styles.actionBtn}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onRerunCluster(episode.ep_id);
          }}
          title="Rerun Clustering"
        >
          🔄
        </button>
        {confirmDelete ? (
          <>
            <button
              className={`${styles.actionBtn} ${styles.actionBtnDelete}`}
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                onDelete(episode.ep_id);
                setConfirmDelete(false);
              }}
              title="Confirm delete"
            >
              ✓
            </button>
            <button
              className={styles.actionBtn}
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setConfirmDelete(false);
              }}
              title="Cancel"
            >
              ✕
            </button>
          </>
        ) : (
          <button
            className={`${styles.actionBtn} ${styles.actionBtnDelete}`}
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setConfirmDelete(true);
            }}
            title="Delete episode"
          >
            🗑
          </button>
        )}
      </div>

      {/* Content */}
      <Link href={`/screenalytics/episodes/${episode.ep_id}`} className={styles.episodeLink}>
        <div className={styles.cardContent}>
          <div className={styles.episodeHeader}>
            <span className={styles.episodeNumber}>
              S{String(episode.season_number).padStart(2, "0")}E{String(episode.episode_number).padStart(2, "0")}
            </span>
            {episode.title && <span className={styles.episodeTitle}>{episode.title}</span>}
          </div>
          <div className={styles.episodeId}>{episode.ep_id}</div>

          {/* Stats row */}
          <div className={styles.episodeStats}>
            {trackCount !== null && (
              <span className={styles.stat}>
                <span>📊</span> {trackCount} tracks
              </span>
            )}
            {clusterCount !== null && (
              <span className={styles.stat}>
                <span>👥</span> {clusterCount} identities
              </span>
            )}
          </div>
        </div>
      </Link>

      {/* Progress bar for processing episodes */}
      {processingStatus === "processing" && (
        <div className={styles.progressBar}>
          <div className={styles.progressFill} style={{ width: "50%" }} />
        </div>
      )}
    </div>
  );
}

// Table view component
function EpisodesTable({
  episodes,
  statusMap,
  selectedIds,
  favoriteIds,
  onSelect,
  onDelete,
  onFavoriteToggle,
  sortBy,
  onSortChange,
}: {
  episodes: EpisodeSummary[];
  statusMap: Map<string, EpisodeStatus>;
  selectedIds: Set<string>;
  favoriteIds: string[];
  onSelect: (epId: string, shiftKey: boolean) => void;
  onDelete: (epId: string) => void;
  onFavoriteToggle: (epId: string) => void;
  sortBy: EpisodeSortOption;
  onSortChange: (sort: EpisodeSortOption) => void;
}) {
  return (
    <div className={styles.tableContainer}>
      <table className={styles.episodesTable}>
        <thead>
          <tr>
            <th style={{ width: 40 }}>
              <input type="checkbox" />
            </th>
            <th
              className={styles.sortable}
              onClick={() => onSortChange("show-season-episode")}
            >
              Episode
              <span className={`${styles.sortIndicator} ${sortBy === "show-season-episode" ? styles.sortActive : ""}`}>
                ↕
              </span>
            </th>
            <th>Show</th>
            <th
              className={styles.sortable}
              onClick={() => onSortChange("newest-first")}
            >
              Air Date
              <span className={`${styles.sortIndicator} ${sortBy === "newest-first" || sortBy === "oldest-first" ? styles.sortActive : ""}`}>
                ↕
              </span>
            </th>
            <th>Status</th>
            <th
              className={styles.sortable}
              onClick={() => onSortChange("most-tracks")}
            >
              Tracks
              <span className={`${styles.sortIndicator} ${sortBy === "most-tracks" ? styles.sortActive : ""}`}>
                ↕
              </span>
            </th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {episodes.map((ep) => {
            const status = statusMap.get(ep.ep_id);
            const processingStatus = getEpisodeProcessingStatus(status);
            const isSelected = selectedIds.has(ep.ep_id);
            const isFavorite = favoriteIds.includes(ep.ep_id);

            return (
              <tr key={ep.ep_id} className={isSelected ? styles.selected : ""}>
                <td>
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={(e) => onSelect(ep.ep_id, e.shiftKey)}
                  />
                </td>
                <td>
                  <Link href={`/screenalytics/episodes/${ep.ep_id}`} className={styles.tableEpisodeLink}>
                    S{String(ep.season_number).padStart(2, "0")}E{String(ep.episode_number).padStart(2, "0")}
                    {ep.title && ` - ${ep.title}`}
                  </Link>
                </td>
                <td>{ep.show_slug}</td>
                <td>{ep.air_date || "-"}</td>
                <td>
                  <StatusBadge status={processingStatus} />
                </td>
                <td>{getTrackCount(status) ?? "-"}</td>
                <td>
                  <div className={styles.tableActions}>
                    <button
                      className={styles.actionBtn}
                      onClick={() => onFavoriteToggle(ep.ep_id)}
                      title={isFavorite ? "Remove from favorites" : "Add to favorites"}
                    >
                      {isFavorite ? "★" : "☆"}
                    </button>
                    <button
                      className={`${styles.actionBtn} ${styles.actionBtnDelete}`}
                      onClick={() => onDelete(ep.ep_id)}
                      title="Delete"
                    >
                      🗑
                    </button>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// Timeline view component
function EpisodesTimeline({
  episodes,
  statusMap,
}: {
  episodes: EpisodeSummary[];
  statusMap: Map<string, EpisodeStatus>;
}) {
  // Group by month
  const grouped = useMemo(() => {
    const byMonth: Record<string, EpisodeSummary[]> = {};

    for (const ep of episodes) {
      const date = ep.air_date ? new Date(ep.air_date) : null;
      const key = date
        ? `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`
        : "Unknown";
      if (!byMonth[key]) byMonth[key] = [];
      byMonth[key].push(ep);
    }

    // Sort months descending
    const sortedKeys = Object.keys(byMonth).sort((a, b) => b.localeCompare(a));
    return sortedKeys.map((key) => ({
      month: key,
      label: key === "Unknown" ? "Unknown Date" : new Date(key + "-01").toLocaleDateString("en-US", { month: "long", year: "numeric" }),
      episodes: byMonth[key].sort((a, b) => {
        const dateA = a.air_date ? new Date(a.air_date).getTime() : 0;
        const dateB = b.air_date ? new Date(b.air_date).getTime() : 0;
        return dateB - dateA;
      }),
    }));
  }, [episodes]);

  return (
    <div className={styles.timelineContainer}>
      {grouped.map(({ month, label, episodes: monthEpisodes }) => (
        <div key={month}>
          <div className={styles.timelineMonth}>{label}</div>
          {monthEpisodes.map((ep) => {
            const status = statusMap.get(ep.ep_id);
            const processingStatus = getEpisodeProcessingStatus(status);

            return (
              <Link
                key={ep.ep_id}
                href={`/screenalytics/episodes/${ep.ep_id}`}
                className={styles.timelineItem}
              >
                <div className={styles.timelineDate}>
                  {ep.air_date ? new Date(ep.air_date).toLocaleDateString("en-US", { month: "short", day: "numeric" }) : "-"}
                </div>
                <div className={styles.timelineInfo}>
                  <div className={styles.timelineTitle}>
                    {ep.show_slug} S{String(ep.season_number).padStart(2, "0")}E{String(ep.episode_number).padStart(2, "0")}
                  </div>
                  <div className={styles.timelineSubtitle}>{ep.title || ep.ep_id}</div>
                </div>
                <div className={styles.timelineStatus}>
                  <StatusBadge status={processingStatus} />
                </div>
              </Link>
            );
          })}
        </div>
      ))}
    </div>
  );
}

// Bulk delete confirmation modal
function BulkDeleteModal({
  episodeIds,
  onConfirm,
  onCancel,
  isDeleting,
}: {
  episodeIds: string[];
  onConfirm: () => void;
  onCancel: () => void;
  isDeleting: boolean;
}) {
  return (
    <div className={styles.modal} onClick={onCancel}>
      <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
        <h3 className={styles.modalTitle}>Delete {episodeIds.length} Episodes?</h3>
        <p className={styles.modalText}>
          This will permanently delete the following episodes and all associated data (video, manifests, frames, embeddings).
        </p>
        <div className={styles.modalList}>
          {episodeIds.map((id) => (
            <div key={id}>{id}</div>
          ))}
        </div>
        <div className={styles.modalActions}>
          <button className={styles.modalCancelBtn} onClick={onCancel} disabled={isDeleting}>
            Cancel
          </button>
          <button className={styles.modalDeleteBtn} onClick={onConfirm} disabled={isDeleting}>
            {isDeleting ? "Deleting..." : `Delete ${episodeIds.length} Episodes`}
          </button>
        </div>
      </div>
    </div>
  );
}

// Main page component
export default function EpisodesListPage() {
  // State
  const [search, setSearch] = useState("");
  const [showFilter, setShowFilter] = useState<string>("all");
  const [sortBy, setSortBy] = useState<EpisodeSortOption>("show-season-episode");
  const [viewMode, setViewMode] = useState<EpisodeViewMode>("card");
  const [expandedShows, setExpandedShows] = useState<Set<string>>(new Set());
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [focusedIndex, setFocusedIndex] = useState<number>(-1);
  const [showBulkDeleteModal, setShowBulkDeleteModal] = useState(false);
  const [showKeyboardHint, setShowKeyboardHint] = useState(false);
  const lastSelectedRef = useRef<string | null>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Hooks
  const episodesQuery = useEpisodes();
  const deleteEpisode = useDeleteEpisode();
  const bulkDelete = useBulkDeleteEpisodes();
  const triggerPhase = useTriggerPhase();
  const { getFavorites, addFavorite, removeFavorite, isFavorite } = useFavorites();
  const { getRecent, addRecent, clearRecent } = useRecentEpisodes();

  const episodes = episodesQuery.data || [];
  const episodeIds = useMemo(() => episodes.map((ep) => ep.ep_id), [episodes]);

  // Fetch statuses for all episodes
  const statusesQuery = useEpisodeStatuses(episodeIds, {
    enabled: episodes.length > 0,
    refetchInterval: 10000, // Every 10 seconds
  });
  const statusMap = statusesQuery.data || new Map<string, EpisodeStatus>();

  // Favorites and recent (from localStorage)
  const [favoriteIds, setFavoriteIds] = useState<string[]>([]);
  const [recentIds, setRecentIds] = useState<string[]>([]);

  useEffect(() => {
    setFavoriteIds(getFavorites());
    setRecentIds(getRecent());
  }, [getFavorites, getRecent]);

  // Get unique shows for filter dropdown
  const shows = useMemo(() => {
    const uniqueShows = new Set(episodes.map((ep) => ep.show_slug || "Unknown"));
    return Array.from(uniqueShows).sort();
  }, [episodes]);

  // Filter and sort episodes
  const filteredEpisodes = useMemo(() => {
    let result = episodes;

    // Filter by show
    if (showFilter !== "all") {
      result = result.filter((ep) => (ep.show_slug || "Unknown") === showFilter);
    }

    // Filter by search
    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter(
        (ep) =>
          ep.ep_id.toLowerCase().includes(q) ||
          ep.title?.toLowerCase().includes(q) ||
          ep.show_slug?.toLowerCase().includes(q) ||
          `s${String(ep.season_number).padStart(2, "0")}e${String(ep.episode_number).padStart(2, "0")}`.includes(q)
      );
    }

    // Sort
    result = sortEpisodes(result, sortBy, statusMap);

    return result;
  }, [episodes, showFilter, search, sortBy, statusMap]);

  // Group by show/season for card view
  const grouped = useMemo(() => groupEpisodes(filteredEpisodes), [filteredEpisodes]);

  // Auto-expand all shows initially
  useEffect(() => {
    if (shows.length > 0 && expandedShows.size === 0) {
      setExpandedShows(new Set(shows));
    }
  }, [shows, expandedShows.size]);

  // Toggle show expansion
  const toggleShow = (show: string) => {
    setExpandedShows((prev) => {
      const next = new Set(prev);
      if (next.has(show)) {
        next.delete(show);
      } else {
        next.add(show);
      }
      return next;
    });
  };

  // Selection handlers
  const handleSelect = useCallback((epId: string, shiftKey: boolean) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);

      if (shiftKey && lastSelectedRef.current) {
        // Range selection
        const allIds = filteredEpisodes.map((ep) => ep.ep_id);
        const startIdx = allIds.indexOf(lastSelectedRef.current);
        const endIdx = allIds.indexOf(epId);

        if (startIdx !== -1 && endIdx !== -1) {
          const [from, to] = startIdx < endIdx ? [startIdx, endIdx] : [endIdx, startIdx];
          for (let i = from; i <= to; i++) {
            next.add(allIds[i]);
          }
        }
      } else {
        // Toggle single selection
        if (next.has(epId)) {
          next.delete(epId);
        } else {
          next.add(epId);
        }
      }

      lastSelectedRef.current = epId;
      return next;
    });
  }, [filteredEpisodes]);

  const handleSelectAll = () => {
    setSelectedIds(new Set(filteredEpisodes.map((ep) => ep.ep_id)));
  };

  const handleClearSelection = () => {
    setSelectedIds(new Set());
    lastSelectedRef.current = null;
  };

  // Delete handlers
  const handleDelete = (epId: string) => {
    deleteEpisode.mutate(epId);
  };

  const handleBulkDelete = () => {
    bulkDelete.mutate(
      { episodeIds: Array.from(selectedIds), includeS3: true },
      {
        onSuccess: () => {
          setShowBulkDeleteModal(false);
          handleClearSelection();
        },
      }
    );
  };

  // Favorite toggle
  const handleFavoriteToggle = (epId: string) => {
    if (isFavorite(epId)) {
      removeFavorite(epId);
    } else {
      addFavorite(epId);
    }
    setFavoriteIds(getFavorites());
  };

  // Quick actions
  const handleOpenFaces = (epId: string) => {
    addRecent(epId);
    window.location.href = `/screenalytics/faces?ep_id=${epId}`;
  };

  const handleRerunCluster = (epId: string) => {
    triggerPhase.mutate({ episodeId: epId, phase: "cluster" });
  };

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      switch (e.key) {
        case "/":
          e.preventDefault();
          searchInputRef.current?.focus();
          break;
        case "?":
          setShowKeyboardHint((prev) => !prev);
          break;
        case "Escape":
          if (selectedIds.size > 0) {
            handleClearSelection();
          }
          break;
        case "a":
          if (e.metaKey || e.ctrlKey) {
            e.preventDefault();
            handleSelectAll();
          }
          break;
        case "Delete":
        case "Backspace":
          if (selectedIds.size > 0) {
            setShowBulkDeleteModal(true);
          }
          break;
        case "ArrowDown":
        case "j":
          e.preventDefault();
          setFocusedIndex((prev) => Math.min(prev + 1, filteredEpisodes.length - 1));
          break;
        case "ArrowUp":
        case "k":
          e.preventDefault();
          setFocusedIndex((prev) => Math.max(prev - 1, 0));
          break;
        case "Enter":
          if (focusedIndex >= 0 && focusedIndex < filteredEpisodes.length) {
            window.location.href = `/screenalytics/episodes/${filteredEpisodes[focusedIndex].ep_id}`;
          }
          break;
        case " ":
          e.preventDefault();
          if (focusedIndex >= 0 && focusedIndex < filteredEpisodes.length) {
            handleSelect(filteredEpisodes[focusedIndex].ep_id, e.shiftKey);
          }
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [filteredEpisodes, focusedIndex, selectedIds, handleSelect]);

  // Get recent and favorite episodes data
  const recentEpisodes = useMemo(() => {
    return recentIds
      .map((id) => episodes.find((ep) => ep.ep_id === id))
      .filter((ep): ep is EpisodeSummary => ep !== undefined)
      .slice(0, 5);
  }, [recentIds, episodes]);

  const favoriteEpisodes = useMemo(() => {
    return favoriteIds
      .map((id) => episodes.find((ep) => ep.ep_id === id))
      .filter((ep): ep is EpisodeSummary => ep !== undefined);
  }, [favoriteIds, episodes]);

  return (
    <div className={styles.page}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h1 className={styles.title}>Episodes</h1>
          <p className={styles.subtitle}>
            {episodes.length} episodes across {shows.length} shows
          </p>
        </div>
        <div className={styles.headerActions}>
          <Link href="/screenalytics/upload" className={styles.uploadBtn}>
            + Upload Episode
          </Link>
        </div>
      </div>

      {/* Quick access sections */}
      {(favoriteEpisodes.length > 0 || recentEpisodes.length > 0) && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Favorites */}
          {favoriteEpisodes.length > 0 && (
            <div className={styles.quickSection}>
              <div className={styles.quickSectionHeader}>
                <span className={styles.quickSectionTitle}>★ Favorites</span>
              </div>
              <div className={styles.quickList}>
                {favoriteEpisodes.map((ep) => (
                  <Link
                    key={ep.ep_id}
                    href={`/screenalytics/episodes/${ep.ep_id}`}
                    className={`${styles.quickChip} ${styles.quickChipFavorite}`}
                  >
                    <span className={styles.starIcon}>★</span>
                    {ep.show_slug} S{String(ep.season_number).padStart(2, "0")}E{String(ep.episode_number).padStart(2, "0")}
                  </Link>
                ))}
              </div>
            </div>
          )}

          {/* Recent */}
          {recentEpisodes.length > 0 && (
            <div className={styles.quickSection}>
              <div className={styles.quickSectionHeader}>
                <span className={styles.quickSectionTitle}>Recent</span>
                <button className={styles.clearRecentBtn} onClick={clearRecent}>
                  Clear
                </button>
              </div>
              <div className={styles.quickList}>
                {recentEpisodes.map((ep) => (
                  <Link
                    key={ep.ep_id}
                    href={`/screenalytics/episodes/${ep.ep_id}`}
                    className={styles.quickChip}
                  >
                    {ep.show_slug} S{String(ep.season_number).padStart(2, "0")}E{String(ep.episode_number).padStart(2, "0")}
                  </Link>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Toolbar */}
      <div className={styles.toolbar}>
        <input
          ref={searchInputRef}
          type="text"
          placeholder="Search episodes... (press / to focus)"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className={styles.searchInput}
        />
        <select
          value={showFilter}
          onChange={(e) => setShowFilter(e.target.value)}
          className={styles.filterSelect}
        >
          <option value="all">All Shows</option>
          {shows.map((show) => (
            <option key={show} value={show}>
              {show}
            </option>
          ))}
        </select>
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as EpisodeSortOption)}
          className={styles.filterSelect}
        >
          <option value="show-season-episode">Show / Season / Episode</option>
          <option value="newest-first">Newest First</option>
          <option value="oldest-first">Oldest First</option>
          <option value="most-tracks">Most Tracks</option>
          <option value="alphabetical">Alphabetical</option>
        </select>
        <div className={styles.viewToggle}>
          <button
            className={`${styles.viewBtn} ${viewMode === "card" ? styles.viewBtnActive : ""}`}
            onClick={() => setViewMode("card")}
          >
            Cards
          </button>
          <button
            className={`${styles.viewBtn} ${viewMode === "table" ? styles.viewBtnActive : ""}`}
            onClick={() => setViewMode("table")}
          >
            Table
          </button>
          <button
            className={`${styles.viewBtn} ${viewMode === "timeline" ? styles.viewBtnActive : ""}`}
            onClick={() => setViewMode("timeline")}
          >
            Timeline
          </button>
        </div>
      </div>

      {/* Selection toolbar */}
      {selectedIds.size > 0 && (
        <div className={styles.selectionToolbar}>
          <span className={styles.selectionCount}>
            {selectedIds.size} selected
          </span>
          <div className={styles.selectionActions}>
            <button className={styles.selectAllBtn} onClick={handleSelectAll}>
              Select All ({filteredEpisodes.length})
            </button>
            <button className={styles.clearSelectionBtn} onClick={handleClearSelection}>
              Clear
            </button>
            <button
              className={styles.bulkDeleteBtn}
              onClick={() => setShowBulkDeleteModal(true)}
              disabled={bulkDelete.isPending}
            >
              Delete Selected
            </button>
          </div>
        </div>
      )}

      {/* Loading state */}
      {episodesQuery.isLoading && (
        <div className={styles.loading}>Loading episodes...</div>
      )}

      {/* Error state */}
      {episodesQuery.isError && (
        <div className={styles.error}>
          Failed to load episodes: {episodesQuery.error?.message || "Unknown error"}
        </div>
      )}

      {/* Empty state */}
      {!episodesQuery.isLoading && episodes.length === 0 && (
        <div className={styles.empty}>
          <div className={styles.emptyIcon}>📺</div>
          <h3 className={styles.emptyTitle}>No episodes yet</h3>
          <p className={styles.emptyText}>
            Upload your first episode to start tracking faces and screen time.
          </p>
          <Link href="/screenalytics/upload" className={styles.emptyAction}>
            + Upload Episode
          </Link>
        </div>
      )}

      {/* No results state */}
      {!episodesQuery.isLoading && episodes.length > 0 && filteredEpisodes.length === 0 && (
        <div className={styles.empty}>
          <div className={styles.emptyIcon}>🔍</div>
          <h3 className={styles.emptyTitle}>No episodes match your filters</h3>
          <p className={styles.emptyText}>
            Try adjusting your search or filter criteria.
          </p>
        </div>
      )}

      {/* Card view (grouped by show/season) */}
      {viewMode === "card" && filteredEpisodes.length > 0 && (
        <div className={styles.showsList}>
          {Object.keys(grouped)
            .sort()
            .map((show) => (
              <div key={show} className={styles.showGroup}>
                <button
                  className={styles.showHeader}
                  onClick={() => toggleShow(show)}
                >
                  <span className={styles.showName}>{show}</span>
                  <span className={styles.showCount}>
                    {Object.values(grouped[show]).flat().length} episodes
                  </span>
                  <span className={`${styles.chevron} ${expandedShows.has(show) ? styles.chevronOpen : ""}`}>
                    ▼
                  </span>
                </button>

                {expandedShows.has(show) && (
                  <div className={styles.seasonsContainer}>
                    {Object.keys(grouped[show])
                      .map(Number)
                      .sort((a, b) => a - b)
                      .map((season) => (
                        <div key={season} className={styles.seasonGroup}>
                          <div className={styles.seasonHeader}>
                            Season {season}
                          </div>
                          <div className={styles.episodesGrid}>
                            {grouped[show][season].map((ep, idx) => {
                              const globalIdx = filteredEpisodes.findIndex((e) => e.ep_id === ep.ep_id);
                              return (
                                <EpisodeCard
                                  key={ep.ep_id}
                                  episode={ep}
                                  status={statusMap.get(ep.ep_id)}
                                  isSelected={selectedIds.has(ep.ep_id)}
                                  isFavorite={favoriteIds.includes(ep.ep_id)}
                                  isFocused={globalIdx === focusedIndex}
                                  onSelect={handleSelect}
                                  onDelete={handleDelete}
                                  onFavoriteToggle={handleFavoriteToggle}
                                  onOpenFaces={handleOpenFaces}
                                  onRerunCluster={handleRerunCluster}
                                />
                              );
                            })}
                          </div>
                        </div>
                      ))}
                  </div>
                )}
              </div>
            ))}
        </div>
      )}

      {/* Table view */}
      {viewMode === "table" && filteredEpisodes.length > 0 && (
        <EpisodesTable
          episodes={filteredEpisodes}
          statusMap={statusMap}
          selectedIds={selectedIds}
          favoriteIds={favoriteIds}
          onSelect={handleSelect}
          onDelete={handleDelete}
          onFavoriteToggle={handleFavoriteToggle}
          sortBy={sortBy}
          onSortChange={setSortBy}
        />
      )}

      {/* Timeline view */}
      {viewMode === "timeline" && filteredEpisodes.length > 0 && (
        <EpisodesTimeline episodes={filteredEpisodes} statusMap={statusMap} />
      )}

      {/* Bulk delete modal */}
      {showBulkDeleteModal && (
        <BulkDeleteModal
          episodeIds={Array.from(selectedIds)}
          onConfirm={handleBulkDelete}
          onCancel={() => setShowBulkDeleteModal(false)}
          isDeleting={bulkDelete.isPending}
        />
      )}

      {/* Keyboard shortcuts hint */}
      {showKeyboardHint && (
        <div className={styles.keyboardHint}>
          <kbd>/</kbd> Search · <kbd>j</kbd>/<kbd>k</kbd> Navigate · <kbd>Space</kbd> Select · <kbd>Enter</kbd> Open · <kbd>?</kbd> Toggle help
        </div>
      )}
    </div>
  );
}
