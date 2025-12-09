"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useEpisodes, useDeleteEpisode } from "@/api/hooks";
import type { EpisodeSummary } from "@/api/types";
import styles from "./episodes.module.css";

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

function EpisodeCard({ episode, onDelete }: { episode: EpisodeSummary; onDelete: (id: string) => void }) {
  const [confirmDelete, setConfirmDelete] = useState(false);

  return (
    <div className={styles.episodeCard}>
      <Link href={`/screenalytics/episodes/${episode.ep_id}`} className={styles.episodeLink}>
        <div className={styles.episodeHeader}>
          <span className={styles.episodeNumber}>E{episode.episode_number}</span>
          {episode.title && <span className={styles.episodeTitle}>{episode.title}</span>}
        </div>
        <div className={styles.episodeId}>{episode.ep_id}</div>
        {episode.air_date && (
          <div className={styles.airDate}>{episode.air_date}</div>
        )}
      </Link>
      <div className={styles.cardActions}>
        {confirmDelete ? (
          <div className={styles.confirmRow}>
            <button
              className={styles.confirmDeleteBtn}
              onClick={() => {
                onDelete(episode.ep_id);
                setConfirmDelete(false);
              }}
            >
              Confirm
            </button>
            <button
              className={styles.cancelBtn}
              onClick={() => setConfirmDelete(false)}
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            className={styles.deleteBtn}
            onClick={() => setConfirmDelete(true)}
            title="Delete episode"
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
}

export default function EpisodesListPage() {
  const [search, setSearch] = useState("");
  const [showFilter, setShowFilter] = useState<string>("all");
  const [expandedShows, setExpandedShows] = useState<Set<string>>(new Set());

  const episodesQuery = useEpisodes();
  const deleteEpisode = useDeleteEpisode();

  const episodes = episodesQuery.data || [];

  // Get unique shows for filter dropdown
  const shows = useMemo(() => {
    const uniqueShows = new Set(episodes.map((ep) => ep.show_slug || "Unknown"));
    return Array.from(uniqueShows).sort();
  }, [episodes]);

  // Filter episodes
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
          ep.show_slug?.toLowerCase().includes(q)
      );
    }

    return result;
  }, [episodes, showFilter, search]);

  // Group by show/season
  const grouped = useMemo(() => groupEpisodes(filteredEpisodes), [filteredEpisodes]);

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

  const handleDelete = (epId: string) => {
    deleteEpisode.mutate(epId);
  };

  // Auto-expand all shows initially
  useEffect(() => {
    if (shows.length > 0 && expandedShows.size === 0) {
      setExpandedShows(new Set(shows));
    }
  }, [shows, expandedShows.size]);

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>Episodes</h1>
          <p className={styles.subtitle}>
            {episodes.length} episodes across {shows.length} shows
          </p>
        </div>
        <Link href="/screenalytics/upload" className={styles.uploadBtn}>
          + Upload Episode
        </Link>
      </div>

      <div className={styles.filters}>
        <input
          type="text"
          placeholder="Search episodes..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className={styles.searchInput}
        />
        <select
          value={showFilter}
          onChange={(e) => setShowFilter(e.target.value)}
          className={styles.showSelect}
        >
          <option value="all">All Shows</option>
          {shows.map((show) => (
            <option key={show} value={show}>
              {show}
            </option>
          ))}
        </select>
      </div>

      {episodesQuery.isLoading && (
        <div className={styles.loading}>Loading episodes...</div>
      )}

      {episodesQuery.isError && (
        <div className={styles.error}>
          Failed to load episodes: {episodesQuery.error?.message || "Unknown error"}
        </div>
      )}

      {!episodesQuery.isLoading && filteredEpisodes.length === 0 && (
        <div className={styles.empty}>
          {search || showFilter !== "all"
            ? "No episodes match your filters."
            : "No episodes yet. Upload one to get started!"}
        </div>
      )}

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
                <span className={styles.chevron}>
                  {expandedShows.has(show) ? "▼" : "▶"}
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
                          {grouped[show][season].map((ep) => (
                            <EpisodeCard
                              key={ep.ep_id}
                              episode={ep}
                              onDelete={handleDelete}
                            />
                          ))}
                        </div>
                      </div>
                    ))}
                </div>
              )}
            </div>
          ))}
      </div>
    </div>
  );
}
