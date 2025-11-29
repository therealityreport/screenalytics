import { http, HttpResponse, delay } from "msw";
import type { EpisodeStatus } from "@/api/types";

const uploads = new Map<string, { status: EpisodeStatus; startedAt?: number }>();

function buildBaseStatus(episodeId: string): EpisodeStatus {
  return {
    ep_id: episodeId,
    detect_track: { phase: "detect_track", status: "pending", detections: 0, tracks: 0 },
    faces_embed: { phase: "faces_embed", status: "pending", faces: 0 },
    cluster: { phase: "cluster", status: "pending" },
    scenes_ready: false,
    tracks_ready: false,
    faces_harvested: false,
    faces_stale: false,
    cluster_stale: false,
    faces_manifest_fallback: false,
    tracks_only_fallback: false,
    coreml_available: null,
  };
}

function ensureEpisode(episodeId: string) {
  if (!uploads.has(episodeId)) {
    uploads.set(episodeId, { status: buildBaseStatus(episodeId) });
  }
  return uploads.get(episodeId)!;
}

function progressPhases(episodeId: string) {
  const entry = ensureEpisode(episodeId);
  if (!entry.startedAt) return entry.status;

  const elapsed = Date.now() - entry.startedAt;
  const status = entry.status;
  const detect = status.detect_track;

  if (detect?.status === "running" && elapsed > 1200) {
    status.detect_track = {
      ...detect,
      status: "success",
      detections: 1200,
      tracks: 900,
      last_run_at: new Date().toISOString(),
    };
    status.tracks_ready = true;
    status.scenes_ready = true;
  }
  if (status.detect_track?.status === "success" && status.faces_embed?.status === "pending") {
    status.faces_embed = {
      ...status.faces_embed,
      status: "success",
      faces: 400,
      manifest_exists: true,
      last_run_at: new Date().toISOString(),
    };
    status.faces_harvested = true;
  }
  if (status.faces_embed?.status === "success" && status.cluster?.status === "pending") {
    status.cluster = {
      ...status.cluster,
      status: "success",
      manifest_exists: true,
      singleton_fraction_before: 0.35,
      singleton_fraction_after: 0.12,
      last_run_at: new Date().toISOString(),
    };
  }
  return status;
}

function startDetectTrack(episodeId: string) {
  const entry = ensureEpisode(episodeId);
  entry.startedAt = Date.now();
  entry.status.detect_track = { ...entry.status.detect_track, status: "running" };
  return entry.status;
}

export const handlers = [
  http.post("/api/episodes", async ({ request }) => {
    const body = (await request.json()) as { show_slug_or_id?: string; season_number?: number; episode_number?: number };
    const epId = body.show_slug_or_id
      ? `${body.show_slug_or_id}-s${body.season_number ?? "x"}e${body.episode_number ?? "x"}`
      : `ep_${Date.now()}`;
    ensureEpisode(epId);
    return HttpResponse.json({ ep_id: epId });
  }),

  http.post("/api/episodes/:episodeId/assets", async ({ params }) => {
    const episodeId = params.episodeId as string;
    ensureEpisode(episodeId);
    return HttpResponse.json({
      ep_id: episodeId,
      method: "PUT",
      bucket: "mock",
      key: `videos/${episodeId}.mp4`,
      object_key: `videos/${episodeId}.mp4`,
      upload_url: "https://mock-bucket.local/upload",
      expires_in: 900,
      headers: {},
      path: `/tmp/${episodeId}.mp4`,
      local_video_path: `/tmp/${episodeId}.mp4`,
      backend: "mock",
    });
  }),

  http.put("https://mock-bucket.local/upload", async () => {
    await delay(600);
    return HttpResponse.json({ ok: true });
  }),

  http.post("/api/jobs/detect_track", async ({ request }) => {
    const body = (await request.json()) as { ep_id: string };
    const episodeId = body.ep_id;
    startDetectTrack(episodeId);
    return HttpResponse.json({ job_id: `job_detect_track_${Date.now()}`, status: "running", phase: "detect_track" });
  }),

  http.post("/api/jobs/cluster", async ({ request }) => {
    const body = (await request.json()) as { ep_id: string };
    const status = ensureEpisode(body.ep_id).status;
    status.cluster = { ...status.cluster, status: "running" };
    return HttpResponse.json({ job_id: `job_cluster_${Date.now()}`, status: "running", phase: "cluster" });
  }),

  http.get("/api/episodes/:episodeId/status", async ({ params }) => {
    await delay(200);
    const episodeId = params.episodeId as string;
    const status = progressPhases(episodeId);
    return HttpResponse.json(status);
  }),
];
