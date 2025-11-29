import type { AssetUploadResponse } from "./types";

export type UploadProgress = {
  loaded: number;
  total: number;
  percent: number;
  speedBps?: number;
};

export type UploadOptions = {
  signal?: AbortSignal;
  onProgress?: (progress: UploadProgress) => void;
};

function noop() {}

/**
 * Browser fetch lacks native upload progress; we simulate progress ticks while the request is in-flight
 * and complete with a 100% update once the request resolves.
 */
export async function uploadFileWithProgress(
  presign: AssetUploadResponse,
  file: File,
  opts?: UploadOptions,
): Promise<Response> {
  const { upload_url: url, method, headers } = presign as AssetUploadResponse & {
    fields?: Record<string, string>;
  };
  const fields = (presign as { fields?: Record<string, string> }).fields;
  if (!url) {
    throw new Error("Missing upload URL");
  }
  const resolvedMethod = method?.toUpperCase() || "PUT";
  const onProgress = opts?.onProgress ?? noop;
  const total = file.size;
  let loaded = 0;
  let lastTick = Date.now();

  const timer = setInterval(() => {
    const now = Date.now();
    const elapsed = now - lastTick;
    lastTick = now;
    // drift upward until 90% to show liveness
    loaded = Math.min(total * 0.9, loaded + (total * elapsed) / 8000);
    const speedBps = elapsed > 0 ? ((loaded / elapsed) * 1000) : undefined;
    onProgress({ loaded, total, percent: Math.min(loaded / total, 0.9), speedBps });
  }, 350);

  try {
    if (resolvedMethod === "POST") {
      const formData = new FormData();
      if (fields) {
        Object.entries(fields).forEach(([key, value]) => formData.append(key, value));
      }
      formData.append("file", file);
      const response = await fetch(url, {
        method: "POST",
        body: formData,
        signal: opts?.signal,
      });
      loaded = total;
      onProgress({ loaded, total, percent: 1 });
      return response;
    }

    const response = await fetch(url, {
      method: resolvedMethod,
      body: file,
      signal: opts?.signal,
      headers,
    });
    loaded = total;
    onProgress({ loaded, total, percent: 1 });
    return response;
  } finally {
    clearInterval(timer);
  }
}
