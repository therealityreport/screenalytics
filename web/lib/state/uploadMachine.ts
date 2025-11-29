import type { ApiError } from "@/api/types";

export type UploadStep =
  | "idle"
  | "preparing"
  | "ready"
  | "uploading"
  | "verifying"
  | "processing"
  | "canceled"
  | "success"
  | "error";

export type UploadMode = "new" | "replace";

export type FileInfo = {
  name: string;
  size: number;
  type?: string;
};

export type UploadFlags = {
  tracks_only_fallback?: boolean;
  faces_manifest_fallback?: boolean;
};

export type UploadState = {
  step: UploadStep;
  mode: UploadMode;
  episodeId?: string;
  file?: FileInfo;
  progress?: number;
  speedBps?: number;
  flags?: UploadFlags;
  jobId?: string;
  error?: ApiError;
  lastMessage?: string;
};

export type UploadAction =
  | { type: "SET_FILE"; file?: FileInfo }
  | { type: "SET_MODE"; mode: UploadMode; episodeId?: string }
  | { type: "SET_STEP"; step: UploadStep; message?: string; flags?: UploadFlags; jobId?: string }
  | { type: "SET_PROGRESS"; progress: number; speedBps?: number }
  | { type: "ERROR"; error: ApiError }
  | { type: "CANCEL" }
  | { type: "RESET" };

export function createInitialState(episodeId?: string): UploadState {
  return {
    step: episodeId ? "ready" : "idle",
    mode: episodeId ? "replace" : "new",
    episodeId,
  };
}

export function uploadReducer(state: UploadState, action: UploadAction): UploadState {
  switch (action.type) {
    case "SET_FILE":
      return { ...state, file: action.file, step: action.file ? "ready" : "idle" };
    case "SET_MODE":
      return { ...state, mode: action.mode, episodeId: action.episodeId };
    case "SET_STEP":
      return {
        ...state,
        step: action.step,
        lastMessage: action.message ?? state.lastMessage,
        flags: action.flags ?? state.flags,
        jobId: action.jobId ?? state.jobId,
        error: action.step === "error" ? state.error : undefined,
      };
    case "SET_PROGRESS":
      return { ...state, progress: action.progress, speedBps: action.speedBps };
    case "ERROR":
      return { ...state, step: "error", error: action.error };
    case "CANCEL":
      return { ...state, step: "canceled" };
    case "RESET":
      return createInitialState(state.mode === "replace" ? state.episodeId : undefined);
    default:
      return state;
  }
}
