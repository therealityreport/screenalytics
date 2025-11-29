import { setupWorker } from "msw";
import { handlers } from "./handlers";

declare global {
  interface Window {
    __mswReady?: boolean;
  }
}

export async function startMockWorker() {
  if (typeof window === "undefined") return;
  if (window.__mswReady) return;
  const worker = setupWorker(...handlers);
  await worker.start({ onUnhandledRequest: "bypass" });
  window.__mswReady = true;
}
