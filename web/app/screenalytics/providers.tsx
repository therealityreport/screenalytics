"use client";

import { ReactNode, useEffect, useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { startMockWorker } from "@/mocks/browser";
import { ToastProvider } from "@/components/toast";
import { DocsProvider } from "@/components/screenalytics/docs-provider";

export function ScreenalyticsProviders({ children }: { children: ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            retry: false,
            refetchOnWindowFocus: true,
          },
          mutations: {
            retry: false,
          },
        },
      }),
  );

  useEffect(() => {
    if (process.env.NEXT_PUBLIC_MSW === "1") {
      startMockWorker();
    }
  }, []);

  return (
    <ToastProvider>
      <QueryClientProvider client={client}>
        <DocsProvider>
          {children}
        </DocsProvider>
        <ReactQueryDevtools initialIsOpen={false} />
      </QueryClientProvider>
    </ToastProvider>
  );
}
