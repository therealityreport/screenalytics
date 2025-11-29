"use client";

import * as Toast from "@radix-ui/react-toast";
import { createContext, useCallback, useContext, useMemo, useState } from "react";
import styles from "./toast.module.css";

type ToastItem = {
  id: number;
  title: string;
  description?: string;
  variant?: "default" | "error";
};

type ToastContextShape = {
  notify: (toast: Omit<ToastItem, "id">) => void;
};

const ToastContext = createContext<ToastContextShape>({
  notify: () => undefined,
});

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);

  const notify = useCallback((toast: Omit<ToastItem, "id">) => {
    setItems((prev) => [...prev, { ...toast, id: Date.now() }]);
  }, []);

  const value = useMemo(() => ({ notify }), [notify]);

  return (
    <ToastContext.Provider value={value}>
      <Toast.Provider swipeDirection="right" duration={4000}>
        {children}
        {items.map((item) => (
          <Toast.Root
            key={item.id}
            className={`${styles.toastRoot} ${item.variant === "error" ? styles.error : ""}`}
            onOpenChange={(open) => {
              if (!open) {
                setItems((prev) => prev.filter((p) => p.id !== item.id));
              }
            }}
            open
          >
            <Toast.Title className={styles.title}>{item.title}</Toast.Title>
            {item.description ? <Toast.Description className={styles.desc}>{item.description}</Toast.Description> : null}
          </Toast.Root>
        ))}
        <Toast.Viewport className={styles.toastViewport} />
      </Toast.Provider>
    </ToastContext.Provider>
  );
}

export function useToast() {
  return useContext(ToastContext);
}
