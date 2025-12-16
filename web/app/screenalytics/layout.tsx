import Link from "next/link";
import type { ReactNode } from "react";
import { ScreenalyticsProviders } from "./providers";
import styles from "./layout.module.css";

export const metadata = {
  title: "Screenalytics",
  description: "Next.js workspace UI for Screenalytics",
};

export default function ScreenalyticsLayout({ children }: { children: ReactNode }) {
  return (
    <ScreenalyticsProviders>
      <div className={styles.shell}>
        <aside className={styles.sidebar}>
          <div className={styles.brand}>Screenalytics</div>
          <div className={styles.nav}>
            <Link className={styles.navLink} href="/screenalytics">
              Dashboard
            </Link>
            <Link className={styles.navLink} href="/screenalytics/upload">
              Upload
            </Link>
            <Link className={styles.navLink} href="/screenalytics/episodes/demo">
              Episode Detail
            </Link>
            <Link className={styles.navLink} href="/screenalytics/faces">
              Faces Review
            </Link>
          </div>
        </aside>
        <div className={styles.content}>
          <div className={styles.header}>
            <div>
              <div className={styles.breadcrumb}>Streamlit → Next.js (Phase 1)</div>
              <h1 style={{ margin: "4px 0 0", fontSize: 24, fontWeight: 700 }}>Workspace UI</h1>
            </div>
            <span className={styles.badge}>CSS Modules</span>
          </div>
          {children}
        </div>
      </div>
    </ScreenalyticsProviders>
  );
}
