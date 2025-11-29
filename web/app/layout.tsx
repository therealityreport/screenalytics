import "./globals.css";
import Link from "next/link";
import { ReactNode } from "react";
import styles from "./layout.module.css";

export const metadata = {
  title: "Screenalytics",
  description: "Screenalytics workspace UI for uploads, pipeline status, and faces review",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className={styles.header}>
          <div>
            <div className={styles.brand}>Screenalytics</div>
            <div className={styles.subdued}>Next.js workspace UI</div>
          </div>
          <Link href="/screenalytics/upload" className={styles.subdued}>
            Go to workspace
          </Link>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
