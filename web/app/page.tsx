import Link from "next/link";
import styles from "./home.module.css";

export default function ScreenalyticsHome() {
  return (
    <div className="card">
      <div className={styles.hero}>
        <div>
          <p className={styles.kicker}>Screenalytics</p>
          <h1 className={styles.title}>Next.js workspace UI</h1>
          <p className={styles.body}>
            Manage uploads, monitor pipeline phases, and review faces in the new Screenalytics experience. Streamlit remains
            available while the Next.js UI reaches full parity.
          </p>
          <div className={styles.actions}>
            <Link href="/screenalytics/upload" className={styles.primary}>
              Go to Upload
            </Link>
            <Link href="/screenalytics/episodes/demo" className={styles.secondary}>
              View Episode Detail
            </Link>
          </div>
        </div>
        <div className={styles.callout}>
          <div className={styles.badge}>Beta</div>
          <div className={styles.calloutText}>
            Need legacy youth-league tools? They&apos;re archived under <code>/_archive-youth-league</code>.
          </div>
        </div>
      </div>
    </div>
  );
}
