import Link from "next/link";

export default function HomePage() {
  return (
    <div className="card">
      <h1 style={{ margin: 0 }}>Screenalytics</h1>
      <p style={{ marginTop: 16 }}>
        Reality TV face tracking and screen time analytics platform.
      </p>
      <div style={{ marginTop: 24, display: "flex", gap: 12 }}>
        <Link href="/screenalytics/upload" style={{ padding: "12px 20px", background: "#0f172a", color: "#fff", borderRadius: 8, textDecoration: "none", fontWeight: 600 }}>
          Upload Episode
        </Link>
        <Link href="/screenalytics/episodes" style={{ padding: "12px 20px", border: "1px solid #cbd5e1", borderRadius: 8, textDecoration: "none", fontWeight: 600 }}>
          Browse Episodes
        </Link>
      </div>
    </div>
  );
}
