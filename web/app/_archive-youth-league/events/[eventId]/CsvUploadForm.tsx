// Archived youth-league pages – not part of the Screenalytics core UI.
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

type UploadSummary = {
  eventId: string;
  totalRows: number;
  divisions: Array<{
    divisionId: string;
    name: string;
    playerCount: number;
    coachCount: number;
    mismatchedGenderCount: number;
  }>;
};

export function CsvUploadForm({ eventId }: { eventId: string }) {
  const [file, setFile] = useState<File | null>(null);
  const [season, setSeason] = useState<string>("");
  const [year, setYear] = useState<string>("");
  const [status, setStatus] = useState<"idle" | "uploading" | "error" | "success">("idle");
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<UploadSummary | null>(null);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !season || !year) {
      setError("Please choose a CSV file, season, and year.");
      return;
    }
    setStatus("uploading");
    setError(null);
    setSummary(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("season", season);
      formData.append("year", year);
      const res = await fetch(`/api/events/${eventId}/upload-csv`, {
        method: "POST",
        body: formData,
      });
      const body = await res.json();
      if (!res.ok) {
        const missing = Array.isArray(body.missingColumns) ? body.missingColumns.join(", ") : "";
        throw new Error(body.error || `Upload failed${missing ? `: ${missing}` : ""}`);
      }
      setSummary(body as UploadSummary);
      setStatus("success");
      router.refresh();
    } catch (err: any) {
      setError(err?.message || "Upload failed");
      setStatus("error");
    }
  };

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <h3>Upload CSV</h3>
      <form onSubmit={handleSubmit} style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <label>
          Season
          <select
            value={season}
            onChange={(e) => setSeason(e.target.value)}
            style={{ display: "block", padding: 6, marginTop: 4 }}
            required
          >
            <option value="">Select…</option>
            <option value="Winter">Winter</option>
            <option value="Spring">Spring</option>
            <option value="Summer">Summer</option>
            <option value="Fall">Fall</option>
            <option value="Pre-Season">Pre-Season</option>
          </select>
        </label>
        <label>
          Year
          <input
            type="number"
            min={2000}
            max={2100}
            value={year}
            onChange={(e) => setYear(e.target.value)}
            style={{ display: "block", padding: 6, marginTop: 4, width: 120 }}
            required
          />
        </label>
        <input
          type="file"
          accept=".csv,text/csv"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          disabled={status === "uploading"}
        />
        <button type="submit" disabled={status === "uploading" || !file || !season || !year}>
          {status === "uploading" ? "Uploading…" : "Upload CSV"}
        </button>
      </form>
      {error && <p style={{ color: "#b91c1c" }}>{error}</p>}
      {summary && (
        <div style={{ marginTop: 12 }}>
          <p>
            Processed <strong>{summary.totalRows}</strong> rows
          </p>
          <table>
            <thead>
              <tr>
                <th>Division</th>
                <th>Players</th>
                <th>Coaches</th>
                <th>Mismatched gender</th>
              </tr>
            </thead>
            <tbody>
              {summary.divisions.map((div) => (
                <tr key={div.divisionId}>
                  <td>{div.name}</td>
                  <td>{div.playerCount}</td>
                  <td>{div.coachCount}</td>
                  <td>{div.mismatchedGenderCount}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
