// Archived youth-league pages – not part of the Screenalytics core UI.
"use client";

import { useEffect, useMemo, useState } from "react";

type Division = {
  id: string;
  name: string;
  eventId: string;
  requestsCount?: number;
};

type AgentStatusResponse = {
  status: string;
  lastRun?: {
    id: string;
    status: string;
    startedAt: string;
    completedAt?: string | null;
    errorMessage?: string | null;
  };
};

type RequestRow = {
  id: string;
  playerId: string;
  playerName: string;
  rawText: string;
  status: string;
  candidatesCount: number;
  autoAcceptedCount?: number;
  uncertainCount?: number;
  rejectedCount?: number;
  topCandidate?: {
    targetType: string;
    targetName: string | null;
    finalConfidence: number | null;
    autoStatus: string | null;
  } | null;
  hasAcceptedDecision?: boolean;
  hasResolvedDecision?: boolean;
};

export default function RequestAnalysisPage() {
  const [divisions, setDivisions] = useState<Division[]>([]);
  const [selectedDivisionId, setSelectedDivisionId] = useState<string>("");
  const [status, setStatus] = useState<string>("idle");
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [lastRunId, setLastRunId] = useState<string | undefined>();
  const [requests, setRequests] = useState<RequestRow[]>([]);
  const [filter, setFilter] = useState<"all" | "needs-review">("all");
  const [selectedRequestId, setSelectedRequestId] = useState<string | null>(null);
  const [candidateDetail, setCandidateDetail] = useState<any>(null);
  const [loadingDetail, setLoadingDetail] = useState<boolean>(false);
  const [decisionError, setDecisionError] = useState<string | null>(null);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);

  useEffect(() => {
    const loadDivisions = async () => {
      try {
        const res = await fetch("/api/divisions");
        if (!res.ok) {
          throw new Error("Failed to load divisions");
        }
        const payload = await res.json();
        const data = Array.isArray(payload) ? payload : payload.divisions || [];
        setDivisions(data);
        if (data.length > 0) {
          setSelectedDivisionId((prev) => prev || data[0].id);
        }
      } catch (err) {
        console.error(err);
        setError("Unable to load divisions");
      }
    };
    loadDivisions();
  }, []);

  useEffect(() => {
    const loadRequests = async () => {
      if (!selectedDivisionId) return;
      try {
        const res = await fetch(`/api/divisions/${selectedDivisionId}/requests`);
        if (!res.ok) {
          throw new Error("Failed to load requests");
        }
        const body = await res.json();
        setRequests(body.requests || []);
      } catch (err) {
        console.error(err);
        setError("Unable to load requests");
      }
    };
    loadRequests();
  }, [selectedDivisionId]);

  useEffect(() => {
    if (!selectedRequestId) return;
    const loadDetail = async () => {
      setLoadingDetail(true);
      setDecisionError(null);
      try {
        const res = await fetch(`/api/requests/${selectedRequestId}/candidates`);
        if (!res.ok) throw new Error("Failed to load candidates");
        const body = await res.json();
        setCandidateDetail(body);
        setSelectedCandidateId(null);
      } catch (err) {
        console.error(err);
        setDecisionError("Unable to load candidates");
      } finally {
        setLoadingDetail(false);
      }
    };
    loadDetail();
  }, [selectedRequestId]);

  const selectedDivision = useMemo(
    () => divisions.find((div) => div.id === selectedDivisionId),
    [divisions, selectedDivisionId],
  );

  const refreshStatus = async (runId?: string) => {
    if (!selectedDivisionId) return undefined;
    try {
      const search = new URLSearchParams();
      search.set("divisionId", selectedDivisionId);
      if (runId) search.set("runId", runId);
      const res = await fetch(`/api/agent-status?${search.toString()}`);
      if (!res.ok) {
        throw new Error("Failed to fetch agent status");
      }
      const body: AgentStatusResponse = await res.json();
      setStatus(body.status);
      setError(body.lastRun?.errorMessage ?? null);
      if (body.lastRun?.id) {
        setLastRunId(body.lastRun.id);
      }
      return body.lastRun?.status ?? body.status;
    } catch (err) {
      console.error(err);
      setError("Error while checking agent status");
      return undefined;
    }
  };

  const fetchRequests = async () => {
    if (!selectedDivisionId) return;
    try {
      const res = await fetch(`/api/divisions/${selectedDivisionId}/requests`);
      if (!res.ok) {
        throw new Error("Failed to load requests");
      }
      const body = await res.json();
      setRequests(body.requests || []);
    } catch (err) {
      console.error(err);
      setError("Unable to load requests");
    }
  };

  const pollUntilComplete = async (runId: string) => {
    let attempts = 0;
    const maxAttempts = 10;
    const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
    let currentStatus: string | undefined;
    while (attempts < maxAttempts) {
      await delay(500);
      currentStatus = await refreshStatus(runId);
      attempts += 1;
      if (currentStatus && !["started", "running"].includes(currentStatus)) {
        break;
      }
    }
    setIsRunning(false);
    await fetchRequests();
  };

  const handleSaveDecision = async () => {
    if (!selectedRequestId) return;
    setDecisionError(null);
    try {
      const res = await fetch(`/api/requests/${selectedRequestId}/decision`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selectedCandidateId }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || "Failed to save decision");
      }
      setSelectedRequestId(null);
      setCandidateDetail(null);
      await fetchRequests();
    } catch (err) {
      console.error(err);
      setDecisionError((err as Error).message);
    }
  };

  const handleRun = async () => {
    if (!selectedDivisionId) return;
    setIsRunning(true);
    setError(null);
    try {
      const res = await fetch("/api/agent-run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ divisionId: selectedDivisionId, runType: "request-parsing" }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || "Failed to start agent run");
      }
      const body = await res.json();
      setLastRunId(body.runId);
      setStatus(body.status || "started");
      await pollUntilComplete(body.runId);
    } catch (err) {
      console.error(err);
      setError("Unable to start agent run");
      setIsRunning(false);
    }
  };

  return (
    <div className="card">
      <h1>Request Analysis</h1>
      <p style={{ color: "#475569" }}>
        Kick off the request parsing agent for a division. Status is stubbed for now but wired end-to-end.
      </p>

      <div style={{ marginTop: 12, display: "flex", gap: 12, alignItems: "center" }}>
        <label htmlFor="division-select" style={{ fontWeight: 600 }}>
          Division
        </label>
        <select
          id="division-select"
          value={selectedDivisionId}
          onChange={(e) => setSelectedDivisionId(e.target.value)}
          style={{ padding: "8px 10px", borderRadius: 6, border: "1px solid #cbd5e1" }}
        >
          {divisions.map((division) => (
            <option key={division.id} value={division.id}>
              {division.name}
            </option>
          ))}
        </select>
        <button onClick={() => refreshStatus(lastRunId)} disabled={!selectedDivisionId || isRunning}>
          Refresh status
        </button>
        <button onClick={handleRun} disabled={!selectedDivisionId || isRunning}>
          {isRunning ? "Running..." : "Run/Refresh Request Analysis"}
        </button>
        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          Filter:
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as "all" | "needs-review")}
            style={{ padding: "6px 8px", borderRadius: 6, border: "1px solid #cbd5e1" }}
          >
            <option value="all">All</option>
            <option value="needs-review">Needs review</option>
          </select>
        </label>
      </div>

      <div style={{ marginTop: 16 }}>
        <p>
          <strong>Status:</strong> {status}
        </p>
        {selectedDivision && (
          <p style={{ margin: 0, color: "#475569" }}>
            Division: {selectedDivision.name} (Event ID: {selectedDivision.eventId})
          </p>
        )}
        {lastRunId && (
          <p style={{ margin: 0, color: "#475569" }}>
            Run ID: <code>{lastRunId}</code>
          </p>
        )}
        {error && (
          <p style={{ color: "#b91c1c" }}>
            <strong>Error:</strong> {error}
          </p>
        )}
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Requests</h3>
        {requests.length === 0 ? (
          <p style={{ color: "#475569" }}>No requests found for this division.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Player</th>
                <th>Request</th>
                <th>Status</th>
                <th>Candidates</th>
                <th>Top candidate</th>
                <th>Review</th>
              </tr>
            </thead>
            <tbody>
              {requests
                .filter((req) =>
                  filter === "needs-review"
                    ? !req.hasResolvedDecision && ((req.autoAcceptedCount ?? 0) > 0 || (req.uncertainCount ?? 0) > 0)
                    : true,
                )
                .map((req) => {
                  const derivedStatus = req.hasResolvedDecision
                    ? "Resolved"
                    : (req.autoAcceptedCount ?? 0) > 0
                      ? "Auto-accepted (pending review)"
                      : (req.uncertainCount ?? 0) > 0
                        ? "Uncertain"
                        : req.status;
                  return (
                    <tr
                      key={req.id}
                      style={{ cursor: "pointer" }}
                      onClick={() => setSelectedRequestId(req.id)}
                    >
                      <td>{req.playerName}</td>
                      <td>{req.rawText}</td>
                      <td>{derivedStatus}</td>
                      <td>
                        {req.autoAcceptedCount ?? 0} auto / {req.uncertainCount ?? 0} uncertain /{" "}
                        {req.rejectedCount ?? 0} rejected
                      </td>
                      <td>
                        {req.topCandidate ? (
                          <>
                            {req.topCandidate.targetName ?? "(unresolved)"} · {req.topCandidate.autoStatus ?? "?"} ·{" "}
                            {(req.topCandidate.finalConfidence ?? 0).toFixed(2)}
                          </>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td>{req.hasResolvedDecision ? "Resolved" : "Needs review"}</td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        )}
      </div>

      {selectedRequestId && (
        <div className="card" style={{ marginTop: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <h3>Request detail</h3>
            <button onClick={() => setSelectedRequestId(null)}>Close</button>
          </div>
          {loadingDetail ? (
            <p>Loading…</p>
          ) : candidateDetail ? (
            <>
              <p>
                <strong>Player:</strong> {candidateDetail.player.firstName} {candidateDetail.player.lastName}
              </p>
              <p>
                <strong>Request:</strong> {candidateDetail.rawText}
              </p>
              <div>
                <label style={{ display: "block", marginBottom: 8 }}>
                  <input
                    type="radio"
                    name="candidate"
                    value=""
                    checked={selectedCandidateId === null}
                    onChange={() => setSelectedCandidateId(null)}
                  />{" "}
                  No match / deny all
                </label>
                {candidateDetail.candidates.map((cand: any) => (
                  <label key={cand.id} style={{ display: "block", marginBottom: 8 }}>
                    <input
                      type="radio"
                      name="candidate"
                      value={cand.id}
                      checked={selectedCandidateId === cand.id}
                      onChange={() => setSelectedCandidateId(cand.id)}
                    />{" "}
                    {cand.targetType}:{" "}
                    {cand.targetPlayer?.name || cand.targetCoach?.name || "(unresolved)"} · conf{" "}
                    {(cand.finalConfidence ?? 0).toFixed(2)} · auto {cand.autoStatus ?? "-"} · decision{" "}
                    {cand.adminDecision ?? "-"}
                  </label>
                ))}
              </div>
              {decisionError && <p style={{ color: "#b91c1c" }}>{decisionError}</p>}
              <button onClick={handleSaveDecision}>Save decision</button>
            </>
          ) : (
            <p>No detail.</p>
          )}
        </div>
      )}
    </div>
  );
}
