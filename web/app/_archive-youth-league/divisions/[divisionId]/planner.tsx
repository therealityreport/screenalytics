// Archived youth-league pages – not part of the Screenalytics core UI.
"use client";

import { useEffect, useMemo, useState } from "react";
import { computeTeamOptions, TeamOption } from "@/src/lib/teamPlanning";

type Config = {
  targetPlayersPerTeam: number;
  minPlayersPerTeam: number;
  maxPlayersPerTeam: number;
  teamCount: number | null;
};

function parseIntSafe(value: string, fallback: number) {
  const parsed = parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function DivisionPlanner({
  divisionId,
  initialConfig,
  playerCount,
  initialOptions,
}: {
  divisionId: string;
  initialConfig: Config;
  playerCount: number;
  initialOptions: TeamOption[];
}) {
  const [config, setConfig] = useState<Config>(initialConfig);
  const [selectedTeamCount, setSelectedTeamCount] = useState<number | null>(initialConfig.teamCount);
  const [options, setOptions] = useState<TeamOption[]>(initialOptions);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const next = computeTeamOptions({
      playerCount,
      targetPlayersPerTeam: config.targetPlayersPerTeam,
      minPlayersPerTeam: config.minPlayersPerTeam,
      maxPlayersPerTeam: config.maxPlayersPerTeam,
    });
    setOptions(next);
  }, [config, playerCount]);

  const validSelection = useMemo(() => options.some((opt) => opt.teamCount === selectedTeamCount), [options, selectedTeamCount]);

  const handleSave = async () => {
    setError(null);
    setMessage(null);
    if (!selectedTeamCount || !validSelection) {
      setError("Select a valid team option before saving.");
      return;
    }
    setSaving(true);
    try {
      const res = await fetch(`/api/divisions/${divisionId}/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          targetPlayersPerTeam: config.targetPlayersPerTeam,
          minPlayersPerTeam: config.minPlayersPerTeam,
          maxPlayersPerTeam: config.maxPlayersPerTeam,
          teamCount: selectedTeamCount,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || "Failed to save configuration");
      }
      setMessage("Configuration saved.");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ marginTop: 16 }}>
      <h3>Team size planning</h3>
      <p style={{ color: "#475569" }}>
        Player count: {playerCount}. Adjust targets to explore possible team distributions.
      </p>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        <label>
          Target players/team
          <input
            type="number"
            min={1}
            value={config.targetPlayersPerTeam}
            onChange={(e) =>
              setConfig((cfg) => ({ ...cfg, targetPlayersPerTeam: parseIntSafe(e.target.value, cfg.targetPlayersPerTeam) }))
            }
            style={{ display: "block", width: 120, padding: 6, marginTop: 4 }}
          />
        </label>
        <label>
          Min players/team
          <input
            type="number"
            min={1}
            value={config.minPlayersPerTeam}
            onChange={(e) =>
              setConfig((cfg) => ({ ...cfg, minPlayersPerTeam: parseIntSafe(e.target.value, cfg.minPlayersPerTeam) }))
            }
            style={{ display: "block", width: 120, padding: 6, marginTop: 4 }}
          />
        </label>
        <label>
          Max players/team
          <input
            type="number"
            min={1}
            value={config.maxPlayersPerTeam}
            onChange={(e) =>
              setConfig((cfg) => ({ ...cfg, maxPlayersPerTeam: parseIntSafe(e.target.value, cfg.maxPlayersPerTeam) }))
            }
            style={{ display: "block", width: 120, padding: 6, marginTop: 4 }}
          />
        </label>
      </div>
      <div style={{ marginTop: 16 }}>
        <h4>Options</h4>
        {options.length === 0 ? (
          <p style={{ color: "#b91c1c" }}>No valid team splits for these constraints.</p>
        ) : (
          <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
            {options.map((opt) => (
              <li key={opt.teamCount} style={{ marginBottom: 8 }}>
                <label style={{ cursor: "pointer" }}>
                  <input
                    type="radio"
                    name="teamOption"
                    value={opt.teamCount}
                    checked={selectedTeamCount === opt.teamCount}
                    onChange={() => setSelectedTeamCount(opt.teamCount)}
                    style={{ marginRight: 8 }}
                  />
                  {opt.teamCount} teams:{" "}
                  {opt.distribution.map((d) => `${d.teams} with ${d.size}`).join(", ")} · Extra to reach perfect multiple:{" "}
                  {opt.extraPlayersToPerfectMultiple}
                </label>
              </li>
            ))}
          </ul>
        )}
      </div>
      {error && <p style={{ color: "#b91c1c" }}>{error}</p>}
      {message && <p style={{ color: "#16a34a" }}>{message}</p>}
      <button onClick={handleSave} disabled={saving || options.length === 0}>
        {saving ? "Saving…" : "Save configuration"}
      </button>
    </div>
  );
}
