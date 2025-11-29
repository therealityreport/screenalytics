// Archived youth-league pages – not part of the Screenalytics core UI.
import Link from "next/link";
import { notFound } from "next/navigation";
import { prisma } from "@/src/lib/prisma";
import { computeTeamOptions } from "@/src/lib/teamPlanning";
import { DivisionPlanner } from "./planner";

type PageProps = {
  params: { divisionId: string };
};

const DEFAULT_CFG = {
  targetPlayersPerTeam: 10,
  minPlayersPerTeam: 9,
  maxPlayersPerTeam: 11,
  teamCount: null as number | null,
};

async function loadDivision(divisionId: string) {
  const division = await prisma.division.findUnique({
    where: { id: divisionId },
    include: { divisionConfig: true },
  });
  if (!division) return null;
  const [playerCount, coachCount, requestCount] = await Promise.all([
    prisma.player.count({ where: { divisionId } }),
    prisma.coach.count({ where: { divisionId } }),
    prisma.request.count({ where: { divisionId } }),
  ]);
  return { division, playerCount, coachCount, requestCount };
}

export default async function DivisionPage({ params }: PageProps) {
  const data = await loadDivision(params.divisionId);
  if (!data) {
    notFound();
  }
  const { division, playerCount, coachCount, requestCount } = data;
  const cfg = division.divisionConfig
    ? {
        targetPlayersPerTeam: division.divisionConfig.targetPlayersPerTeam,
        minPlayersPerTeam: division.divisionConfig.minPlayersPerTeam,
        maxPlayersPerTeam: division.divisionConfig.maxPlayersPerTeam,
        teamCount: division.divisionConfig.teamCount,
      }
    : DEFAULT_CFG;
  const options = computeTeamOptions({
    playerCount,
    targetPlayersPerTeam: cfg.targetPlayersPerTeam,
    minPlayersPerTeam: cfg.minPlayersPerTeam,
    maxPlayersPerTeam: cfg.maxPlayersPerTeam,
  });

  return (
    <div className="card">
      <Link href={`/_archive-youth-league/events/${division.eventId}`}>← Back to Event</Link>
      <h1 style={{ marginTop: 8 }}>{division.name}</h1>
      <p style={{ color: "#475569" }}>
        Players: {playerCount} · Coaches: {coachCount} · Requests: {requestCount}
      </p>
      <DivisionPlanner
        divisionId={division.id}
        initialConfig={cfg}
        playerCount={playerCount}
        initialOptions={options}
      />
    </div>
  );
}
