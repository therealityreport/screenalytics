import Link from "next/link";
import { notFound } from "next/navigation";
import { prisma } from "@/src/lib/prisma";
import { CsvUploadForm } from "./CsvUploadForm";

type PageProps = {
  params: { eventId: string };
};

async function getEvent(eventId: string) {
  return prisma.event.findUnique({
    where: { id: eventId },
    select: { id: true, name: true, season: true, year: true },
  });
}

async function getDivisionsWithCounts(eventId: string) {
  const divisions = await prisma.division.findMany({
    where: { eventId },
    orderBy: { createdAt: "asc" },
    select: { id: true, name: true, eventPrice: true, gender: true, divisionConfig: { select: { id: true } }, _count: { select: { teams: true } } },
  });
  const divisionIds = divisions.map((d) => d.id);
  const playerCounts = await prisma.player.groupBy({
    by: ["divisionId"],
    _count: { _all: true },
    where: { divisionId: { in: divisionIds } },
  });
  const coachCounts = await prisma.coach.groupBy({
    by: ["divisionId"],
    _count: { _all: true },
    where: { divisionId: { in: divisionIds } },
  });
  const mismatchedCounts = await prisma.player.groupBy({
    by: ["divisionId"],
    _count: { _all: true },
    where: { divisionId: { in: divisionIds }, mismatchedGender: true },
  });
  const requestCounts = await prisma.request.groupBy({
    by: ["divisionId"],
    _count: { _all: true },
    where: { divisionId: { in: divisionIds } },
  });
  const playerCountMap = new Map(playerCounts.map((c) => [c.divisionId, c._count._all]));
  const coachCountMap = new Map(coachCounts.map((c) => [c.divisionId, c._count._all]));
  const mismatchedMap = new Map(mismatchedCounts.map((c) => [c.divisionId, c._count._all]));
  const requestCountMap = new Map(requestCounts.map((c) => [c.divisionId, c._count._all]));
  return divisions.map((division) => ({
    ...division,
    playerCount: playerCountMap.get(division.id) ?? 0,
    coachCount: coachCountMap.get(division.id) ?? 0,
    mismatchedGenderCount: mismatchedMap.get(division.id) ?? 0,
    requestsCount: requestCountMap.get(division.id) ?? 0,
    status: division.divisionConfig
      ? division._count?.teams && division._count.teams > 0
        ? "Teams generated"
        : "Setup in progress"
      : "Not started",
  }));
}

export default async function EventPage({ params }: PageProps) {
  const event = await getEvent(params.eventId);
  if (!event) {
    notFound();
  }
  const divisions = await getDivisionsWithCounts(params.eventId);

  return (
    <div className="card">
      <Link href="/">← Back to Events</Link>
      <h1 style={{ marginTop: 8 }}>{event.name}</h1>
      <p style={{ color: "#475569" }}>
        Season: {event.season ? `${event.season} ${event.year ?? ""}` : "Not set"}
      </p>
      <p style={{ color: "#475569" }}>Division dashboard</p>
      {divisions.length === 0 ? (
        <p style={{ marginTop: 12 }}>No divisions yet for this event. Upload a CSV to create them.</p>
      ) : (
        <table style={{ marginTop: 16 }}>
          <thead>
            <tr>
              <th>Name</th>
              <th>Event Price</th>
              <th>Players</th>
              <th>Coaches</th>
              <th>Mismatched gender</th>
              <th>Requests</th>
              <th>Status</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {divisions.map((division) => {
              return (
                <tr key={division.id}>
                  <td>{division.name}</td>
                  <td>{division.eventPrice ?? "—"}</td>
                  <td>{division.playerCount}</td>
                  <td>{division.coachCount}</td>
                  <td>{division.mismatchedGenderCount}</td>
                  <td>{division.requestsCount ?? 0}</td>
                  <td>{division.status}</td>
                  <td>
                    <Link href={`/divisions/${division.id}`}>Configure</Link>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <CsvUploadForm eventId={params.eventId} />
    </div>
  );
}
