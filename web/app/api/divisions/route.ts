import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/src/lib/prisma";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const eventId = searchParams.get("eventId");
  try {
    const divisions = await prisma.division.findMany({
      where: eventId ? { eventId } : undefined,
      orderBy: { createdAt: "asc" },
      select: {
        id: true,
        name: true,
        eventId: true,
        eventPrice: true,
        gender: true,
        createdAt: true,
        divisionConfig: { select: { id: true } },
        _count: { select: { teams: true } },
      },
    });
    const divisionIds = divisions.map((d) => d.id);
    const playerCounts = await prisma.player.groupBy({
      by: ["divisionId"],
      _count: { _all: true },
      where: divisionIds.length ? { divisionId: { in: divisionIds } } : undefined,
    });
    const coachCounts = await prisma.coach.groupBy({
      by: ["divisionId"],
      _count: { _all: true },
      where: divisionIds.length ? { divisionId: { in: divisionIds } } : undefined,
    });
    const mismatchedCounts = divisionIds.length
      ? await prisma.player.groupBy({
          by: ["divisionId"],
          _count: { _all: true },
          where: { divisionId: { in: divisionIds }, mismatchedGender: true },
        })
      : [];
    const requestCounts = divisionIds.length
      ? await prisma.request.groupBy({
          by: ["divisionId"],
          _count: { _all: true },
          where: { divisionId: { in: divisionIds } },
        })
      : [];

    const playerCountMap = new Map(playerCounts.map((c) => [c.divisionId, c._count._all]));
    const coachCountMap = new Map(coachCounts.map((c) => [c.divisionId, c._count._all]));
    const mismatchedMap = new Map(mismatchedCounts.map((c) => [c.divisionId, c._count._all]));
    const requestCountMap = new Map(requestCounts.map((c) => [c.divisionId, c._count._all]));

    const augmented = divisions.map((division) => {
      const playerCount = playerCountMap.get(division.id) ?? 0;
      const coachCount = coachCountMap.get(division.id) ?? 0;
      const mismatchedGenderCount = mismatchedMap.get(division.id) ?? 0;
      const requestsCount = requestCountMap.get(division.id) ?? 0;
      const status = division.divisionConfig
        ? division._count?.teams && division._count.teams > 0
          ? "Teams generated"
          : "Setup in progress"
        : "Not started";
      const { divisionConfig, _count, ...rest } = division;
      return {
        ...rest,
        playerCount,
        coachCount,
        mismatchedGenderCount,
        requestsCount,
        status,
      };
    });
    return NextResponse.json({ divisions: augmented });
  } catch (error) {
    console.error("GET /api/divisions failed", error);
    return NextResponse.json({ error: "Unable to fetch divisions" }, { status: 500 });
  }
}
