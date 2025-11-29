import { NextResponse } from "next/server";
import { prisma } from "@/src/lib/prisma";

export async function GET(_request: Request, { params }: { params: { divisionId: string } }) {
  const { divisionId } = params;
  try {
    const division = await prisma.division.findUnique({
      where: { id: divisionId },
      select: {
        id: true,
        name: true,
        eventId: true,
        eventPrice: true,
        gender: true,
        divisionConfig: true,
        _count: { select: { teams: true } },
      },
    });
    if (!division) {
      return NextResponse.json({ error: "Division not found" }, { status: 404 });
    }
    const [playerCount, coachCount, requestCount, mismatchedCount] = await Promise.all([
      prisma.player.count({ where: { divisionId } }),
      prisma.coach.count({ where: { divisionId } }),
      prisma.request.count({ where: { divisionId } }),
      prisma.player.count({ where: { divisionId, mismatchedGender: true } }),
    ]);
    const statuses = division._count?.teams && division._count.teams > 0 ? "Teams generated" : "Setup in progress";
    const status = division.divisionConfig ? statuses : "Not started";
    return NextResponse.json({
      id: division.id,
      name: division.name,
      eventId: division.eventId,
      eventPrice: division.eventPrice,
      gender: division.gender,
      playerCount,
      coachCount,
      requestCount,
      mismatchedGenderCount: mismatchedCount,
      status,
    });
  } catch (error) {
    console.error("GET /api/divisions/[divisionId] failed", error);
    return NextResponse.json({ error: "Unable to fetch division" }, { status: 500 });
  }
}
