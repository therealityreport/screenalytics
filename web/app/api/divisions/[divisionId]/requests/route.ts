import { NextResponse } from "next/server";
import { prisma } from "@/src/lib/prisma";

export async function GET(_request: Request, { params }: { params: { divisionId: string } }) {
  const { divisionId } = params;
  try {
    const division = await prisma.division.findUnique({ where: { id: divisionId } });
    if (!division) {
      return NextResponse.json({ error: "Division not found" }, { status: 404 });
    }

    const requests = await prisma.request.findMany({
      where: { divisionId },
      orderBy: { createdAt: "desc" },
      include: {
        player: true,
        candidates: true,
      },
    });

    const playerMap = new Map(
      requests.map((r) => [r.playerId, `${r.player.firstName} ${r.player.lastName}`]),
    );
    const coachIds = Array.from(
      new Set(requests.flatMap((r) => r.candidates.map((c) => c.targetCoachId).filter(Boolean))),
    ) as string[];
    const coaches = await prisma.coach.findMany({
      where: { id: { in: coachIds } },
      select: { id: true, name: true },
    });
    const coachMap = new Map(coaches.map((c) => [c.id, c.name]));

    return NextResponse.json({
      divisionId,
      requests: requests.map((req) => ({
        id: req.id,
        playerId: req.playerId,
        playerName: `${req.player.firstName} ${req.player.lastName}`,
        rawText: req.rawText,
        status: req.status,
        candidatesCount: req.candidates.length,
        autoAcceptedCount: req.candidates.filter((c) => c.autoStatus === "AUTO_ACCEPT").length,
        uncertainCount: req.candidates.filter((c) => c.autoStatus === "UNCERTAIN").length,
        rejectedCount: req.candidates.filter((c) => c.adminDecision === "DENIED" || c.autoStatus === "REJECTED").length,
        hasAcceptedDecision: req.candidates.some((c) => c.adminDecision === "ACCEPTED"),
        hasResolvedDecision: req.candidates.every((c) => c.adminDecision && c.adminDecision !== "PENDING"),
        topCandidate: (() => {
          if (!req.candidates.length) return null;
          const sorted = [...req.candidates].sort(
            (a, b) => (b.finalConfidence ?? 0) - (a.finalConfidence ?? 0),
          );
          const top = sorted[0];
          const targetName =
            top.targetPlayerId && playerMap.get(top.targetPlayerId)
              ? playerMap.get(top.targetPlayerId)
              : top.targetCoachId && coachMap.get(top.targetCoachId)
                ? coachMap.get(top.targetCoachId)
                : null;
          return {
            targetType: top.targetType,
            targetName,
            finalConfidence: top.finalConfidence,
            autoStatus: top.autoStatus,
          };
        })(),
      })),
    });
  } catch (error) {
    console.error("GET /api/divisions/[divisionId]/requests failed", error);
    return NextResponse.json({ error: "Unable to fetch requests" }, { status: 500 });
  }
}
