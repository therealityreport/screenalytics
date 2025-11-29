import { NextResponse } from "next/server";
import { prisma } from "@/src/lib/prisma";

export async function GET(_request: Request, { params }: { params: { requestId: string } }) {
  const { requestId } = params;
  try {
    const request = await prisma.request.findUnique({
      where: { id: requestId },
      include: {
        player: true,
        candidates: true,
      },
    });
    if (!request) {
      return NextResponse.json({ error: "Request not found" }, { status: 404 });
    }

    const playerMap = new Map([[request.player.id, `${request.player.firstName} ${request.player.lastName}`]]);
    const coachIds = request.candidates.map((c) => c.targetCoachId).filter(Boolean) as string[];
    const playerIds = request.candidates.map((c) => c.targetPlayerId).filter(Boolean) as string[];
    const coaches = coachIds.length
      ? await prisma.coach.findMany({ where: { id: { in: coachIds } }, select: { id: true, name: true } })
      : [];
    const players =
      playerIds.length && playerIds[0] !== request.player.id
        ? await prisma.player.findMany({
            where: { id: { in: playerIds.filter((id) => id !== request.player.id) } },
            select: { id: true, firstName: true, lastName: true },
          })
        : [];
    players.forEach((p) => playerMap.set(p.id, `${p.firstName} ${p.lastName}`));
    const coachMap = new Map(coaches.map((c) => [c.id, c.name]));

    return NextResponse.json({
      requestId: request.id,
      rawText: request.rawText,
      status: request.status,
      player: {
        id: request.player.id,
        firstName: request.player.firstName,
        lastName: request.player.lastName,
      },
      candidates: request.candidates.map((cand) => ({
        id: cand.id,
        targetType: cand.targetType,
        targetPlayer: cand.targetPlayerId
          ? { id: cand.targetPlayerId, name: playerMap.get(cand.targetPlayerId) }
          : null,
        targetCoach: cand.targetCoachId ? { id: cand.targetCoachId, name: coachMap.get(cand.targetCoachId) } : null,
        llmConfidence: cand.llmConfidence,
        similarityScore: cand.similarityScore,
        finalConfidence: cand.finalConfidence,
        autoStatus: cand.autoStatus,
        adminDecision: cand.adminDecision,
        decisionReason: cand.decisionReason,
      })),
    });
  } catch (error) {
    console.error("GET /api/requests/[requestId]/candidates failed", error);
    return NextResponse.json({ error: "Unable to fetch candidates" }, { status: 500 });
  }
}
