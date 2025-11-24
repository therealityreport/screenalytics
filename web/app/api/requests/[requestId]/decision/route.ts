import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/src/lib/prisma";
import { logAgentInvocation } from "@/src/lib/audit";

const DecisionSchema = z.object({
  selectedCandidateId: z.string().nullable(),
});

export async function POST(request: NextRequest, { params }: { params: { requestId: string } }) {
  const { requestId } = params;
  try {
    const body = await request.json();
    const parsed = DecisionSchema.safeParse(body);
    if (!parsed.success) {
      return NextResponse.json({ error: "Invalid payload", details: parsed.error.errors }, { status: 400 });
    }
    const { selectedCandidateId } = parsed.data;

    const reqRecord = await prisma.request.findUnique({
      where: { id: requestId },
      include: { candidates: true, division: true },
    });
    if (!reqRecord) {
      return NextResponse.json({ error: "Request not found" }, { status: 404 });
    }
    if (reqRecord.candidates.length === 0) {
      return NextResponse.json({ error: "No candidates to decide" }, { status: 400 });
    }
    if (selectedCandidateId) {
      const belongs = reqRecord.candidates.some((c) => c.id === selectedCandidateId);
      if (!belongs) {
        return NextResponse.json({ error: "Candidate does not belong to this request" }, { status: 400 });
      }
      await prisma.requestMatchCandidate.updateMany({
        where: { requestId, id: selectedCandidateId },
        data: { adminDecision: "ACCEPTED" },
      });
      await prisma.requestMatchCandidate.updateMany({
        where: { requestId, id: { not: selectedCandidateId } },
        data: { adminDecision: "DENIED" },
      });
    } else {
      await prisma.requestMatchCandidate.updateMany({
        where: { requestId },
        data: { adminDecision: "DENIED" },
      });
    }

    await prisma.request.update({
      where: { id: requestId },
      data: { status: "RESOLVED" },
    });

    await logAgentInvocation({
      runType: "REQUEST_DECISION",
      divisionId: reqRecord.divisionId,
      eventId: reqRecord.eventId,
      payload: { requestId, selectedCandidateId },
      agentRunId: null,
    });

    const updated = await prisma.request.findUnique({
      where: { id: requestId },
      include: { candidates: true },
    });

    return NextResponse.json({
      requestId,
      status: reqRecord.status,
      candidates: updated?.candidates ?? [],
    });
  } catch (error) {
    console.error("POST /api/requests/[requestId]/decision failed", error);
    return NextResponse.json({ error: "Unable to store decision" }, { status: 500 });
  }
}
