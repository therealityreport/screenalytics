import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/src/lib/prisma";
import { logAgentInvocation } from "@/src/lib/audit";
import { runRequestParsingForDivision } from "@/src/server/services/requestParsingRunner";

const RunPayloadSchema = z.object({
  divisionId: z.string().min(1),
  runType: z.string().min(1),
});

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const parsed = RunPayloadSchema.safeParse(body);
    if (!parsed.success) {
      return NextResponse.json({ error: "Invalid payload", details: parsed.error.errors }, { status: 400 });
    }
    const { divisionId, runType } = parsed.data;

    const division = await prisma.division.findUnique({ where: { id: divisionId } });
    if (!division) {
      return NextResponse.json({ error: "Division not found" }, { status: 404 });
    }

    const run = await prisma.agentRun.create({
      data: {
        divisionId,
        runType,
        status: "running",
      },
    });

    await logAgentInvocation({
      runType,
      divisionId,
      eventId: division.eventId,
      payload: { divisionId, runType },
      agentRunId: run.id,
    });

    try {
      if (runType === "request-parsing") {
        await runRequestParsingForDivision(divisionId);
      }
      await prisma.agentRun.update({
        where: { id: run.id },
        data: { status: "completed", completedAt: new Date() },
      });
      return NextResponse.json({ runId: run.id, status: "completed" });
    } catch (err: any) {
      const message = err?.message || "Agent run failed";
      await prisma.agentRun.update({
        where: { id: run.id },
        data: { status: "failed", error: message, completedAt: new Date() },
      });
      return NextResponse.json({ runId: run.id, status: "failed", error: message }, { status: 500 });
    }

  } catch (error) {
    console.error("POST /api/agent-run failed", error);
    return NextResponse.json({ error: "Unable to start agent run" }, { status: 500 });
  }
}
