import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/src/lib/prisma";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const runId = searchParams.get("runId");
  const divisionId = searchParams.get("divisionId");

  try {
    let run = null;
    if (runId) {
      run = await prisma.agentRun.findUnique({ where: { id: runId } });
    } else if (divisionId) {
      run = await prisma.agentRun.findFirst({
        where: { divisionId },
        orderBy: { startedAt: "desc" },
      });
    }

    if (!run) {
      return NextResponse.json({ status: "idle" });
    }

    const response = {
      status: run.status,
      lastRun: {
        id: run.id,
        status: run.status,
        startedAt: run.startedAt,
        completedAt: run.completedAt,
        errorMessage: run.error ?? null,
      },
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("GET /api/agent-status failed", error);
    return NextResponse.json({ error: "Unable to fetch agent status" }, { status: 500 });
  }
}
