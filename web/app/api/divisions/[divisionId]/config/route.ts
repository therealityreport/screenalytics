import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/src/lib/prisma";

const DEFAULT_CONFIG = {
  targetPlayersPerTeam: 10,
  minPlayersPerTeam: 9,
  maxPlayersPerTeam: 11,
  teamCount: null as number | null,
};

const ConfigSchema = z.object({
  targetPlayersPerTeam: z.number().int().positive(),
  minPlayersPerTeam: z.number().int().positive(),
  maxPlayersPerTeam: z.number().int().positive(),
  teamCount: z.number().int().positive().nullable().optional(),
});

export async function GET(_request: NextRequest, { params }: { params: { divisionId: string } }) {
  const { divisionId } = params;
  try {
    const division = await prisma.division.findUnique({
      where: { id: divisionId },
      include: { divisionConfig: true },
    });
    if (!division) {
      return NextResponse.json({ error: "Division not found" }, { status: 404 });
    }
    const cfg = division.divisionConfig;
    if (!cfg) {
      return NextResponse.json({ divisionId, ...DEFAULT_CONFIG });
    }
    return NextResponse.json({
      divisionId,
      targetPlayersPerTeam: cfg.targetPlayersPerTeam,
      minPlayersPerTeam: cfg.minPlayersPerTeam,
      maxPlayersPerTeam: cfg.maxPlayersPerTeam,
      teamCount: cfg.teamCount,
    });
  } catch (error) {
    console.error("GET /api/divisions/[divisionId]/config failed", error);
    return NextResponse.json({ error: "Unable to fetch division config" }, { status: 500 });
  }
}

export async function POST(request: NextRequest, { params }: { params: { divisionId: string } }) {
  const { divisionId } = params;
  try {
    const body = await request.json();
    const parsed = ConfigSchema.safeParse(body);
    if (!parsed.success) {
      return NextResponse.json({ error: "Invalid payload", details: parsed.error.errors }, { status: 400 });
    }
    const { targetPlayersPerTeam, minPlayersPerTeam, maxPlayersPerTeam, teamCount } = parsed.data;
    if (!(minPlayersPerTeam <= targetPlayersPerTeam && targetPlayersPerTeam <= maxPlayersPerTeam)) {
      return NextResponse.json(
        { error: "Validation failed", details: "min <= target <= max must hold" },
        { status: 400 },
      );
    }
    const division = await prisma.division.findUnique({ where: { id: divisionId } });
    if (!division) {
      return NextResponse.json({ error: "Division not found" }, { status: 404 });
    }
    const saved = await prisma.divisionConfig.upsert({
      where: { divisionId },
      update: { targetPlayersPerTeam, minPlayersPerTeam, maxPlayersPerTeam, teamCount: teamCount ?? null },
      create: {
        divisionId,
        targetPlayersPerTeam,
        minPlayersPerTeam,
        maxPlayersPerTeam,
        teamCount: teamCount ?? null,
      },
    });
    return NextResponse.json({
      divisionId,
      targetPlayersPerTeam: saved.targetPlayersPerTeam,
      minPlayersPerTeam: saved.minPlayersPerTeam,
      maxPlayersPerTeam: saved.maxPlayersPerTeam,
      teamCount: saved.teamCount,
    });
  } catch (error) {
    console.error("POST /api/divisions/[divisionId]/config failed", error);
    return NextResponse.json({ error: "Unable to save division config" }, { status: 500 });
  }
}
