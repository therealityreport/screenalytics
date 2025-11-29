import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/src/lib/prisma";
import { ingestCsvForEvent } from "@/src/server/services/csvIngest";

export async function POST(request: NextRequest, { params }: { params: { eventId: string } }) {
  const { eventId } = params;
  try {
    const event = await prisma.event.findUnique({ where: { id: eventId } });
    if (!event) {
      return NextResponse.json({ error: "Event not found" }, { status: 404 });
    }
    const formData = await request.formData();
    const file = formData.get("file");
    const seasonRaw = formData.get("season");
    const yearRaw = formData.get("year");
    if (!(file instanceof File)) {
      return NextResponse.json({ error: "CSV file is required (field: file)" }, { status: 400 });
    }
    const season = typeof seasonRaw === "string" ? seasonRaw.trim() : "";
    const yearVal = typeof yearRaw === "string" ? parseInt(yearRaw, 10) : NaN;
    if (!season || Number.isNaN(yearVal) || yearVal < 2000 || yearVal > 2100) {
      return NextResponse.json(
        { error: "Invalid season or year", details: { season, year: yearRaw } },
        { status: 400 },
      );
    }
    await prisma.event.update({
      where: { id: eventId },
      data: { season, year: yearVal },
    });

    try {
      const summary = await ingestCsvForEvent(eventId, file);
      return NextResponse.json(summary);
    } catch (error: any) {
      if (error?.missing) {
        return NextResponse.json(
          { error: "Missing required columns", missingColumns: error.missing },
          { status: 400 },
        );
      }
      console.error("CSV ingest failed", error);
      return NextResponse.json({ error: "Internal server error" }, { status: 500 });
    }
  } catch (error) {
    console.error("Upload CSV handler failed", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
