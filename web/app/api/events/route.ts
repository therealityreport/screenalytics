import { NextResponse } from "next/server";
import { prisma } from "@/src/lib/prisma";

export async function GET() {
  try {
    const events = await prisma.event.findMany({
      orderBy: { createdAt: "desc" },
    });
    return NextResponse.json({ events });
  } catch (error) {
    console.error("GET /api/events failed", error);
    return NextResponse.json({ error: "Unable to fetch events" }, { status: 500 });
  }
}

export async function POST() {
  try {
    const now = new Date();
    const event = await prisma.event.create({
      data: {
        name: `Untitled Event ${now.toISOString()}`,
        startDate: now,
      },
    });
    return NextResponse.json({ event }, { status: 201 });
  } catch (error) {
    console.error("POST /api/events failed", error);
    return NextResponse.json({ error: "Unable to create event" }, { status: 500 });
  }
}
