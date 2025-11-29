import Link from "next/link";
import { prisma } from "@/src/lib/prisma";

async function getEvents() {
  return prisma.event.findMany({
    orderBy: { createdAt: "desc" },
  });
}

async function createSampleEvent() {
  "use server";
  const now = new Date();
  const name = `Sample Event ${now.toISOString().slice(0, 10)}`;
  await prisma.event.create({
    data: {
      name,
      startDate: now,
    },
  });
}

export default async function HomePage() {
  const events = await getEvents();
  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h1 style={{ margin: 0 }}>Events</h1>
        <form action={createSampleEvent}>
          <button type="submit">Create sample event</button>
        </form>
      </div>
      {events.length === 0 ? (
        <p style={{ marginTop: 16 }}>No events yet. Create one to get started.</p>
      ) : (
        <ul style={{ marginTop: 16, paddingLeft: 18 }}>
          {events.map((event) => (
            <li key={event.id} style={{ marginBottom: 8 }}>
              <Link href={`/events/${event.id}`}>{event.name}</Link>
            </li>
          ))}
        </ul>
      )}
      <p style={{ marginTop: 16 }}>
        Need request analysis? Visit <Link href="/request-analysis">Request Analysis</Link>.
      </p>
    </div>
  );
}
