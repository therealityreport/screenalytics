import { Readable } from "node:stream";
import { parse } from "csv-parse";
import { RequestStatus } from "@prisma/client";
import { prisma } from "@/src/lib/prisma";

const REQUIRED_COLUMNS = [
  "Event Name",
  "Event Price",
  "Reg ID",
  "Reg Date/Time",
  "Registration Type",
  "Participant Status",
  "First Name",
  "Last Name",
  "DOB",
  "Current Grade",
  "Gender",
  "School Name",
  "City",
  "State",
  "Parent/Guardian Name",
  "Parent/Guardian Cell Phone",
  "Parent/Guardian E-mail",
  "Primary Contact E-mail",
  "Do you want to coach a team?",
  "(If you want to coach a team) Coach's Name - Or Type N/A if NOT Wanting To Coach",
  "(If you want to coach a team) Coach's Cell Phone - Or Type N/A if NOT Wanting To Coach",
  "(If you want to coach a team) Coach's Email - Or Type N/A if NOT Wanting To Coach",
  "Player Request - You may request a friend you would like to be on a team with - Or Type N/A If None",
];

const INSURANCE_KEYWORDS = ["insurance"];

type DivisionKey =
  | "pre-k"
  | "k-1"
  | "2-3-boys"
  | "2-3-girls"
  | "4-5-boys"
  | "4-5-girls";

type DivisionTemplate = {
  key: DivisionKey;
  name: string;
  expectedGender: "male" | "female" | "coed";
};

const DIVISION_TEMPLATES: DivisionTemplate[] = [
  { key: "pre-k", name: "Pre-K Co-ed", expectedGender: "coed" },
  { key: "k-1", name: "K-1 Co-ed", expectedGender: "coed" },
  { key: "2-3-boys", name: "2nd-3rd Grade Boys Division", expectedGender: "male" },
  { key: "2-3-girls", name: "2nd-3rd Grade Girls Division", expectedGender: "female" },
  { key: "4-5-boys", name: "4th-5th Grade Boys Division", expectedGender: "male" },
  { key: "4-5-girls", name: "4th-5th Grade Girls Division", expectedGender: "female" },
];

const REQUIRED_DIVISION_KEYS = new Set<DivisionKey>(DIVISION_TEMPLATES.map((t) => t.key));

class MissingColumnsError extends Error {
  missing: string[];
  constructor(missing: string[]) {
    super("Missing required columns");
    this.missing = missing;
  }
}

const stringOrNull = (value: unknown): string | null => {
  if (value === null || value === undefined) return null;
  const str = String(value).trim();
  return str.length === 0 ? null : str;
};

const normalizeGender = (value: string | null): "male" | "female" | "other" | null => {
  if (!value) return null;
  const normalized = value.trim().toLowerCase();
  if (normalized.startsWith("m")) return "male";
  if (normalized.startsWith("f")) return "female";
  return "other";
};

const normalizePhone = (value: string | null): string | null => {
  if (!value) return null;
  const digits = value.replace(/\D/g, "");
  if (digits.length < 7) return null;
  return digits;
};

const normalizeEmail = (value: string | null): string | null => {
  if (!value) return null;
  const trimmed = value.trim().toLowerCase();
  if (!trimmed || !trimmed.includes("@")) {
    return null;
  }
  return trimmed;
};

const isYes = (value: string | null): boolean => {
  if (!value) return false;
  const normalized = value.trim().toLowerCase();
  return ["yes", "y", "true", "1"].includes(normalized);
};

const cleanCoachName = (value: string | null): string | null => {
  if (!value) return null;
  let cleaned = value.trim();
  cleaned = cleaned.replace(/\(.*?\)/g, "");
  cleaned = cleaned.replace(/N\/A|NA|none|no/i, "").trim();
  cleaned = cleaned.replace(/my husband/gi, "").trim();
  if (!cleaned) return null;
  return cleaned;
};

const deriveCoachKey = (opts: { coachEmail?: string | null; coachPhone?: string | null; primaryEmail?: string | null; coachName?: string | null }) => {
  const email = normalizeEmail(opts.coachEmail) || normalizeEmail(opts.primaryEmail);
  if (email) return `email:${email}`;
  const phone = normalizePhone(opts.coachPhone);
  if (phone) return `phone:${phone}`;
  const name = opts.coachName ? opts.coachName.trim().toLowerCase() : null;
  if (name) return `name:${name}`;
  return null;
};

const normalizeRequestText = (value: string | null): string | null => {
  if (!value) return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  const lowered = trimmed.toLowerCase();
  if (["na", "n/a", "none", "no"].includes(lowered)) return null;
  return trimmed;
};

function resolveDivisionKey(eventPrice: string): DivisionTemplate | null {
  const normalized = eventPrice.trim().toLowerCase();
  if (!normalized || INSURANCE_KEYWORDS.some((kw) => normalized.includes(kw))) {
    return null;
  }
  const includesGrades = (grades: string[]) => grades.every((g) => normalized.includes(g));
  if (normalized.includes("pre-k") || normalized.includes("prek") || normalized.includes("pre k")) {
    return DIVISION_TEMPLATES.find((t) => t.key === "pre-k") ?? null;
  }
  if (normalized.includes("k-1") || normalized.includes("k/1") || normalized.includes("k-1st") || normalized.includes("kindergarten")) {
    return DIVISION_TEMPLATES.find((t) => t.key === "k-1") ?? null;
  }
  const has23 = includesGrades(["2", "3"]) || normalized.includes("2nd-3rd") || normalized.includes("2nd/3rd");
  const has45 = includesGrades(["4", "5"]) || normalized.includes("4th-5th") || normalized.includes("4th/5th");
  const isBoys = normalized.includes("boys") || normalized.includes("boy");
  const isGirls = normalized.includes("girls") || normalized.includes("girl");
  if (has23 && isBoys) return DIVISION_TEMPLATES.find((t) => t.key === "2-3-boys") ?? null;
  if (has23 && isGirls) return DIVISION_TEMPLATES.find((t) => t.key === "2-3-girls") ?? null;
  if (has45 && isBoys) return DIVISION_TEMPLATES.find((t) => t.key === "4-5-boys") ?? null;
  if (has45 && isGirls) return DIVISION_TEMPLATES.find((t) => t.key === "4-5-girls") ?? null;
  return null;
}

type ParsedRow = {
  divisionKey: DivisionKey;
  eventPrice: string;
  regId: string | null;
  participantStatus: string | null;
  firstName: string;
  lastName: string;
  gender: "male" | "female" | "other" | null;
  mismatchedGender: boolean;
  dobRaw: string | null;
  currentGrade: string | null;
  schoolName: string | null;
  city: string | null;
  state: string | null;
  parentName: string | null;
  parentPhone: string | null;
  parentEmail: string | null;
  primaryEmail: string | null;
  registrationType: string | null;
    playerRequest: string | null;
    coachName: string | null;
    coachEmail: string | null;
    coachPhone: string | null;
    coachKey: string | null;
    normalizedRequest: string | null;
};

type DivisionRecord = { id: string; name: string; expectedGender: "male" | "female" | "coed" };

export type IngestSummary = {
  eventId: string;
  totalRows: number;
  divisions: Array<{
    divisionId: string;
    name: string;
    playerCount: number;
    coachCount: number;
    mismatchedGenderCount: number;
    requestsCount: number;
  }>;
};

export async function ingestCsvForEvent(eventId: string, file: File): Promise<IngestSummary> {
  const nodeStream = Readable.fromWeb(file.stream() as any);
  let missingColumns: string[] = [];
  const parsedRows: ParsedRow[] = [];
  let totalRows = 0;
  const divisionUsage = new Map<DivisionKey, { template: DivisionTemplate; eventPrice: string }>();

  const parser = parse({
    columns: (header) => {
      missingColumns = REQUIRED_COLUMNS.filter((required) => !header.includes(required));
      if (missingColumns.length > 0) {
        throw new MissingColumnsError(missingColumns);
      }
      return header;
    },
    skip_empty_lines: true,
    trim: true,
  });

  try {
    for await (const record of nodeStream.pipe(parser)) {
      totalRows += 1;
      const eventPriceRaw = stringOrNull(record["Event Price"]);
      const participantStatus = stringOrNull(record["Participant Status"]);
      if (!eventPriceRaw || !participantStatus) {
        continue;
      }
      if (participantStatus.toLowerCase() !== "approved") {
        continue;
      }
      const divisionTemplate = resolveDivisionKey(eventPriceRaw);
      if (!divisionTemplate || !REQUIRED_DIVISION_KEYS.has(divisionTemplate.key)) {
        continue;
      }
      divisionUsage.set(divisionTemplate.key, { template: divisionTemplate, eventPrice: eventPriceRaw });

      const firstName = stringOrNull(record["First Name"]);
      const lastName = stringOrNull(record["Last Name"]);
      if (!firstName || !lastName) {
        continue;
      }

      const gender = normalizeGender(stringOrNull(record["Gender"]));
      const mismatchedGender =
        divisionTemplate.expectedGender !== "coed" && gender !== null && gender !== divisionTemplate.expectedGender;

      const coachName = cleanCoachName(
        stringOrNull(
          record["(If you want to coach a team) Coach's Name - Or Type N/A if NOT Wanting To Coach"],
        ),
      );
      const coachEmail = stringOrNull(
        record["(If you want to coach a team) Coach's Email - Or Type N/A if NOT Wanting To Coach"],
      );
      const coachPhone = stringOrNull(
        record["(If you want to coach a team) Coach's Cell Phone - Or Type N/A if NOT Wanting To Coach"],
      );
      const wantsCoach = isYes(stringOrNull(record["Do you want to coach a team?"]));
      const coachKey = wantsCoach
        ? deriveCoachKey({ coachEmail, coachPhone, primaryEmail: stringOrNull(record["Primary Contact E-mail"]), coachName })
        : null;

      parsedRows.push({
        divisionKey: divisionTemplate.key,
        eventPrice: eventPriceRaw,
        regId: stringOrNull(record["Reg ID"]),
        participantStatus,
        firstName,
        lastName,
        gender,
        mismatchedGender,
        dobRaw: stringOrNull(record["DOB"]),
        currentGrade: stringOrNull(record["Current Grade"]),
        schoolName: stringOrNull(record["School Name"]),
        city: stringOrNull(record["City"]),
        state: stringOrNull(record["State"]),
        parentName: stringOrNull(record["Parent/Guardian Name"]),
        parentPhone: stringOrNull(record["Parent/Guardian Cell Phone"]),
        parentEmail: stringOrNull(record["Parent/Guardian E-mail"]),
        primaryEmail: stringOrNull(record["Primary Contact E-mail"]),
        registrationType: stringOrNull(record["Registration Type"]),
        playerRequest: stringOrNull(
          record["Player Request - You may request a friend you would like to be on a team with - Or Type N/A If None"],
        ),
        coachName: wantsCoach ? coachName : null,
        coachEmail: wantsCoach ? normalizeEmail(coachEmail) : null,
        coachPhone: wantsCoach ? normalizePhone(coachPhone) : null,
        coachKey: wantsCoach ? coachKey : null,
        normalizedRequest: normalizeRequestText(
          stringOrNull(
            record["Player Request - You may request a friend you would like to be on a team with - Or Type N/A If None"],
          ),
        ),
      });
    }
  } catch (error) {
    if (error instanceof MissingColumnsError) {
      throw error;
    }
    throw new Error("Failed to parse CSV");
  }

  if (missingColumns.length > 0) {
    throw new MissingColumnsError(missingColumns);
  }

  // Ensure divisions exist
  const divisionRecords = new Map<DivisionKey, DivisionRecord>();
  for (const [key, payload] of divisionUsage.entries()) {
    const existing = await prisma.division.findFirst({
      where: { eventId, name: payload.template.name },
    });
    if (existing) {
      divisionRecords.set(key, {
        id: existing.id,
        name: existing.name,
        expectedGender: payload.template.expectedGender,
      });
    } else {
      const created = await prisma.division.create({
        data: {
          eventId,
          name: payload.template.name,
          eventPrice: payload.eventPrice,
          gender: payload.template.expectedGender,
        },
      });
      divisionRecords.set(key, {
        id: created.id,
        name: created.name,
        expectedGender: payload.template.expectedGender,
      });
    }
  }

  // Upsert coaches
  const coachIds = new Map<string, string>();
  const coachCandidates = parsedRows.filter((row) => row.coachKey && row.coachName);
  for (const candidate of coachCandidates) {
    if (!candidate.coachKey || coachIds.has(candidate.coachKey)) continue;
    const divisionId = divisionRecords.get(candidate.divisionKey)?.id;
    const coach = await prisma.coach.upsert({
      where: { coachKey: candidate.coachKey },
      update: {
        name: candidate.coachName!,
        email: candidate.coachEmail ?? undefined,
        phone: candidate.coachPhone ?? undefined,
        divisionId: divisionId ?? undefined,
      },
      create: {
        coachKey: candidate.coachKey,
        name: candidate.coachName!,
        email: candidate.coachEmail,
        phone: candidate.coachPhone,
        divisionId: divisionId ?? null,
        eventId,
      },
    });
    coachIds.set(candidate.coachKey, coach.id);
  }

  // Upsert players + requests
  for (const row of parsedRows) {
    const division = divisionRecords.get(row.divisionKey);
    if (!division) continue;
    const coachId = row.coachKey ? coachIds.get(row.coachKey) : null;
    const data = {
      eventId,
      divisionId: division.id,
      firstName: row.firstName,
      lastName: row.lastName,
      regId: row.regId ?? undefined,
      participantStatus: row.participantStatus ?? undefined,
      eventPrice: row.eventPrice,
      status: "approved",
      gender: row.gender ?? undefined,
      mismatchedGender: row.mismatchedGender,
      dobRaw: row.dobRaw ?? undefined,
      currentGrade: row.currentGrade ?? undefined,
      schoolName: row.schoolName ?? undefined,
      city: row.city ?? undefined,
      state: row.state ?? undefined,
      parentName: row.parentName ?? undefined,
      parentPhone: row.parentPhone ? normalizePhone(row.parentPhone) ?? undefined : undefined,
      parentEmail: row.parentEmail ? normalizeEmail(row.parentEmail) ?? undefined : undefined,
      primaryEmail: row.primaryEmail ? normalizeEmail(row.primaryEmail) ?? undefined : undefined,
      registrationType: row.registrationType ?? undefined,
      playerRequest: row.playerRequest ?? undefined,
      coachId: coachId ?? undefined,
    };
    let player;
    if (row.regId) {
      player = await prisma.player.upsert({
        where: { regId: row.regId },
        update: data,
        create: data,
      });
    } else {
      player = await prisma.player.create({ data });
    }

    // Upsert request for players with a meaningful request
    if (row.normalizedRequest) {
      await prisma.request.upsert({
        where: {
          playerId_divisionId: {
            playerId: player.id,
            divisionId: division.id,
          },
        },
        update: {
          rawText: row.normalizedRequest,
          status: RequestStatus.UNPROCESSED,
          eventId,
        },
        create: {
          rawText: row.normalizedRequest,
          status: RequestStatus.UNPROCESSED,
          eventId,
          divisionId: division.id,
          playerId: player.id,
        },
      });
    }
  }

  const divisionIds = Array.from(divisionRecords.values()).map((d) => d.id);
  const countsByDivision = new Map<string, { playerCount: number; coachCount: number; mismatched: number }>();
  const playerCounts = await prisma.player.groupBy({
    by: ["divisionId"],
    _count: { _all: true },
    where: { divisionId: { in: divisionIds } },
  });
  const coachCounts = await prisma.coach.groupBy({
    by: ["divisionId"],
    _count: { _all: true },
    where: { divisionId: { in: divisionIds } },
  });
  const mismatchedCounts = await prisma.player.groupBy({
    by: ["divisionId"],
    _count: { _all: true },
    where: { divisionId: { in: divisionIds }, mismatchedGender: true },
  });
  const requestCounts = await prisma.request.groupBy({
    by: ["divisionId"],
    _count: { _all: true },
    where: { divisionId: { in: divisionIds } },
  });

  for (const division of divisionRecords.values()) {
    countsByDivision.set(division.id, { playerCount: 0, coachCount: 0, mismatched: 0 });
  }
  for (const item of playerCounts) {
    if (!item.divisionId) continue;
    const entry = countsByDivision.get(item.divisionId);
    if (entry) entry.playerCount = item._count._all;
  }
  for (const item of coachCounts) {
    if (!item.divisionId) continue;
    const entry = countsByDivision.get(item.divisionId);
    if (entry) entry.coachCount = item._count._all;
  }
  for (const item of mismatchedCounts) {
    if (!item.divisionId) continue;
    const entry = countsByDivision.get(item.divisionId);
    if (entry) entry.mismatched = item._count._all;
  }
  const requestsMap = new Map<string, number>();
  for (const item of requestCounts) {
    if (!item.divisionId) continue;
    requestsMap.set(item.divisionId, item._count._all);
  }

  const divisionsSummary = Array.from(divisionRecords.values()).map((division) => {
    const counts = countsByDivision.get(division.id) ?? { playerCount: 0, coachCount: 0, mismatched: 0 };
    return {
      divisionId: division.id,
      name: division.name,
      playerCount: counts.playerCount,
      coachCount: counts.coachCount,
      mismatchedGenderCount: counts.mismatched,
      requestsCount: requestsMap.get(division.id) ?? 0,
    };
  });

  return {
    eventId,
    totalRows,
    divisions: divisionsSummary,
  };
}
