import { RequestStatus } from "@prisma/client";
import { compareTwoStrings } from "string-similarity";
import { prisma } from "@/src/lib/prisma";
import { callRequestParserModel } from "@/src/server/openai/client";

type CandidateInput = {
  raw?: string;
  first_name?: string;
  last_name?: string;
  confidence?: number;
  similarityScore?: number;
};

const AUTO_ACCEPT_THRESHOLD = Number(process.env.AUTO_ACCEPT_THRESHOLD ?? 0.8);
const UNCERTAIN_THRESHOLD = Number(process.env.UNCERTAIN_THRESHOLD ?? 0.5);

const normalizeName = (value: string) =>
  value
    .toLowerCase()
    .replace(/coach\s+/g, "")
    .replace(/[^a-z0-9\s]/g, "")
    .replace(/\s+/g, " ")
    .trim();

function calcSimilarity(a: string, b: string) {
  const na = normalizeName(a);
  const nb = normalizeName(b);
  if (!na || !nb) return 0;
  return compareTwoStrings(na, nb);
}

function finalConfidence(llm: number, similarity: number) {
  const a = Number.isFinite(llm) ? llm : 0;
  const b = Number.isFinite(similarity) ? similarity : 0;
  return 0.5 * a + 0.5 * b;
}

export async function runRequestParsingForDivision(divisionId: string) {
  const division = await prisma.division.findUnique({
    where: { id: divisionId },
  });
  if (!division) {
    throw new Error("Division not found");
  }

  const players = await prisma.player.findMany({
    where: { divisionId },
    select: { id: true, firstName: true, lastName: true },
  });
  const coaches = await prisma.coach.findMany({
    where: { divisionId },
    select: { id: true, name: true },
  });

  const requests = await prisma.request.findMany({
    where: { divisionId, status: { in: [RequestStatus.UNPROCESSED, RequestStatus.ERROR] } },
    include: { player: true },
  });

  if (!requests.length) {
    return { processed: 0 };
  }

  for (const req of requests) {
    await prisma.requestMatchCandidate.deleteMany({ where: { requestId: req.id } });
    try {
      const parsed = await callRequestParserModel({
        requestText: req.rawText,
        requestingPlayer: { firstName: req.player.firstName, lastName: req.player.lastName },
        players,
        coaches: coaches.map((c) => ({ id: c.id, name: c.name })),
      });

      const parsedPlayers: CandidateInput[] = parsed.requested_players ?? [];
      const parsedCoaches: CandidateInput[] = parsed.requested_coaches ?? [];

      const createCandidates = async (
        parsedList: CandidateInput[],
        type: "PLAYER" | "COACH",
      ) => {
        for (const parsedItem of parsedList) {
          const rawName = parsedItem.raw || `${parsedItem.first_name ?? ""} ${parsedItem.last_name ?? ""}`.trim();
          const llmConf = parsedItem.confidence ?? 0.5;
          const simScoreHint = parsedItem.similarityScore ?? null;
          const targets =
            type === "PLAYER"
              ? players.map((p) => ({
                  id: p.id,
                  label: `${p.firstName} ${p.lastName}`,
                  similarity: simScoreHint ?? calcSimilarity(rawName, `${p.firstName} ${p.lastName}`),
                  targetPlayerId: p.id,
                  targetCoachId: null,
                }))
              : coaches.map((c) => ({
                  id: c.id,
                  label: c.name,
                  similarity: simScoreHint ?? calcSimilarity(rawName, c.name),
                  targetPlayerId: null,
                  targetCoachId: c.id,
                }));
          targets.sort((a, b) => b.similarity - a.similarity);
          const top = targets[0];
          const second = targets[1];
          const final = top ? finalConfidence(llmConf, top.similarity) : llmConf;
          if (!top || final < UNCERTAIN_THRESHOLD) {
            await prisma.requestMatchCandidate.create({
              data: {
                requestId: req.id,
                targetType: type,
                targetPlayerId: null,
                targetCoachId: null,
                finalConfidence: final,
                autoStatus: "REJECTED",
                adminDecision: "PENDING",
                decisionReason: "No confident match",
              },
            });
            continue;
          }
          const isAuto =
            final >= AUTO_ACCEPT_THRESHOLD && (!second || final - finalConfidence(llmConf, second.similarity) >= 0.15);
          const status = isAuto ? "AUTO_ACCEPT" : "UNCERTAIN";
          const candidatesToPersist = targets
            .filter((t, idx) => finalConfidence(llmConf, t.similarity) >= UNCERTAIN_THRESHOLD && idx < 3)
            .map((t) => ({
              targetPlayerId: t.targetPlayerId,
              targetCoachId: t.targetCoachId,
              finalConfidence: finalConfidence(llmConf, t.similarity),
            }));
          for (const cand of candidatesToPersist) {
            await prisma.requestMatchCandidate.create({
              data: {
                requestId: req.id,
                targetType: type,
                targetPlayerId: cand.targetPlayerId,
                targetCoachId: cand.targetCoachId,
                finalConfidence: cand.finalConfidence,
                autoStatus: status,
                adminDecision: "PENDING",
              },
            });
          }
        }
      };

      await createCandidates(parsedPlayers, "PLAYER");
      await createCandidates(parsedCoaches, "COACH");

          await prisma.request.update({
            where: { id: req.id },
            data: { status: RequestStatus.PROCESSED },
          });
    } catch (err: any) {
      const message = err?.message || "Parsing failed";
      await prisma.requestMatchCandidate.create({
        data: {
          requestId: req.id,
          targetType: "UNRESOLVED",
          autoStatus: "ERROR",
          adminDecision: "PENDING",
          decisionReason: message.slice(0, 255),
        },
      });
      await prisma.request.update({
        where: { id: req.id },
        data: { status: RequestStatus.ERROR, notes: message },
      });
      throw err;
    }
  }

  return { processed: requests.length };
}
