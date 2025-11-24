import { Prisma } from "@prisma/client";
import { prisma } from "./prisma";

type LogAgentInvocationArgs = {
  runType: string;
  divisionId?: string;
  eventId?: string;
  payload?: unknown;
  message?: string;
  agentRunId?: string;
};

export async function logAgentInvocation(args: LogAgentInvocationArgs) {
  const { runType, divisionId, eventId, payload, message, agentRunId } = args;
  const log = await prisma.auditLog.create({
    data: {
      eventType: `agent:${runType}`,
      divisionId: divisionId ?? null,
      eventId: eventId ?? null,
      payload: (payload ?? undefined) as Prisma.InputJsonValue | undefined,
      message: message ?? "Agent invocation recorded",
    },
  });
  if (agentRunId) {
    await prisma.agentRun.update({
      where: { id: agentRunId },
      data: { auditLogId: log.id },
    });
  }
  return log;
}
