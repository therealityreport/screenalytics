import OpenAI from "openai";

const MODEL = process.env.REQUEST_PARSER_MODEL || "gpt-4o-mini";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

type ParserInput = {
  requestText: string;
  requestingPlayer?: { firstName: string; lastName: string } | null;
  players: { id: string; firstName: string; lastName: string }[];
  coaches: { id: string; name: string }[];
};

type ParsedResult = {
  requested_players?: Array<{ raw?: string; first_name?: string; last_name?: string; confidence?: number }>;
  requested_coaches?: Array<{ raw?: string; first_name?: string; last_name?: string; confidence?: number }>;
};

function buildPrompt(input: ParserInput) {
  const playerNames = input.players.map((p) => `${p.firstName} ${p.lastName}`).join(", ") || "None";
  const coachNames = input.coaches.map((c) => c.name).join(", ") || "None";
  const requester = input.requestingPlayer
    ? `The request was entered by ${input.requestingPlayer.firstName} ${input.requestingPlayer.lastName}.`
    : "";
  return [
    {
      role: "system" as const,
      content:
        "You extract requested players/coaches from free text. Only use names that appear implied in the request. Do not invent names.",
    },
    {
      role: "user" as const,
      content: `
${requester}
Players in division: ${playerNames}
Coaches in division: ${coachNames}
Request text: """${input.requestText}"""

Return JSON ONLY with the shape:
{
  "requested_players": [{"raw":"", "first_name":"", "last_name":"", "confidence":0.0}],
  "requested_coaches": [{"raw":"", "first_name":"", "last_name":"", "confidence":0.0}]
}
If none, return empty arrays.`,
    },
  ];
}

export async function callRequestParserModel(input: ParserInput): Promise<ParsedResult> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is not set");
  }
  const messages = buildPrompt(input);
  const response = await client.chat.completions.create({
    model: MODEL,
    messages,
    temperature: 0.2,
    response_format: { type: "json_object" },
  });
  const content = response.choices[0]?.message?.content;
  if (!content) {
    throw new Error("Model returned empty content");
  }
  try {
    return JSON.parse(content) as ParsedResult;
  } catch (err) {
    throw new Error("Failed to parse model JSON");
  }
}
