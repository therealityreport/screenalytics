export type TeamOption = {
  teamCount: number;
  distribution: { size: number; teams: number }[];
  extraPlayersToPerfectMultiple: number;
};

const DEFAULT_MAX_OPTIONS = 20;

export function computeTeamOptions(params: {
  playerCount: number;
  targetPlayersPerTeam: number;
  minPlayersPerTeam: number;
  maxPlayersPerTeam: number;
}): TeamOption[] {
  const { playerCount: N, targetPlayersPerTeam: T, minPlayersPerTeam: minSize, maxPlayersPerTeam: maxSize } = params;
  if (N <= 0 || T <= 0 || minSize <= 0 || maxSize <= 0 || minSize > maxSize) {
    return [];
  }
  const k0 = Math.max(1, Math.round(N / T));
  const kMin = Math.max(1, Math.ceil(N / maxSize));
  const kMax = Math.max(kMin, Math.floor(N / minSize));

  const candidates = new Set<number>();
  [k0 - 1, k0, k0 + 1]
    .filter((k) => k >= 1)
    .forEach((k) => candidates.add(k));
  for (let k = kMin; k <= kMax; k += 1) {
    candidates.add(k);
  }

  const candidateList = Array.from(candidates).sort((a, b) => a - b).slice(0, DEFAULT_MAX_OPTIONS);
  const extraPlayersToPerfectMultiple = Math.ceil(N / T) * T - N;

  const options: TeamOption[] = [];
  for (const k of candidateList) {
    const baseSize = Math.floor(N / k);
    const remainder = N % k;
    const largeSize = baseSize + 1;

    if (baseSize < minSize || baseSize > maxSize) {
      continue;
    }
    if (remainder > 0 && largeSize > maxSize) {
      continue;
    }

    const dist: { size: number; teams: number }[] = [];
    if (largeSize <= maxSize && remainder > 0) {
      dist.push({ size: largeSize, teams: remainder });
    }
    dist.push({ size: baseSize, teams: k - remainder });

    options.push({
      teamCount: k,
      distribution: dist.sort((a, b) => a.size - b.size),
      extraPlayersToPerfectMultiple,
    });
  }

  options.sort((a, b) => {
    const avgA = N / a.teamCount;
    const avgB = N / b.teamCount;
    const diffA = Math.abs(avgA - T);
    const diffB = Math.abs(avgB - T);
    return diffA === diffB ? a.teamCount - b.teamCount : diffA - diffB;
  });

  return options;
}
