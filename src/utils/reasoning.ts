export interface ReasoningExtraction {
  message: string;
  reasoning?: string;
}

export function extractThinkBlocks(content: string): ReasoningExtraction {
  const reasoning: string[] = [];
  const message = content
    .replace(/<think>([\s\S]*?)<\/think>/gi, (_match, inner: string) => {
      const trimmed = inner.trim();
      if (trimmed.length > 0) reasoning.push(trimmed);
      return '';
    })
    .trim();

  return {
    message,
    reasoning: joinReasoning(reasoning),
  };
}

export function joinReasoning(parts: Array<string | null | undefined>): string | undefined {
  const normalized = parts
    .map(part => part?.trim())
    .filter((part): part is string => Boolean(part));

  return normalized.length > 0 ? normalized.join('\n\n') : undefined;
}
