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

/**
 * For a length-truncated response, tell whether the budget ran out during reasoning
 * (no answer text yet) or during the answer. Undefined for other finish reasons or
 * when there is neither answer nor reasoning to judge from.
 */
export function lengthCauseFor(
  finishReason: string | undefined,
  message: string,
  reasoning: string | undefined,
): 'reasoning' | 'answer' | undefined {
  if (finishReason !== 'length') return undefined;
  // An unclosed <think> block means the model was still reasoning at the cap.
  if (/^\s*<think>/i.test(message) && !/<\/think>/i.test(message)) return 'reasoning';
  if (message.trim().length > 0) return 'answer';
  return reasoning && reasoning.trim().length > 0 ? 'reasoning' : undefined;
}

export function joinReasoning(parts: Array<string | null | undefined>): string | undefined {
  const normalized = parts
    .map(part => part?.trim())
    .filter((part): part is string => Boolean(part));

  return normalized.length > 0 ? normalized.join('\n\n') : undefined;
}
