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
 *
 * `message` may still contain think tags the caller did not strip. `thinkingModel`
 * marks models whose chat template opens the think block itself (QwQ, R1-Distill),
 * so their output has no opening tag: untagged text from them, with no separate
 * reasoning field, is treated as reasoning.
 */
export function lengthCauseFor(
  finishReason: string | undefined,
  message: string,
  reasoning: string | undefined,
  options: { thinkingModel?: boolean } = {},
): 'reasoning' | 'answer' | undefined {
  if (finishReason !== 'length') return undefined;
  const lower = message.toLowerCase();
  const lastOpen = lower.lastIndexOf('<think>');
  const lastClose = lower.lastIndexOf('</think>');
  // An unclosed <think> block means the model was still reasoning at the cap.
  if (lastOpen > lastClose) return 'reasoning';
  if (lastClose >= 0) {
    // Only text after the last closing tag is answer.
    return message.slice(lastClose + '</think>'.length).trim().length > 0 ? 'answer' : 'reasoning';
  }
  const hasReasoning = !!reasoning && reasoning.trim().length > 0;
  if (message.trim().length > 0) {
    return options.thinkingModel && !hasReasoning ? 'reasoning' : 'answer';
  }
  return hasReasoning ? 'reasoning' : undefined;
}

export function joinReasoning(parts: Array<string | null | undefined>): string | undefined {
  const normalized = parts
    .map(part => part?.trim())
    .filter((part): part is string => Boolean(part));

  return normalized.length > 0 ? normalized.join('\n\n') : undefined;
}
