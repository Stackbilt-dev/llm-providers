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
  // Only think structure at the start of the message counts; tags later on are
  // answer text (e.g. code that mentions '<think>').
  let rest = message.trimStart();
  let sawThinkBlock = false;
  if (options.thinkingModel && !/^<think>/i.test(rest)) {
    // Template-opened block: the chat template sent '<think>', so the output starts
    // inside it and only closes it. Without a closing tag, the final check below applies.
    const firstClose = rest.toLowerCase().indexOf('</think>');
    if (firstClose >= 0) {
      rest = rest.slice(firstClose + '</think>'.length).trimStart();
      sawThinkBlock = true;
    }
  }
  for (let match = /^<think>[\s\S]*?<\/think>/i.exec(rest); match; match = /^<think>[\s\S]*?<\/think>/i.exec(rest)) {
    rest = rest.slice(match[0].length).trimStart();
    sawThinkBlock = true;
  }
  // An unclosed <think> block before any answer means the model was still reasoning at the cap.
  if (/^<think>/i.test(rest)) return 'reasoning';
  if (sawThinkBlock) return rest.length > 0 ? 'answer' : 'reasoning';
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
