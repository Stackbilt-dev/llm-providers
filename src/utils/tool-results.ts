import type { ToolResult } from '../types.js';

/**
 * `content` of an OpenAI-compatible `role: 'tool'` message. A failed tool's
 * error must reach the model, so a result with `error` is sent as
 * `{ output, error }` JSON; a successful one is sent as its output.
 */
export function openAIToolResultContent(result: ToolResult): string {
  return result.error
    ? JSON.stringify({ output: result.output, error: result.error })
    : result.output;
}
