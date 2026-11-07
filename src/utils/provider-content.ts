import type { ProviderContent, ToolCall } from '../types.js';
import { ConfigurationError } from '../errors.js';

/**
 * Anthropic block types that belong to user turns only. They never appear in
 * an Anthropic response, so replayed assistant content containing one was not
 * produced by the provider and is rejected.
 */
const ANTHROPIC_USER_ONLY_BLOCK_TYPES = new Set(['tool_result', 'image', 'document', 'search_result']);

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0;
}

function isValidAnthropicBlock(block: Record<string, unknown>): boolean {
  if (ANTHROPIC_USER_ONLY_BLOCK_TYPES.has(block.type as string)) return false;
  switch (block.type) {
    case 'tool_use':
      return isNonEmptyString(block.id) && isNonEmptyString(block.name) && isPlainObject(block.input);
    // Anthropic rejects a thinking block sent back without its signature or text
    // (empty text is valid: omitted display), a redacted_thinking block without
    // its data, and a text block without text, so such a turn can't be replayed.
    case 'thinking':
      return typeof block.thinking === 'string' && isNonEmptyString(block.signature);
    case 'redacted_thinking':
      return isNonEmptyString(block.data);
    case 'text':
      return typeof block.text === 'string';
    default:
      return true;
  }
}

/**
 * Shape check shared by the Anthropic provider (before replaying a turn) and
 * the factory response cache (before serving a hit), so both accept the same
 * values: a provider name and a non-empty array of typed blocks. For Anthropic
 * content it also rejects user-turn block types, `tool_use` blocks without an
 * id, name or input, `thinking` blocks without string `thinking` text or a
 * signature, `redacted_thinking` blocks without data and `text` blocks without
 * string `text`. Other block types pass unchanged, so
 * blocks Anthropic adds later still round-trip.
 */
export function isValidProviderContent(value: unknown): value is ProviderContent {
  if (!isPlainObject(value)) return false;
  const { provider, blocks } = value;
  if (!isNonEmptyString(provider)) return false;
  if (!Array.isArray(blocks) || blocks.length === 0) return false;
  return blocks.every(block => isPlainObject(block)
    && isNonEmptyString(block.type)
    && (provider !== 'anthropic' || isValidAnthropicBlock(block)));
}

/**
 * Whether the `tool_use` blocks in the content are exactly the given tool calls,
 * by id. A turn sent back with a `tool_use` that has no matching result is
 * rejected by Anthropic, so content and calls must agree before a replay.
 */
export function toolUseIdsMatch(content: ProviderContent, toolCalls: ToolCall[] | undefined): boolean {
  const blockIds = content.blocks
    .filter(block => block.type === 'tool_use')
    .map(block => block.id as string);
  const callIds = (toolCalls ?? []).map(call => call.id);
  if (blockIds.length !== callIds.length) return false;
  const remaining = new Set(callIds);
  return remaining.size === callIds.length && blockIds.every(id => remaining.delete(id));
}

/**
 * Deep copy, so edits to a copied message or response can't reach the original
 * blocks. Non-array blocks are copied as-is so the provider still rejects them
 * at dispatch; blocks that can't be represented as JSON (BigInt, circular
 * references) throw ConfigurationError.
 */
export function copyProviderContent(content: ProviderContent): ProviderContent {
  if (!Array.isArray(content.blocks)) return { provider: content.provider, blocks: content.blocks };
  let blocks: ProviderContent['blocks'];
  try {
    blocks = JSON.parse(JSON.stringify(content.blocks)) as ProviderContent['blocks'];
  } catch (error) {
    throw new ConfigurationError(
      typeof content.provider === 'string' ? content.provider : 'unknown',
      `providerContent.blocks must be JSON content blocks passed back unchanged: ${(error as Error).message}`
    );
  }
  return { provider: content.provider, blocks };
}
