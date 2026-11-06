import type { ProviderContent } from '../types.js';

/**
 * Anthropic block types that belong to user turns only. They never appear in
 * an Anthropic response, so replayed assistant content containing one was not
 * produced by the provider and is rejected.
 */
const ANTHROPIC_USER_ONLY_BLOCK_TYPES = new Set(['tool_result', 'image', 'document', 'search_result']);

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * Shape check shared by the Anthropic provider (before replaying a turn) and
 * the factory response cache (before serving a hit), so both accept the same
 * values: a provider name and a non-empty array of typed blocks, with no
 * user-turn block types for Anthropic content.
 */
export function isValidProviderContent(value: unknown): value is ProviderContent {
  if (!isPlainObject(value)) return false;
  const { provider, blocks } = value;
  if (typeof provider !== 'string' || provider.length === 0) return false;
  if (!Array.isArray(blocks) || blocks.length === 0) return false;
  return blocks.every(block => isPlainObject(block)
    && typeof block.type === 'string' && block.type.length > 0
    && !(provider === 'anthropic' && ANTHROPIC_USER_ONLY_BLOCK_TYPES.has(block.type)));
}

/** Deep copy, so edits to a copied message or response can't reach the original blocks. */
export function copyProviderContent(content: ProviderContent): ProviderContent {
  return {
    provider: content.provider,
    // A malformed value is copied as-is so the provider still rejects it at dispatch.
    blocks: Array.isArray(content.blocks)
      ? JSON.parse(JSON.stringify(content.blocks)) as ProviderContent['blocks']
      : content.blocks,
  };
}
