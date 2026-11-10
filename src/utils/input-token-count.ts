import { ConfigurationError } from '../errors.js';

/** Whether a value is a usable `LLMRequest.inputTokenCount`: a positive safe integer. */
export function isValidInputTokenCount(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 1;
}

/**
 * Adds `extra` estimated tokens to a valid caller count. Returns undefined when
 * there is no valid count, or when the sum would leave the safe-integer range;
 * callers then drop the count and fall back to the estimate instead of throwing.
 */
export function addToInputTokenCount(count: unknown, extra: number): number | undefined {
  if (!isValidInputTokenCount(count)) return undefined;
  const sum = count + extra;
  return isValidInputTokenCount(sum) ? sum : undefined;
}

/**
 * Rejects a set but unusable `LLMRequest.inputTokenCount` with a `ConfigurationError`.
 * The factory calls it on entry, so a bad value fails before the response cache or
 * any provider. Catalog helpers and cost estimates ignore a bad value instead.
 */
export function assertInputTokenCount(value: unknown, provider = 'factory'): void {
  if (value !== undefined && !isValidInputTokenCount(value)) {
    throw new ConfigurationError(provider, 'inputTokenCount must be a positive safe integer');
  }
}
