import { ConfigurationError } from '../errors.js';

/** Whether a value is a usable `LLMRequest.inputTokenCount`: a positive safe integer. */
export function isValidInputTokenCount(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 1;
}

/**
 * Rejects a set but unusable `LLMRequest.inputTokenCount` with a `ConfigurationError`.
 * The factory calls it on entry, so a bad value fails before the response cache or
 * any provider. Catalog helpers and cost estimates ignore a bad value instead.
 */
export function assertInputTokenCount(value: unknown, provider = 'factory'): void {
  if (value !== undefined && !isValidInputTokenCount(value)) {
    throw new ConfigurationError(provider, 'inputTokenCount must be a positive integer');
  }
}
