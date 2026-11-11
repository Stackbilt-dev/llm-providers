import { ConfigurationError } from '../errors.js';

/**
 * Requests whose `inputTokenCount` is the caller's count plus an estimate the package
 * added (tool-loop turns, classify's system prompt). Kept outside the request, so it
 * never reaches a response-cache key or a provider payload; code that copies such a
 * request carries the mark over with `carryInputTokenCountMark`.
 */
const partlyEstimatedCounts = new WeakSet<object>();

/** Marks a request whose `inputTokenCount` includes the package's own estimate. */
export function markInputTokenCountPartlyEstimated(request: object): void {
  partlyEstimatedCounts.add(request);
}

/** Copies the partly-estimated mark from `from` to its copy `to`. */
export function carryInputTokenCountMark(from: object, to: object): void {
  if (partlyEstimatedCounts.has(from)) partlyEstimatedCounts.add(to);
}

/** Where a request's input token count came from, for `ContextWindowExceededError`. */
export function inputTokenCountSource(request: object & { inputTokenCount?: unknown }): 'caller' | 'mixed' | 'estimate' {
  if (!isValidInputTokenCount(request.inputTokenCount)) return 'estimate';
  return partlyEstimatedCounts.has(request) ? 'mixed' : 'caller';
}

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
