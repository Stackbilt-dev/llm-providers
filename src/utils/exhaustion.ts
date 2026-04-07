/**
 * Exhaustion Registry
 *
 * Tracks which providers are currently quota-exhausted and auto-resets
 * after a configurable cooldown period. Eliminates the need for callers
 * to maintain their own exhaustion maps.
 */

export interface ExhaustionEntry {
  provider: string;
  exhaustedAt: number;
  resetAt: number;
}

/** Default cooldown: 5 minutes */
const DEFAULT_RESET_MS = 300_000;

export class ExhaustionRegistry {
  private entries: Map<string, ExhaustionEntry> = new Map();
  readonly defaultResetMs: number;

  constructor(defaultResetMs: number = DEFAULT_RESET_MS) {
    this.defaultResetMs = defaultResetMs;
  }

  /**
   * Mark a provider as quota-exhausted.
   * Automatically clears after `resetAfterMs` (default: 5 min).
   */
  markExhausted(provider: string, resetAfterMs?: number): void {
    const resetMs = resetAfterMs ?? this.defaultResetMs;
    const now = Date.now();
    this.entries.set(provider, {
      provider,
      exhaustedAt: now,
      resetAt: now + resetMs,
    });
  }

  /**
   * Check if a provider is currently exhausted.
   * Automatically clears expired entries.
   */
  isExhausted(provider: string): boolean {
    const entry = this.entries.get(provider);
    if (!entry) return false;

    if (Date.now() >= entry.resetAt) {
      this.entries.delete(provider);
      return false;
    }

    return true;
  }

  /** Manually clear exhaustion for a provider. */
  clearExhaustion(provider: string): void {
    this.entries.delete(provider);
  }

  /** Get list of currently exhausted providers. */
  getExhaustedProviders(): string[] {
    const now = Date.now();
    const exhausted: string[] = [];

    for (const [provider, entry] of this.entries) {
      if (now >= entry.resetAt) {
        this.entries.delete(provider);
      } else {
        exhausted.push(provider);
      }
    }

    return exhausted;
  }

  /** Get the entry for a specific provider (if exhausted). */
  getEntry(provider: string): ExhaustionEntry | undefined {
    if (!this.isExhausted(provider)) return undefined;
    return this.entries.get(provider);
  }

  /** Clear all exhaustion state. */
  reset(): void {
    this.entries.clear();
  }
}

/** Shared singleton — same pattern as defaultCircuitBreakerManager. */
export const defaultExhaustionRegistry = new ExhaustionRegistry();
