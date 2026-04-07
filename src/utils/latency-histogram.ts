/**
 * Latency Histogram
 *
 * Per-provider latency tracking with percentile computation.
 * Uses a sorted insertion ring buffer to keep memory bounded
 * while supporting efficient percentile queries.
 */

/** Maximum samples per provider before oldest are evicted. */
const DEFAULT_MAX_SAMPLES = 1000;

export interface LatencySummary {
  p50: number;
  p95: number;
  p99: number;
  min: number;
  max: number;
  mean: number;
  count: number;
}

export class LatencyHistogram {
  private buffers: Map<string, number[]> = new Map();
  private maxSamples: number;

  constructor(maxSamples: number = DEFAULT_MAX_SAMPLES) {
    this.maxSamples = maxSamples;
  }

  /** Record a latency measurement for a provider (in milliseconds). */
  record(provider: string, latencyMs: number): void {
    let buffer = this.buffers.get(provider);
    if (!buffer) {
      buffer = [];
      this.buffers.set(provider, buffer);
    }

    buffer.push(latencyMs);

    // Evict oldest when buffer is full.
    if (buffer.length > this.maxSamples) {
      buffer.shift();
    }
  }

  /**
   * Compute a specific percentile for a provider.
   * @param p Percentile as a number between 0 and 100 (e.g. 95 for p95).
   * Returns 0 if no data.
   */
  percentile(provider: string, p: number): number {
    const buffer = this.buffers.get(provider);
    if (!buffer || buffer.length === 0) return 0;

    const sorted = [...buffer].sort((a, b) => a - b);
    const index = Math.ceil((p / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)]!;
  }

  /** Get a full summary for a provider. */
  summary(provider: string): LatencySummary {
    const buffer = this.buffers.get(provider);
    if (!buffer || buffer.length === 0) {
      return { p50: 0, p95: 0, p99: 0, min: 0, max: 0, mean: 0, count: 0 };
    }

    const sorted = [...buffer].sort((a, b) => a - b);
    const count = sorted.length;
    const sum = sorted.reduce((a, b) => a + b, 0);

    return {
      p50: sorted[Math.max(0, Math.ceil(0.50 * count) - 1)]!,
      p95: sorted[Math.max(0, Math.ceil(0.95 * count) - 1)]!,
      p99: sorted[Math.max(0, Math.ceil(0.99 * count) - 1)]!,
      min: sorted[0]!,
      max: sorted[count - 1]!,
      mean: sum / count,
      count,
    };
  }

  /** Get summaries for all tracked providers. */
  allSummaries(): Record<string, LatencySummary> {
    const result: Record<string, LatencySummary> = {};
    for (const provider of this.buffers.keys()) {
      result[provider] = this.summary(provider);
    }
    return result;
  }

  /** Reset one or all providers. */
  reset(provider?: string): void {
    if (provider) {
      this.buffers.delete(provider);
    } else {
      this.buffers.clear();
    }
  }
}

/** Shared singleton. */
export const defaultLatencyHistogram = new LatencyHistogram();
