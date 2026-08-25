export interface LatencySummary {
  p50: number;
  p95: number;
  p99: number;
  min: number;
  max: number;
  mean: number;
  count: number;
}

const EMPTY_SUMMARY: LatencySummary = {
  p50: 0,
  p95: 0,
  p99: 0,
  min: 0,
  max: 0,
  mean: 0,
  count: 0,
};

/** Dependency-free bounded histogram safe to instantiate in Workers. */
export class LatencyHistogram {
  private readonly maxSamples: number;
  private readonly samples = new Map<string, number[]>();

  constructor(maxSamples: number = 1000) {
    if (!Number.isInteger(maxSamples) || maxSamples <= 0) {
      throw new RangeError('maxSamples must be a positive integer');
    }
    this.maxSamples = maxSamples;
  }

  record(provider: string, latencyMs: number): void {
    if (!Number.isFinite(latencyMs)) return;
    const values = this.samples.get(provider) ?? [];
    values.push(latencyMs);
    if (values.length > this.maxSamples) {
      values.splice(0, values.length - this.maxSamples);
    }
    this.samples.set(provider, values);
  }

  percentile(provider: string, p: number): number {
    const values = this.samples.get(provider);
    if (!values?.length) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const bounded = Math.min(100, Math.max(0, p));
    const index = Math.ceil((bounded / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)];
  }

  summary(provider: string): LatencySummary {
    const values = this.samples.get(provider);
    if (!values?.length) return { ...EMPTY_SUMMARY };
    let min = Infinity;
    let max = -Infinity;
    let total = 0;
    for (const value of values) {
      min = Math.min(min, value);
      max = Math.max(max, value);
      total += value;
    }
    return {
      p50: this.percentile(provider, 50),
      p95: this.percentile(provider, 95),
      p99: this.percentile(provider, 99),
      min,
      max,
      mean: total / values.length,
      count: values.length,
    };
  }

  allSummaries(): Record<string, LatencySummary> {
    const result: Record<string, LatencySummary> = {};
    for (const provider of this.samples.keys()) {
      result[provider] = this.summary(provider);
    }
    return result;
  }

  reset(provider?: string): void {
    if (provider !== undefined) {
      this.samples.delete(provider);
    } else {
      this.samples.clear();
    }
  }
}

export const defaultLatencyHistogram = new LatencyHistogram();
