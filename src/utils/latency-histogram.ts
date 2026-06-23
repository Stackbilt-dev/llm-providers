import { LatencyHistogram as WasmLatencyHistogram } from '@stackbilt/wasm-core';

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
  private inner: WasmLatencyHistogram;
  private providers: Set<string> = new Set();

  constructor(maxSamples: number = 1000) {
    this.inner = new WasmLatencyHistogram(maxSamples);
  }

  record(provider: string, latencyMs: number): void {
    this.providers.add(provider);
    this.inner.record(provider, latencyMs);
  }

  percentile(provider: string, p: number): number {
    return this.inner.percentile(provider, p);
  }

  summary(provider: string): LatencySummary {
    return this.inner.summary(provider) as LatencySummary;
  }

  allSummaries(): Record<string, LatencySummary> {
    const result: Record<string, LatencySummary> = {};
    for (const provider of this.providers) {
      result[provider] = this.inner.summary(provider) as LatencySummary;
    }
    return result;
  }

  reset(provider?: string): void {
    if (provider !== undefined) {
      this.providers.delete(provider);
      this.inner.reset(provider);
    } else {
      this.providers.clear();
      this.inner.reset();
    }
  }
}

export const defaultLatencyHistogram = new LatencyHistogram();
