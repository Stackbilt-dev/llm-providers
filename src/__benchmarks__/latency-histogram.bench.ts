import { bench, describe } from 'vitest';
import { LatencyHistogram } from '../utils/latency-histogram';

const PROVIDERS = ['openai', 'anthropic', 'groq', 'cerebras', 'nvidia', 'cf'];
const SAMPLES = 1000;

function filledHistogram(maxSamples = SAMPLES): LatencyHistogram {
  const h = new LatencyHistogram(maxSamples);
  for (const p of PROVIDERS)
    for (let i = 0; i < maxSamples; i++) h.record(p, Math.random() * 2000);
  return h;
}

describe('LatencyHistogram — hot path ops (6 providers × 1000 samples)', () => {
  const h = filledHistogram();

  bench('record() eviction path — one provider at capacity', () => {
    h.record('groq', Math.random() * 2000);
  });

  bench('percentile() p95 — single provider', () => {
    h.percentile('groq', 95);
  });

  bench('summary() — single provider', () => {
    h.summary('groq');
  });

  bench('allSummaries() — all 6 providers', () => {
    h.allSummaries();
  });
});

describe('LatencyHistogram — allSummaries() scaling', () => {
  bench('1 provider × 1000 samples', () => {
    const h = new LatencyHistogram(1000);
    for (let i = 0; i < 1000; i++) h.record('groq', Math.random() * 2000);
    h.allSummaries();
  }, { iterations: 200 });

  bench('6 providers × 1000 samples', () => {
    filledHistogram().allSummaries();
  }, { iterations: 200 });
});
