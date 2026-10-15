import { CircuitBreakerManager, defaultCircuitBreakerManager } from './circuit-breaker.js';
import { ExhaustionRegistry, defaultExhaustionRegistry } from './exhaustion.js';
import { CostTracker, defaultCostTracker } from './cost-tracker.js';
import { LatencyHistogram, defaultLatencyHistogram } from './latency-histogram.js';
import type { Logger } from './logger.js';

/** Share explicitly for one credential/account scope, or allocate per client. */
export class ResilienceContext {
  readonly circuitBreakers: CircuitBreakerManager;
  readonly exhaustion = new ExhaustionRegistry();
  readonly latency = new LatencyHistogram();
  readonly costTracker: CostTracker;

  constructor(logger?: Logger) {
    this.circuitBreakers = new CircuitBreakerManager(undefined, logger);
    this.costTracker = new CostTracker({}, undefined, logger);
  }
}

/** Compatibility context; isolated state can become the default in the next major. */
export const defaultResilienceContext: ResilienceContext = {
  circuitBreakers: defaultCircuitBreakerManager,
  exhaustion: defaultExhaustionRegistry,
  latency: defaultLatencyHistogram,
  costTracker: defaultCostTracker,
};
