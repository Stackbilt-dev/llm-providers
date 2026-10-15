/**
 * Observability Hooks
 *
 * Structured lifecycle events for LLM provider operations.
 * Callers inject an ObservabilityHooks implementation to receive
 * typed events at every interesting moment in the request lifecycle.
 *
 * This replaces string-based logger calls for telemetry. The Logger
 * interface stays for debug output; hooks are for structured observability.
 */

import type {
  CacheObservability,
  CircuitBreakerState,
  ProviderBalance,
  ProviderMetrics,
  QuotaCheckInput,
  QuotaCheckResult,
  TokenUsage,
  CostProvenance,
  TokenProvenance
} from '../types.js';

// ── Event types ──────────────────────────────────────────────────────────

export interface RequestStartEvent {
  provider: string;
  model: string;
  requestId?: string;
  tenantId?: string;
  timestamp: number;
}

export interface RequestEndEvent {
  provider: string;
  model: string;
  requestId?: string;
  tenantId?: string;
  durationMs: number;
  usage: TokenUsage;
  finishReason?: string;
  outcome?: 'success' | 'error' | 'cancelled';
  cache?: CacheObservability;
  timestamp: number;
}

export interface CacheEvent {
  provider?: string;
  model?: string;
  requestId?: string;
  tenantId?: string;
  layer: 'provider-prefix' | 'ai-gateway-response' | 'factory-response';
  status: string;
  cache: CacheObservability;
  timestamp: number;
}

export interface RequestErrorEvent {
  provider: string;
  model: string;
  requestId?: string;
  tenantId?: string;
  error: Error;
  errorCode?: string;
  attempt: number;
  willRetry: boolean;
  timestamp: number;
}

export interface RetryEvent {
  provider: string;
  requestId?: string;
  attempt: number;
  maxAttempts: number;
  delayMs: number;
  error: Error;
  timestamp: number;
}

export interface FallbackEvent {
  fromProvider: string;
  toProvider: string;
  requestId?: string;
  reason: string;
  errorCode?: string;
  timestamp: number;
}

export interface CircuitStateChangeEvent {
  provider: string;
  fromState: CircuitBreakerState['state'];
  toState: CircuitBreakerState['state'];
  consecutiveFailures: number;
  trafficPct: number;
  timestamp: number;
}

export interface QuotaExhaustedEvent {
  provider: string;
  resetAfterMs: number;
  timestamp: number;
}

export interface BudgetThresholdEvent {
  provider: string;
  tier: 'warning' | 'critical' | 'emergency';
  utilizationPct: number;
  spend: number;
  budget: number;
  timestamp: number;
}

export interface QuotaCheckEvent {
  input: QuotaCheckInput;
  result: QuotaCheckResult;
  timestamp: number;
}

export interface QuotaDeniedEvent {
  input: QuotaCheckInput;
  reason?: string;
  timestamp: number;
}

export interface ProviderBalanceEvent {
  provider: string;
  balance: ProviderBalance;
  timestamp: number;
}

export interface SchemaDriftEvent {
  provider: string;
  model?: string;
  requestId?: string;
  path: string;
  expected: string;
  actual: string;
  timestamp: number;
}

export type ImageAnalysisMetadata = Record<string, string | number | boolean | null>;

export interface SanitizedTelemetryError {
  code: string;
  category:
    | 'authentication'
    | 'rate_limit'
    | 'quota'
    | 'timeout'
    | 'circuit_breaker'
    | 'schema_drift'
    | 'invalid_request'
    | 'configuration'
    | 'provider'
    | 'unknown';
}

/** One physical provider call, including provider-internal retries. */
export interface ImageAnalysisAttemptEvent {
  operation: 'analyze_image';
  requestId?: string;
  metadata?: ImageAnalysisMetadata;
  provider: string;
  model: string;
  /** Monotonic attempt number across the whole analyzeImage operation. */
  attempt: number;
  /** Attempt number within this provider's retry loop. */
  providerAttempt: number;
  outcome: 'success' | 'error';
  latencyMs: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  tokenProvenance: TokenProvenance;
  imageCount: number;
  decodedImageBytes: number;
  costUsd: number | null;
  costProvenance: CostProvenance;
  retry: boolean;
  willRetry: boolean;
  fallback: boolean;
  fallbackFromProvider?: string;
  error?: SanitizedTelemetryError;
  timestamp: number;
}

/** Terminal aggregate for one analyzeImage operation. */
export interface ImageAnalysisCompleteEvent {
  operation: 'analyze_image';
  requestId?: string;
  metadata?: ImageAnalysisMetadata;
  outcome: 'success' | 'error';
  attempts: number;
  totalLatencyMs: number;
  durationMs: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  imageCount: number;
  decodedImageBytes: number;
  providerReportedCostUsd: number;
  catalogEstimatedCostUsd: number;
  unknownCostAttempts: number;
  finalProvider?: string;
  finalModel?: string;
  error?: SanitizedTelemetryError;
  timestamp: number;
}

// ── Hooks interface ──────────────────────────────────────────────────────

export interface ObservabilityHooks {
  onRequestStart?(event: RequestStartEvent): void;
  onRequestEnd?(event: RequestEndEvent): void;
  onCache?(event: CacheEvent): void;
  onRequestError?(event: RequestErrorEvent): void;
  onRetry?(event: RetryEvent): void;
  onFallback?(event: FallbackEvent): void;
  onCircuitStateChange?(event: CircuitStateChangeEvent): void;
  onQuotaExhausted?(event: QuotaExhaustedEvent): void;
  onBudgetThreshold?(event: BudgetThresholdEvent): void;
  onQuotaCheck?(event: QuotaCheckEvent): void;
  onQuotaDenied?(event: QuotaDeniedEvent): void;
  onProviderBalance?(event: ProviderBalanceEvent): void;
  onSchemaDrift?(event: SchemaDriftEvent): void;
  onImageAnalysisAttempt?(event: ImageAnalysisAttemptEvent): void;
  onImageAnalysisComplete?(event: ImageAnalysisCompleteEvent): void;
}

/** Silent hooks — default. */
export const noopHooks: ObservabilityHooks = {};

/**
 * Merge multiple hook implementations. Each matching handler is called
 * in order. Errors in one handler don't block others.
 */
export function composeHooks(...implementations: ObservabilityHooks[]): ObservabilityHooks {
  const composed: ObservabilityHooks = {};

  for (const method of HOOK_METHODS) {
    const owners = implementations.filter(impl => typeof impl[method] === 'function');

    if (owners.length > 0) {
      (composed as Record<string, Function>)[method] = (event: unknown) => {
        for (const owner of owners) invokeHook(owner, method, event);
      };
    }
  }

  return composed;
}

const HOOK_METHODS = [
  'onRequestStart', 'onRequestEnd', 'onCache', 'onRequestError',
  'onRetry', 'onFallback', 'onCircuitStateChange',
  'onQuotaExhausted', 'onBudgetThreshold', 'onQuotaCheck',
  'onQuotaDenied', 'onProviderBalance', 'onSchemaDrift',
  'onImageAnalysisAttempt', 'onImageAnalysisComplete',
] as const;

/** Call one handler with its owner as `this`; sync throws and async rejections are swallowed. */
function invokeHook(owner: ObservabilityHooks, method: typeof HOOK_METHODS[number], event: unknown): void {
  try {
    const handler = owner[method] as ((event: unknown) => unknown) | undefined;
    if (typeof handler !== 'function') return;
    const result = handler.call(owner, event);
    if (result && typeof (result as PromiseLike<unknown>).then === 'function') {
      (result as PromiseLike<unknown>).then(undefined, () => {});
    }
  } catch {
    // Observability must not break the request path.
  }
}

/**
 * Wrap a caller's hooks so no handler can throw into the request path.
 * Handlers are looked up at call time, so later mutation of `hooks` is seen.
 * Internal: the factory routes every lifecycle event through this (#116).
 */
export function guardHooks(hooks: ObservabilityHooks): ObservabilityHooks {
  const guarded: ObservabilityHooks = {};
  for (const method of HOOK_METHODS) {
    (guarded as Record<string, Function>)[method] = (event: unknown) => invokeHook(hooks, method, event);
  }
  return guarded;
}
