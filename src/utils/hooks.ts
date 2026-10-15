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

const HOOK_METHODS = [
  'onRequestStart', 'onRequestEnd', 'onCache', 'onRequestError',
  'onRetry', 'onFallback', 'onCircuitStateChange',
  'onQuotaExhausted', 'onBudgetThreshold', 'onQuotaCheck',
  'onQuotaDenied', 'onProviderBalance', 'onSchemaDrift',
  'onImageAnalysisAttempt', 'onImageAnalysisComplete',
] as const;

// Compile-time check: adding a hook to ObservabilityHooks without listing it here fails typecheck.
type MissingHookMethods = Exclude<keyof ObservabilityHooks, typeof HOOK_METHODS[number]>;
const hookMethodsAreExhaustive: [MissingHookMethods] extends [never] ? true : MissingHookMethods = true;
void hookMethodsAreExhaustive;

/** Receives a handler failure that was contained; must not throw itself. */
export type HookErrorReporter = (method: string, error: unknown) => void;

/** Call one handler with `owner` as `this`; sync throws and async rejections are contained. */
function invokeHandler(
  owner: object, method: string, handler: Function, event: unknown, onError?: HookErrorReporter
): void {
  const report = (error: unknown) => {
    try { onError?.(method, error); } catch { /* Reporting is best effort. */ }
  };
  try {
    const result: unknown = handler.call(owner, event);
    if (result && typeof (result as PromiseLike<unknown>).then === 'function') {
      (result as PromiseLike<unknown>).then(undefined, report);
    }
  } catch (error) {
    // Observability must not break the request path.
    report(error);
  }
}

/**
 * Merge multiple hook implementations. Each matching handler is called
 * in order. Errors in one handler (thrown or rejected) don't block others.
 * Handlers are captured when composing and called with their implementation as `this`.
 */
export function composeHooks(...implementations: ObservabilityHooks[]): ObservabilityHooks {
  const composed: ObservabilityHooks = {};

  for (const method of HOOK_METHODS) {
    const handlers = implementations
      .filter(impl => typeof impl[method] === 'function')
      .map(impl => ({ owner: impl, handler: impl[method] as Function }));

    if (handlers.length > 0) {
      (composed as Record<string, Function>)[method] = (event: unknown) => {
        for (const { owner, handler } of handlers) invokeHandler(owner, method, handler, event);
      };
    }
  }

  return composed;
}

const guardedHooks = new WeakSet<object>();

/**
 * Wrap a caller's hooks so no handler can throw or reject into the request path.
 * Reads pass through to `hooks` at call time: only methods the caller defined
 * are present, later mutation is seen, and handlers keep `hooks` as `this`.
 * Internal: the factory and every adapter route lifecycle events through this (#116).
 */
export function guardHooks(hooks: ObservabilityHooks, onError?: HookErrorReporter): ObservabilityHooks {
  if (guardedHooks.has(hooks)) return hooks;
  const guarded = new Proxy(hooks, {
    get(target, property, receiver) {
      const value: unknown = Reflect.get(target, property, receiver);
      if (typeof value !== 'function' || typeof property !== 'string') return value;
      return (event: unknown) => invokeHandler(target, property, value, event, onError);
    },
  });
  guardedHooks.add(guarded);
  return guarded;
}

/** Report each failing hook method once per reporter, so a broken sink cannot flood logs. */
export function warnOnceHookErrors(warn: (message: string, error: unknown) => void, source: string): HookErrorReporter {
  const reported = new Set<string>();
  return (method, error) => {
    if (reported.has(method)) return;
    reported.add(method);
    warn(`[${source}] Observability hook ${method} failed; further failures are not logged:`,
      error instanceof Error ? error.message : error);
  };
}
