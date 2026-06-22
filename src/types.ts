/**
 * LLM Provider Types
 * Unified types for all LLM providers with v2 architecture support
 */

import type { Logger } from './utils/logger.js';
import type { ObservabilityHooks } from './utils/hooks.js';
export type { Logger, ObservabilityHooks };

export interface LLMMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
  toolCalls?: ToolCall[];
  toolResults?: ToolResult[];
}

export interface LLMImageInput {
  data?: string;
  url?: string;
  mimeType?: string;
}

export interface GatewayMetadata {
  requestId?: string;
  /** cf-aig-cache-key: Cloudflare AI Gateway *response* cache key (distinct from provider prompt caching). */
  cacheKey?: string;
  /** cf-aig-cache-ttl: Cloudflare AI Gateway response cache TTL in seconds. */
  cacheTtl?: number;
  /** cf-aig-skip-cache: bypass the Cloudflare AI Gateway response cache for this request. */
  skipCache?: boolean;
  customMetadata?: Record<string, string>;
}

/**
 * First-class Cloudflare AI Gateway passthrough config attached to an HTTP
 * provider via `cfGateway`. When set (and no explicit `baseUrl` override is
 * present), the provider derives its base URL as
 * `https://gateway.ai.cloudflare.com/v1/{accountId}/{gatewayId}/{suffix}` and
 * injects `cf-aig-*` headers from `LLMRequest.gatewayMetadata`.
 *
 * Both fields are required; an empty string for either throws synchronously in
 * the provider constructor (CF_GATEWAY_INVALID_CONFIG).
 */
export interface CfGatewayConfig {
  /** Cloudflare account ID. Must be non-empty. */
  accountId: string;
  /** AI Gateway ID (the named gateway within the account). Must be non-empty. */
  gatewayId: string;
}

/**
 * Provider prompt/prefix cache hints. Distinct from GatewayMetadata.cacheKey/cacheTtl,
 * which control Cloudflare AI Gateway *response* caching. These hints control
 * provider-side prefix/prompt caching (Anthropic cache_control, Groq/Cerebras automatic, etc.).
 */
/**
 * Provider-agnostic response cache adapter for factory-level response deduplication.
 * Distinct from CacheHints, which controls provider-side prefix/prompt caching.
 * Works with any backing store (Cloudflare KV, Redis, in-memory map).
 *
 * Keys are derived from (model, messages, temperature, maxTokens). For KV stores
 * with a 512-byte key limit, wrap this adapter to hash the key before forwarding.
 */
export interface ResponseCacheAdapter {
  get(key: string): Promise<string | null>;
  put(key: string, value: string, ttlSeconds?: number): Promise<void>;
}

export interface CacheHints {
  /** 'off' disables any provider cache opt-in. Default: let provider decide. */
  strategy?: 'off' | 'provider-prefix' | 'response' | 'both';
  /** Provider-specific cache key hint (e.g. OpenAI prompt_cache_key). */
  key?: string;
  /** Desired TTL. Clamped to provider-supported values at translation time. */
  ttl?: '5m' | '1h' | '24h' | number;
  /** Cloudflare Workers AI: x-session-affinity value for prefix caching. */
  sessionId?: string;
  /** Which prefix to mark cacheable when explicit breakpoints are required (Anthropic). */
  cacheablePrefix?: 'auto' | 'system' | 'tools' | 'messages';
}

export interface CacheObservability {
  /** Provider-side prompt/prefix cache behavior, normalized from TokenUsage fields. */
  providerPrefix?: {
    requested?: boolean;
    strategy?: CacheHints['strategy'];
    sessionId?: string;
    cachedInputTokens?: number;
    cacheReadInputTokens?: number;
    cacheWriteInputTokens?: number;
    cacheCreationInputTokens?: number;
    hit?: boolean;
    write?: boolean;
  };
  /** Cloudflare AI Gateway exact-response cache behavior when the provider exposes it. */
  aiGateway?: {
    requested?: boolean;
    status?: string;
    cacheKey?: string;
    cacheTtl?: number;
  };
  /** Factory/local response cache behavior through ResponseCacheAdapter. */
  factory?: {
    status: 'hit' | 'miss' | 'write' | 'error' | 'bypass';
    key?: string;
    ttlSeconds?: number;
    error?: string;
  };
}

export interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

export interface ToolResult {
  id: string;
  output: string;
  error?: string;
}

export interface LLMRequest {
  messages: LLMMessage[];
  model?: string;
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
  systemPrompt?: string;
  images?: LLMImageInput[];
  tools?: Tool[];
  /**
   * Provider-hosted built-in tools (Groq web search, code interpreter, …).
   * Only honored by providers/models that advertise `supportsBuiltInTools`;
   * adapters translate to the native wire shape and gate at the boundary.
   * Results are surfaced on `LLMResponse.metadata.builtInToolResults`
   * (cast to `BuiltInToolResult[]`).
   */
  builtInTools?: BuiltInTool[];
  toolChoice?: 'auto' | 'none' | { type: 'function'; function: { name: string } };
  response_format?:
    | { type: 'json_object' | 'text' }
    | { type: 'json_schema'; json_schema: { name: string; schema: Record<string, unknown>; strict?: boolean } };
  seed?: number;
  gatewayMetadata?: GatewayMetadata;
  cache?: CacheHints;
  /**
   * Cloudflare Workers AI only: LoRA adapter name or UUID to apply at inference.
   * Ignored by all other providers. The package does not validate the identifier;
   * it is forwarded as-is to the Workers AI binding.
   */
  lora?: string;
  /** Nucleus sampling cutoff. Forwarded by the Cloudflare provider; ignored by others. */
  topP?: number;
  /** Penalizes repeated tokens. Forwarded by the Cloudflare provider; ignored by others. */
  frequencyPenalty?: number;
  /** Provider reasoning/thinking controls. Translated to provider-native params at call time. */
  reasoning?: {
    /** Token budget / quality hint. 'none' disables reasoning entirely where supported. */
    effort?: 'low' | 'medium' | 'high' | 'none';
    /** Cerebras: controls how reasoning tokens appear in the response. */
    format?: 'parsed' | 'raw' | 'hidden';
    /** Cerebras zai-glm-4.7: set false to preserve reasoning across turns (improves cache hit rate). */
    clearThinking?: boolean;
  };
  /** Predicted output hint for speculative decoding (Cerebras gpt-oss-120b, zai-glm-4.7). Incompatible with tools. */
  prediction?: string;
  tenantId?: string;
  requestId?: string;
  metadata?: Record<string, unknown>;
}

export interface Tool {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: {
      type: 'object';
      properties: Record<string, unknown>;
      required?: string[];
    };
  };
}

/**
 * Provider-hosted built-in tools (e.g. Groq web search / code interpreter).
 * Conceptually distinct from caller-defined function `Tool`s: the provider runs
 * them server-side and returns their results inline. Identifiers are normalized
 * across providers (Groq's compound vocabulary is the canonical superset); each
 * adapter translates to its native wire shape and gates on model capability.
 */
export type BuiltInToolType =
  | 'web_search'
  | 'visit_website'
  | 'browser_automation'
  | 'code_interpreter'
  | 'wolfram_alpha';

export interface BuiltInTool {
  type: BuiltInToolType;
}

/**
 * A single built-in tool execution surfaced on `LLMResponse.metadata.builtInToolResults`.
 * Shape mirrors Groq's `message.executed_tools[]` (verified live, issue #69):
 * `type` is provider-native and open-ended (`'search'`, `'browser_search'`,
 * `'browser.open'`, …) — do not treat it as a closed enum. Results carry the
 * provider's web-search citations when the execution produced any.
 */
export interface BuiltInToolResult {
  /** Provider-native tool kind. Open-ended — varies by provider and model. */
  type: string;
  /** Tool name when the provider supplies one (Groq gpt-oss); absent on compound. */
  name?: string;
  /** Raw JSON-string arguments the model passed, e.g. `'{"query":"…"}'`. */
  arguments?: string;
  /** Search results / citations carried by this execution, when present. */
  results: Array<{ title: string; url: string; content: string; score: number }>;
}

export interface LLMResponse {
  id?: string;
  /** Chain-of-thought / thinking trace when the provider exposes it separately. */
  reasoning?: string;
  message: string;
  content?: string; // Alternative to message for consistency
  usage: TokenUsage;
  model: string;
  provider: string;
  responseTime: number;
  finishReason?: 'stop' | 'length' | 'tool_calls' | 'content_filter';
  toolCalls?: ToolCall[];
  cache?: CacheObservability;
  metadata?: Record<string, unknown>;
}

export interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  cost: number; // Cost in USD
  /** Tokens served from provider-side prefix/prompt cache (Groq, Cerebras, OpenAI automatic). */
  cachedInputTokens?: number;
  /** Anthropic: tokens read from a cache_control breakpoint (cache hit). */
  cacheReadInputTokens?: number;
  /** Anthropic: tokens written to a new cache_control breakpoint (cache miss/create). */
  cacheCreationInputTokens?: number;
  /** Provider-agnostic alias for cache write/create tokens. */
  cacheWriteInputTokens?: number;
}

export interface LLMProvider {
  name: string;
  models: string[];
  supportsStreaming: boolean;
  supportsTools: boolean;
  supportsBatching: boolean;
  supportsVision?: boolean;

  generateResponse(request: LLMRequest): Promise<LLMResponse>;
  streamResponse?(request: LLMRequest): Promise<ReadableStream<string>>;
  getProviderBalance?(): Promise<ProviderBalance>;
  validateConfig(): boolean;
  getModels(): string[];
  estimateCost(request: LLMRequest): number;
  healthCheck(): Promise<boolean>;
  getMetrics(): ProviderMetrics;
  resetMetrics(): void;
}

export interface LLMConfig {
  provider: 'openai' | 'anthropic' | 'cloudflare' | 'cerebras' | 'groq' | 'nvidia' | 'auto';
  model: string;
  temperature: number;
  maxTokens: number;
  apiKey?: string;
  baseUrl?: string;
  fallbackProvider?: 'openai' | 'anthropic' | 'cloudflare' | 'cerebras' | 'groq' | 'nvidia';
  fallbackModel?: string;
  timeout?: number;
  retries?: number;
  retryDelay?: number;
}

export interface ProviderConfig {
  apiKey?: string;
  baseUrl?: string;
  timeout?: number;
  maxRetries?: number;
  retryDelay?: number;
  rateLimitRpm?: number;
  rateLimitRpd?: number;
  organization?: string;
  project?: string;
  logger?: Logger;
  hooks?: ObservabilityHooks;
}

export interface OpenAIConfig extends ProviderConfig {
  organization?: string;
  project?: string;
  /** Route through Cloudflare AI Gateway. Ignored when `baseUrl` is set. */
  cfGateway?: CfGatewayConfig;
}

export interface AnthropicConfig extends ProviderConfig {
  version?: string;
  /** Route through Cloudflare AI Gateway. Ignored when `baseUrl` is set. */
  cfGateway?: CfGatewayConfig;
}

export interface CloudflareAIGatewayOptions {
  /** AI Gateway id for Workers AI binding calls, e.g. "default" or a custom gateway id. */
  id: string;
  /** Workers AI binding Gateway response-cache key. */
  cacheKey?: string;
  /** Workers AI binding Gateway response-cache TTL in seconds. */
  cacheTtl?: number;
  /** Workers AI binding Gateway response-cache bypass flag. */
  skipCache?: boolean;
  /** Gateway metadata visible in AI Gateway logs. */
  metadata?: Record<string, number | string | boolean | null | bigint>;
  collectLog?: boolean;
  eventId?: string;
  requestTimeoutMs?: number;
}

export interface CloudflareConfig extends ProviderConfig {
  accountId?: string;
  ai?: Ai; // Cloudflare AI binding
  /** Workers AI binding Gateway options passed as the third env.AI.run() argument. */
  gateway?: CloudflareAIGatewayOptions;
}

export interface CerebrasConfig extends ProviderConfig {
  // Cerebras uses OpenAI-compatible API; no extra fields needed beyond ProviderConfig
  /** Route through Cloudflare AI Gateway. Ignored when `baseUrl` is set. */
  cfGateway?: CfGatewayConfig;
}

export interface GroqConfig extends ProviderConfig {
  // Groq uses OpenAI-compatible API; no extra fields needed beyond ProviderConfig
  /** Route through Cloudflare AI Gateway. Ignored when `baseUrl` is set. */
  cfGateway?: CfGatewayConfig;
}

export interface NvidiaConfig extends ProviderConfig {
  // NVIDIA NIM uses OpenAI-compatible API; no extra fields needed beyond ProviderConfig
  /** Route through Cloudflare AI Gateway. Ignored when `baseUrl` is set. */
  cfGateway?: CfGatewayConfig;
}

export interface LLMError extends Error {
  code: string;
  provider: string;
  retryable: boolean;
  statusCode?: number;
  rateLimited?: boolean;
  quotaExceeded?: boolean;
}

export interface CircuitBreakerConfig {
  failureThreshold: number;
  resetTimeout: number;
  monitoringPeriod: number;
  minRequests?: number;
  degradationCurve?: number[];
}

export interface CircuitBreakerState {
  state: 'CLOSED' | 'DEGRADED' | 'RECOVERING' | 'OPEN';
  failures: number;
  consecutiveFailures: number;
  primaryTrafficPct: number;
  totalFailures: number;
  totalSuccesses: number;
  totalRequests: number;
  lastFailure?: number;
  lastSuccess?: number;
  lastRequest?: number;
  nextAttempt?: number;
}

export interface RetryConfig {
  maxRetries: number;
  initialDelay: number;
  maxDelay: number;
  backoffMultiplier: number;
  retryableErrors: string[];
}

export interface CostConfig {
  inputTokenCost: number; // Cost per 1k input tokens
  outputTokenCost: number; // Cost per 1k output tokens
  maxMonthlyCost?: number;
  alertThreshold?: number;
}

export interface ProviderMetrics {
  requestCount: number;
  successCount: number;
  errorCount: number;
  averageLatency: number;
  totalCost: number;
  rateLimitHits: number;
  lastUsed: number;
}

export interface FallbackRule {
  condition: 'error' | 'rate_limit' | 'cost' | 'latency';
  threshold?: number;
  fallbackProvider: string;
  fallbackModel?: string;
}

export interface ModelCapabilities {
  maxContextLength: number;
  supportsStreaming: boolean;
  supportsTools: boolean;
  supportsVision?: boolean;
  toolCalling?: boolean;
  supportsBatching: boolean;
  inputTokenCost: number;
  outputTokenCost: number;
  description: string;
  /** Provider-side prompt/prefix caching supported for this model. */
  supportsPromptCache?: boolean;
  /**
   * Provider-hosted built-in tools this model can run server-side, normalized
   * to `BuiltInToolType`. Absent/empty means the model is function-tools only.
   * Drives capability-aware routing and boundary gating for `LLMRequest.builtInTools`.
   */
  supportsBuiltInTools?: BuiltInToolType[];
  /**
   * True for models that output chain-of-thought reasoning traces as part of their
   * response (e.g. QwQ, DeepSeek-R1, GLM-4.7-Flash). These models are unsuitable
   * for direct-response routing (summary, classification, chat) unless the caller
   * explicitly handles reasoning traces. Routers must exclude thinking models from
   * all non-RESEARCH use-case pools.
   */
  thinkingModel?: boolean;
}

export interface ProviderCapabilities {
  models: Record<string, ModelCapabilities>;
  supportsStreaming: boolean;
  supportsTools: boolean;
  supportsBatching: boolean;
  maxBatchSize?: number;
  rateLimits: {
    requestsPerMinute: number;
    requestsPerDay: number;
    tokensPerMinute: number;
  };
}

// Streaming support
export interface StreamChunk {
  id: string;
  content: string;
  delta: string;
  finishReason?: 'stop' | 'length' | 'tool_calls' | 'content_filter';
  usage?: Partial<TokenUsage>;
}

export interface StreamResponse {
  stream: ReadableStream<StreamChunk>;
  controller: ReadableStreamDefaultController<StreamChunk>;
}

export interface QuotaCheckInput {
  tenantId?: string;
  provider: string;
  model: string;
  estimatedCost: number;
  metadata?: Record<string, unknown>;
}

export interface QuotaCheckResult {
  allowed: boolean;
  reason?: string;
  remainingBudget?: number;
}

export interface QuotaRecordInput {
  tenantId?: string;
  provider: string;
  model: string;
  actualCost: number;
  inputTokens?: number;
  outputTokens?: number;
  metadata?: Record<string, unknown>;
}

export interface QuotaHook {
  check(input: QuotaCheckInput): Promise<QuotaCheckResult>;
  record(input: QuotaRecordInput): Promise<void>;
}

export interface ToolExecutor {
  execute(name: string, argumentsValue: unknown): Promise<unknown>;
}

export interface ToolLoopState {
  iteration: number;
  cumulativeCost: number;
  messageCount: number;
  lastToolCalls: ToolCall[];
}

export interface ToolLoopAbortSignal {
  abort: true;
  reason?: string;
}

export interface ToolLoopOptions {
  maxIterations?: number;
  maxCostUSD?: number;
  /**
   * Called after each iteration. Return `{ abort: true, reason? }` to stop the
   * loop immediately and throw `ToolLoopAbortedError`; return void to continue.
   */
  onIteration?: (
    iteration: number,
    state: ToolLoopState
  ) => void | ToolLoopAbortSignal | Promise<void | ToolLoopAbortSignal>;
  abortSignal?: AbortSignal;
}

export interface ClassifyOptions<T = unknown> {
  schema?: Record<string, unknown> | { parse(data: unknown): T };
  systemPrompt?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  confidenceField?: string;
  seed?: number;
}

export interface ClassifyResult<T = unknown> {
  data: T;
  confidence?: number;
  response: LLMResponse;
}

export interface AnalyzeImageInput {
  image: LLMImageInput;
  prompt: string;
  model?: string;
  systemPrompt?: string;
  temperature?: number;
  maxTokens?: number;
  response_format?: LLMRequest['response_format'];
  tenantId?: string;
  requestId?: string;
  metadata?: Record<string, unknown>;
}

export interface RateLimitBalance {
  limit?: number;
  used?: number;
  remaining?: number;
}

export interface ProviderBalance {
  provider: string;
  status: 'available' | 'unavailable' | 'error';
  source: 'provider_api' | 'ledger' | 'headers' | 'not_supported';
  currentSpend?: number;
  monthlyBudget?: number;
  remainingBudget?: number;
  usedTokens?: number;
  requestCount?: number;
  rateLimits?: Record<string, RateLimitBalance>;
  resetAt?: string;
  message?: string;
  raw?: unknown;
}

// Batch processing
export interface BatchRequest {
  id: string;
  request: LLMRequest;
  priority?: 'low' | 'normal' | 'high';
}

export interface BatchResponse {
  id: string;
  response?: LLMResponse;
  error?: LLMError;
  processingTime: number;
}

export interface BatchJob {
  id: string;
  requests: BatchRequest[];
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  createdAt: number;
  completedAt?: number;
  results: BatchResponse[];
}
