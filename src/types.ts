/**
 * LLM Provider Types
 * Unified types for all LLM providers with v2 architecture support
 */

import type { Logger } from './utils/logger';
import type { ObservabilityHooks } from './utils/hooks';
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
  customMetadata?: Record<string, string>;
}

/**
 * Provider prompt/prefix cache hints. Distinct from GatewayMetadata.cacheKey/cacheTtl,
 * which control Cloudflare AI Gateway *response* caching. These hints control
 * provider-side prefix/prompt caching (Anthropic cache_control, Groq/Cerebras automatic, etc.).
 */
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
  toolChoice?: 'auto' | 'none' | { type: 'function'; function: { name: string } };
  response_format?: { type: 'json_object' | 'text' };
  seed?: number;
  gatewayMetadata?: GatewayMetadata;
  cache?: CacheHints;
  /**
   * Cloudflare Workers AI only: LoRA adapter name or UUID to apply at inference.
   * Ignored by all other providers. The package does not validate the identifier;
   * it is forwarded as-is to the Workers AI binding.
   */
  lora?: string;
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

export interface LLMResponse {
  id?: string;
  message: string;
  content?: string; // Alternative to message for consistency
  usage: TokenUsage;
  model: string;
  provider: string;
  responseTime: number;
  finishReason?: 'stop' | 'length' | 'tool_calls' | 'content_filter';
  toolCalls?: ToolCall[];
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
  provider: 'openai' | 'anthropic' | 'cloudflare' | 'cerebras' | 'groq' | 'auto';
  model: string;
  temperature: number;
  maxTokens: number;
  apiKey?: string;
  baseUrl?: string;
  fallbackProvider?: 'openai' | 'anthropic' | 'cloudflare' | 'cerebras' | 'groq';
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
}

export interface AnthropicConfig extends ProviderConfig {
  version?: string;
}

export interface CloudflareConfig extends ProviderConfig {
  accountId?: string;
  ai?: Ai; // Cloudflare AI binding
}

export interface CerebrasConfig extends ProviderConfig {
  // Cerebras uses OpenAI-compatible API; no extra fields needed beyond ProviderConfig
}

export interface GroqConfig extends ProviderConfig {
  // Groq uses OpenAI-compatible API; no extra fields needed beyond ProviderConfig
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

export interface ToolLoopOptions {
  maxIterations?: number;
  maxCostUSD?: number;
  onIteration?: (iteration: number, state: ToolLoopState) => void | Promise<void>;
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
