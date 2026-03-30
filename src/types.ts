/**
 * LLM Provider Types
 * Unified types for all LLM providers with v2 architecture support
 */

export interface LLMMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
  toolCalls?: ToolCall[];
  toolResults?: ToolResult[];
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
  tools?: Tool[];
  toolChoice?: 'auto' | 'none' | { type: 'function'; function: { name: string } };
  tenantId?: string;
  requestId?: string;
  metadata?: Record<string, any>;
}

export interface Tool {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: {
      type: 'object';
      properties: Record<string, any>;
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
  metadata?: Record<string, any>;
}

export interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  cost?: number; // Cost in USD
}

export interface LLMProvider {
  name: string;
  models: string[];
  supportsStreaming: boolean;
  supportsTools: boolean;
  supportsBatching: boolean;

  generateResponse(request: LLMRequest): Promise<LLMResponse>;
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
  toolCalling?: boolean;
  supportsBatching: boolean;
  inputTokenCost: number;
  outputTokenCost: number;
  description: string;
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
