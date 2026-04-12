/**
 * Base Provider
 * Abstract base class for all LLM providers with common functionality
 */

import type {
  LLMProvider,
  LLMImageInput,
  LLMRequest,
  LLMResponse,
  ProviderConfig,
  ModelCapabilities,
  ProviderMetrics,
  ToolCall
} from '../types';
import type { Logger } from '../utils/logger';
import { noopLogger } from '../utils/logger';
import { RetryManager } from '../utils/retry';
import { CircuitBreaker, defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { CostTracker } from '../utils/cost-tracker';
import { defaultLatencyHistogram } from '../utils/latency-histogram';
import { ConfigurationError, TimeoutError, InvalidRequestError } from '../errors';

export abstract class BaseProvider implements LLMProvider {
  abstract name: string;
  abstract models: string[];
  abstract supportsStreaming: boolean;
  abstract supportsTools: boolean;
  abstract supportsBatching: boolean;

  protected config: ProviderConfig;
  protected logger: Logger;
  protected retryManager: RetryManager;
  protected circuitBreaker: CircuitBreaker;
  protected costTracker: CostTracker;
  protected metrics: ProviderMetrics;

  constructor(config: ProviderConfig = {}) {
    this.config = {
      timeout: config.timeout ?? 30000,
      maxRetries: config.maxRetries ?? 3,
      retryDelay: config.retryDelay ?? 1000,
      ...config
    };

    this.logger = config.logger ?? noopLogger;

    this.retryManager = new RetryManager({
      maxRetries: this.config.maxRetries,
      initialDelay: this.config.retryDelay
    }, this.logger);

    // Note: this.name is set by the subclass after super() returns.
    // The circuit breaker name is updated lazily on first use.
    this.circuitBreaker = new CircuitBreaker('pending', {}, this.logger);
    this.costTracker = new CostTracker({}, undefined, this.logger);

    this.metrics = {
      requestCount: 0,
      successCount: 0,
      errorCount: 0,
      averageLatency: 0,
      totalCost: 0,
      rateLimitHits: 0,
      lastUsed: 0
    };
  }

  /**
   * Abstract method that must be implemented by each provider
   */
  abstract generateResponse(request: LLMRequest): Promise<LLMResponse>;

  /**
   * Abstract method for configuration validation
   */
  abstract validateConfig(): boolean;

  /**
   * Abstract method to get available models
   */
  abstract getModels(): string[];

  /**
   * Abstract method to estimate cost
   */
  abstract estimateCost(request: LLMRequest): number;

  /**
   * Abstract method for health check
   */
  abstract healthCheck(): Promise<boolean>;

  /**
   * Common HTTP request method with timeout and error handling
   */
  protected async makeRequest(
    url: string,
    options: RequestInit = {},
    timeoutMs?: number
  ): Promise<Response> {
    const timeout = timeoutMs || this.config.timeout || 30000;
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'ai-platform/llm-providers',
          ...options.headers
        }
      });

      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error instanceof Error && error.name === 'AbortError') {
        throw new TimeoutError(this.name, `Request timeout after ${timeout}ms`);
      }
      
      throw error;
    }
  }

  /**
   * Execute request with circuit breaker and retry logic
   */
  protected async executeWithResiliency<T>(
    operation: () => Promise<T>
  ): Promise<T> {
    return this.getCircuitBreaker().execute(
      () => this.retryManager.execute(operation)
    );
  }

  /**
   * Update metrics after request
   */
  protected updateMetrics(
    responseTime: number,
    success: boolean,
    cost: number = 0
  ): void {
    const measuredLatency = Math.max(responseTime, 1);

    this.metrics.requestCount++;
    this.metrics.lastUsed = Date.now();

    if (success) {
      this.metrics.successCount++;
    } else {
      this.metrics.errorCount++;
    }

    // Update average latency
    const totalLatency = this.metrics.averageLatency * (this.metrics.requestCount - 1);
    this.metrics.averageLatency = (totalLatency + measuredLatency) / this.metrics.requestCount;

    this.metrics.totalCost += cost;

    // Record to latency histogram for percentile tracking
    defaultLatencyHistogram.record(this.name, measuredLatency);
  }

  /**
   * Get provider metrics
   */
  getMetrics(): ProviderMetrics {
    return { ...this.metrics };
  }

  /**
   * Reset metrics
   */
  resetMetrics(): void {
    this.metrics = {
      requestCount: 0,
      successCount: 0,
      errorCount: 0,
      averageLatency: 0,
      totalCost: 0,
      rateLimitHits: 0,
      lastUsed: 0
    };
  }

  /**
   * Get provider health status
   */
  getHealth(): {
    healthy: boolean;
    circuitBreakerState: string;
    metrics: ProviderMetrics;
    lastError?: number;
  } {
    const circuitState = this.getCircuitBreaker().getState();
    const successRate = this.metrics.requestCount > 0 
      ? this.metrics.successCount / this.metrics.requestCount 
      : 1;

    return {
      healthy: circuitState.state === 'CLOSED' && successRate > 0.8,
      circuitBreakerState: circuitState.state,
      metrics: this.getMetrics(),
      lastError: circuitState.lastFailure
    };
  }

  /**
   * Update provider configuration
   */
  updateConfig(config: Partial<ProviderConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration (without sensitive data)
   */
  getConfig(): Omit<ProviderConfig, 'apiKey'> {
    const { apiKey, ...safeConfig } = this.config;
    return safeConfig;
  }

  /**
   * Providers share the named singleton breaker so factory-level routing and
   * per-provider execution observe the same failure history.
   */
  protected getCircuitBreaker(): CircuitBreaker {
    if (this.circuitBreaker.name !== this.name) {
      this.circuitBreaker = defaultCircuitBreakerManager.getBreaker(this.name);
    }

    return this.circuitBreaker;
  }

  /**
   * Common model capability definitions
   */
  protected getModelCapabilities(): Record<string, ModelCapabilities> {
    // Override in subclasses to provide model-specific capabilities
    return {};
  }

  /**
   * Validate request before processing
   */
  protected validateRequest(request: LLMRequest): void {
    if (!request.messages || request.messages.length === 0) {
      throw new ConfigurationError(this.name, 'Request must contain at least one message');
    }

    if (request.maxTokens && request.maxTokens < 1) {
      throw new ConfigurationError(this.name, 'maxTokens must be greater than 0');
    }

    if (request.temperature && (request.temperature < 0 || request.temperature > 2)) {
      throw new ConfigurationError(this.name, 'temperature must be between 0 and 2');
    }

    // Validate model if specified
    if (request.model && !this.models.includes(request.model)) {
      throw new ConfigurationError(
        this.name,
        `Model '${request.model}' not supported. Available models: ${this.models.join(', ')}`
      );
    }

    if (request.images) {
      for (const image of request.images) {
        this.validateImageInput(image);
      }
    }
  }

  protected estimateTextTokens(request: LLMRequest): number {
    return (
      (request.systemPrompt ? Math.ceil(request.systemPrompt.length / 4) : 0) +
      request.messages.reduce((sum, msg) => sum + Math.ceil(msg.content.length / 4), 0)
    );
  }

  protected validateImageInput(image: LLMImageInput): void {
    if (!image.data && !image.url) {
      throw new ConfigurationError(this.name, 'Image input must include data or url');
    }

    if (image.data && !image.mimeType) {
      throw new ConfigurationError(this.name, 'Base64 image input must include mimeType');
    }
  }

  protected getAIGatewayHeaders(request?: LLMRequest): Record<string, string> {
    const baseUrl = typeof this.config.baseUrl === 'string' ? this.config.baseUrl : '';
    if (!baseUrl.includes('gateway.ai.cloudflare.com') || !request?.gatewayMetadata) {
      return {};
    }

    const headers: Record<string, string> = {};
    const metadata = {
      ...(request.gatewayMetadata.customMetadata ?? {}),
      ...(request.gatewayMetadata.requestId ? { requestId: request.gatewayMetadata.requestId } : {}),
      ...(request.requestId ? { llmRequestId: request.requestId } : {}),
      ...(request.tenantId ? { tenantId: request.tenantId } : {}),
    };

    if (Object.keys(metadata).length > 0) {
      headers['cf-aig-metadata'] = JSON.stringify(metadata);
    }
    if (request.gatewayMetadata.cacheKey) {
      headers['cf-aig-cache-key'] = request.gatewayMetadata.cacheKey;
    }
    if (typeof request.gatewayMetadata.cacheTtl === 'number') {
      headers['cf-aig-cache-ttl'] = String(request.gatewayMetadata.cacheTtl);
    }

    return headers;
  }

  /**
   * Validate and sanitize tool calls returned by the provider.
   *
   * Ensures each tool call has the required fields (`id`, `type`, `function.name`,
   * `function.arguments`) and that their types are correct. Malformed entries are
   * dropped and logged rather than propagated to the caller.
   *
   * Returns `undefined` when the input is empty or all entries are invalid, so the
   * result can be assigned directly to `response.toolCalls`.
   */
  protected validateToolCalls(toolCalls: ToolCall[] | undefined): ToolCall[] | undefined {
    if (!toolCalls || toolCalls.length === 0) {
      return undefined;
    }

    const valid: ToolCall[] = [];

    for (let i = 0; i < toolCalls.length; i++) {
      const tc = toolCalls[i];

      // Must be an object
      if (tc == null || typeof tc !== 'object') {
        this.logger.warn(`[${this.name}] Dropping tool_call[${i}]: not an object`);
        continue;
      }

      // id — must be a non-empty string
      if (typeof tc.id !== 'string' || tc.id.length === 0) {
        this.logger.warn(`[${this.name}] Dropping tool_call[${i}]: missing or empty id`);
        continue;
      }

      // type — must be 'function'
      if (tc.type !== 'function') {
        this.logger.warn(`[${this.name}] Dropping tool_call[${i}]: invalid type "${String(tc.type)}"`);
        continue;
      }

      // function — must be an object with name and arguments
      if (tc.function == null || typeof tc.function !== 'object') {
        this.logger.warn(`[${this.name}] Dropping tool_call[${i}]: missing function object`);
        continue;
      }

      if (typeof tc.function.name !== 'string' || tc.function.name.length === 0) {
        this.logger.warn(`[${this.name}] Dropping tool_call[${i}]: missing or empty function.name`);
        continue;
      }

      if (typeof tc.function.arguments !== 'string') {
        this.logger.warn(`[${this.name}] Dropping tool_call[${i}]: function.arguments is not a string`);
        continue;
      }

      valid.push({
        id: tc.id,
        type: 'function',
        function: {
          name: tc.function.name,
          arguments: tc.function.arguments,
        },
      });
    }

    return valid.length > 0 ? valid : undefined;
  }

  /**
   * Calculate token usage cost
   */
  protected calculateCost(
    inputTokens: number,
    outputTokens: number,
    model: string
  ): number {
    const capabilities = this.getModelCapabilities()[model];
    if (!capabilities) return 0;

    const inputCost = (inputTokens / 1000) * capabilities.inputTokenCost;
    const outputCost = (outputTokens / 1000) * capabilities.outputTokenCost;

    return inputCost + outputCost;
  }

  /**
   * Common response formatting
   */
  protected buildResponse(
    content: string,
    usage: { inputTokens: number; outputTokens: number },
    model: string,
    responseTime: number,
    metadata?: Record<string, unknown>
  ): LLMResponse {
    const cost = this.calculateCost(usage.inputTokens, usage.outputTokens, model);

    return {
      message: content,
      content,
      usage: {
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        totalTokens: usage.inputTokens + usage.outputTokens,
        cost
      },
      model,
      provider: this.name,
      responseTime,
      finishReason: 'stop',
      metadata
    };
  }

  /**
   * Log request/response for debugging
   */
  protected logRequest(request: LLMRequest, response?: LLMResponse, error?: Error): void {
    const logData = {
      provider: this.name,
      model: request.model,
      messageCount: request.messages.length,
      requestId: request.requestId,
      tenantId: request.tenantId,
      success: !error,
      responseTime: response?.responseTime,
      usage: response?.usage,
      error: error?.message
    };

    if (error) {
      this.logger.error(`[${this.name}] Request failed:`, logData);
    } else {
      this.logger.debug(`[${this.name}] Request completed:`, logData);
    }
  }
}
