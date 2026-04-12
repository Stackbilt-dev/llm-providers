/**
 * LLM Provider Factory
 * Creates and manages LLM provider instances with intelligent fallback logic
 */

import type {
  LLMProvider,
  LLMConfig,
  LLMRequest,
  LLMResponse,
  AnalyzeImageInput,
  ClassifyOptions,
  ClassifyResult,
  ProviderConfig,
  OpenAIConfig,
  AnthropicConfig,
  CloudflareConfig,
  CerebrasConfig,
  GroqConfig,
  FallbackRule,
  ProviderMetrics,
  CircuitBreakerState,
  ProviderBalance,
  QuotaHook,
  QuotaCheckInput,
  QuotaRecordInput,
  ToolExecutor,
  ToolLoopOptions,
  ToolLoopState
} from './types';

import type { Logger } from './utils/logger';
import { noopLogger } from './utils/logger';
import type { ObservabilityHooks } from './utils/hooks';
import { noopHooks } from './utils/hooks';
import { OpenAIProvider } from './providers/openai';
import { AnthropicProvider } from './providers/anthropic';
import { CloudflareProvider } from './providers/cloudflare';
import { CerebrasProvider } from './providers/cerebras';
import { GroqProvider } from './providers/groq';
import { CostTracker, defaultCostTracker } from './utils/cost-tracker';
import type { ProviderCostBreakdownEntry } from './utils/cost-tracker';
import type { CreditLedger } from './utils/credit-ledger';
import { defaultCircuitBreakerManager } from './utils/circuit-breaker';
import { defaultExhaustionRegistry } from './utils/exhaustion';
import { defaultLatencyHistogram } from './utils/latency-histogram';
import {
  LLMProviderError,
  ConfigurationError,
  CircuitBreakerOpenError,
  AuthenticationError,
  RateLimitError,
  QuotaExceededError,
  ToolLoopAbortedError,
  ToolLoopLimitError,
} from './errors';

export interface ProviderFactoryConfig {
  openai?: OpenAIConfig;
  anthropic?: AnthropicConfig;
  cloudflare?: CloudflareConfig;
  cerebras?: CerebrasConfig;
  groq?: GroqConfig;
  defaultProvider?: 'openai' | 'anthropic' | 'cloudflare' | 'cerebras' | 'groq' | 'auto';
  fallbackRules?: FallbackRule[];
  costOptimization?: boolean;
  enableCircuitBreaker?: boolean;
  enableRetries?: boolean;
  ledger?: CreditLedger;
  quotaHook?: QuotaHook;
  quotaFailPolicy?: 'closed' | 'open';
  defaultVisionModel?: string;
  logger?: Logger;
  hooks?: ObservabilityHooks;
}

export interface CostAnalytics {
  breakdown?: Record<string, ProviderCostBreakdownEntry>;
  total?: number;
  recommendations?: string[];
  message?: string;
}

export interface ProviderHealthEntry {
  healthy: boolean;
  metrics?: ProviderMetrics;
  circuitBreaker?: CircuitBreakerState | null;
  models?: string[];
  capabilities?: {
    streaming: boolean;
    tools: boolean;
    batching: boolean;
  };
  error?: string;
}

interface FallbackDecision {
  shouldFallback: boolean;
  fallbackProvider?: string;
  fallbackModel?: string;
}

export class LLMProviderFactory {
  private providers: Map<string, LLMProvider> = new Map();
  private config: ProviderFactoryConfig;
  private costTracker: CostTracker;
  private fallbackRules: FallbackRule[];
  private logger: Logger;
  private hooks: ObservabilityHooks;

  constructor(config: ProviderFactoryConfig) {
    this.config = config;
    this.logger = config.logger ?? noopLogger;
    this.hooks = config.hooks ?? noopHooks;
    this.costTracker = config.ledger
      ? new CostTracker({}, config.ledger, this.logger)
      : defaultCostTracker;
    this.fallbackRules = config.fallbackRules || this.getDefaultFallbackRules();

    this.initializeProviders();
  }

  /**
   * Initialize all configured providers
   */
  private initializeProviders(): void {
    const providerEntries: Array<[string, new (config: ProviderConfig) => LLMProvider]> = [
      ['openai', OpenAIProvider],
      ['anthropic', AnthropicProvider],
      ['cloudflare', CloudflareProvider],
      ['cerebras', CerebrasProvider],
      ['groq', GroqProvider],
    ];

    for (const [name, ProviderClass] of providerEntries) {
      const providerConfig = this.config[name as keyof ProviderFactoryConfig] as ProviderConfig | undefined;
      if (!providerConfig) continue;

      try {
        const retryConfig: Partial<ProviderConfig> =
          this.config.enableRetries === false && providerConfig.maxRetries === undefined
            ? { maxRetries: 0 }
            : {};
        const provider = new ProviderClass({
          ...providerConfig,
          ...retryConfig,
          logger: this.logger,
          hooks: this.hooks,
        });
        if (provider.validateConfig()) {
          this.providers.set(name, provider);
          this.logger.info(`[LLMProviderFactory] ${name} provider initialized`);
        }
      } catch (error) {
        this.logger.warn(`[LLMProviderFactory] Failed to initialize ${name} provider:`, (error as Error).message);
      }
    }

    if (this.providers.size === 0) {
      throw new ConfigurationError('factory', 'No valid providers configured');
    }
  }

  /**
   * Generate response with intelligent provider selection and fallback
   */
  async generateResponse(request: LLMRequest): Promise<LLMResponse> {
    const providerChain = this.buildProviderChain(request);
    const providerModels = new Map<string, string>();
    let lastError: Error | null = null;
    let previousProvider: string | null = null;

    for (let index = 0; index < providerChain.length; index++) {
      const providerName = providerChain[index];

      try {
        const provider = this.providers.get(providerName);
        if (!provider) continue;

        // Check exhaustion registry
        if (defaultExhaustionRegistry.isExhausted(providerName)) {
          this.logger.warn(`[LLMProviderFactory] Provider ${providerName} is quota-exhausted, skipping`);
          continue;
        }

        // Check circuit breaker
        if (this.config.enableCircuitBreaker) {
          const breaker = defaultCircuitBreakerManager.getBreaker(providerName);
          if (breaker.isOpen()) {
            this.logger.warn(`[LLMProviderFactory] Circuit breaker open for ${providerName}, skipping`);
            continue;
          }
        }

        if (this.config.ledger && this.isLedgerLimited(providerName)) {
          continue;
        }

        // Emit fallback event if this isn't the first provider attempted
        if (previousProvider && lastError) {
          this.hooks.onFallback?.({
            fromProvider: previousProvider,
            toProvider: providerName,
            requestId: request.requestId,
            reason: lastError.message,
            errorCode: (lastError as { code?: string }).code,
            timestamp: Date.now(),
          });
        }

        this.logger.debug(`[LLMProviderFactory] Trying provider: ${providerName}`);

        const providerRequest = this.requestForProvider(request, providerName, providerModels);
        const model = providerRequest.model || provider.models[0] || 'unknown';
        await this.checkQuota(providerName, provider, providerRequest, model);

        this.hooks.onRequestStart?.({
          provider: providerName,
          model,
          requestId: request.requestId,
          tenantId: request.tenantId,
          timestamp: Date.now(),
        });

        const startTime = Date.now();
        const response = await provider.generateResponse(providerRequest);
        const durationMs = Date.now() - startTime;

        this.hooks.onRequestEnd?.({
          provider: providerName,
          model: response.model,
          requestId: request.requestId,
          tenantId: request.tenantId,
          durationMs,
          usage: response.usage,
          finishReason: response.finishReason,
          timestamp: Date.now(),
        });

        // Track spend whenever analytics or ledger accounting is configured.
        if (this.config.costOptimization || this.config.ledger) {
          this.costTracker.trackCost(providerName, response);
        }
        this.recordQuota(providerName, response, providerRequest);

        this.logger.debug(`[LLMProviderFactory] Successfully used provider: ${providerName}`);
        return response;

      } catch (error) {
        const err = error as Error;
        lastError = err;
        previousProvider = providerName;
        this.logger.warn(`[LLMProviderFactory] Provider ${providerName} failed:`, err.message);

        this.hooks.onRequestError?.({
          provider: providerName,
          model: request.model || 'unknown',
          requestId: request.requestId,
          tenantId: request.tenantId,
          error: err,
          errorCode: (err as { code?: string }).code,
          attempt: 1,
          willRetry: this.shouldFallback(err),
          timestamp: Date.now(),
        });

        // Auto-mark quota-exhausted providers
        if (err instanceof QuotaExceededError) {
          defaultExhaustionRegistry.markExhausted(providerName);
          this.hooks.onQuotaExhausted?.({
            provider: providerName,
            resetAfterMs: defaultExhaustionRegistry.defaultResetMs,
            timestamp: Date.now(),
          });
        }

        const fallbackDecision = this.getFallbackDecision(error as Error);
        if (!fallbackDecision.shouldFallback) {
          throw error;
        }

        this.applyFallbackDecision(
          fallbackDecision,
          providerName,
          providerChain,
          index,
          providerModels
        );
      }
    }

    // All providers failed
    throw lastError || new LLMProviderError(
      'All providers failed',
      'ALL_PROVIDERS_FAILED',
      'factory',
      false
    );
  }

  async generateResponseStream(request: LLMRequest): Promise<ReadableStream<string>> {
    const providerChain = this.buildProviderChain({ ...request, stream: true });
    const providerModels = new Map<string, string>();
    let lastError: Error | null = null;
    let previousProvider: string | null = null;

    for (let index = 0; index < providerChain.length; index++) {
      const providerName = providerChain[index];
      try {
        const provider = this.providers.get(providerName);
        if (!provider || !provider.supportsStreaming || !provider.streamResponse) continue;
        if (defaultExhaustionRegistry.isExhausted(providerName)) continue;
        if (this.config.enableCircuitBreaker && defaultCircuitBreakerManager.getBreaker(providerName).isOpen()) continue;
        if (this.config.ledger && this.isLedgerLimited(providerName)) continue;

        if (previousProvider && lastError) {
          this.hooks.onFallback?.({
            fromProvider: previousProvider,
            toProvider: providerName,
            requestId: request.requestId,
            reason: lastError.message,
            errorCode: (lastError as { code?: string }).code,
            timestamp: Date.now(),
          });
        }

        const providerRequest = {
          ...this.requestForProvider(request, providerName, providerModels),
          stream: true
        };
        const model = providerRequest.model || provider.models[0] || 'unknown';
        const estimatedCost = await this.checkQuota(providerName, provider, providerRequest, model);

        this.hooks.onRequestStart?.({
          provider: providerName,
          model,
          requestId: request.requestId,
          tenantId: request.tenantId,
          timestamp: Date.now(),
        });

        const startTime = Date.now();
        const opened = await this.openStreamWithFirstChunk(provider, providerRequest);
        return this.buildFactoryStream(
          opened.reader,
          opened.firstChunk,
          opened.done,
          providerName,
          model,
          providerRequest,
          startTime,
          estimatedCost
        );
      } catch (error) {
        const err = error as Error;
        lastError = err;
        previousProvider = providerName;

        this.hooks.onRequestError?.({
          provider: providerName,
          model: request.model || 'unknown',
          requestId: request.requestId,
          tenantId: request.tenantId,
          error: err,
          errorCode: (err as { code?: string }).code,
          attempt: 1,
          willRetry: this.shouldFallback(err),
          timestamp: Date.now(),
        });

        const fallbackDecision = this.getFallbackDecision(err);
        if (!fallbackDecision.shouldFallback) {
          throw error;
        }

        this.applyFallbackDecision(fallbackDecision, providerName, providerChain, index, providerModels);
      }
    }

    throw lastError || new LLMProviderError(
      'All streaming providers failed',
      'ALL_PROVIDERS_FAILED',
      'factory',
      false
    );
  }

  async generateResponseWithTools(
    request: LLMRequest,
    toolExecutor: ToolExecutor,
    opts: ToolLoopOptions = {}
  ): Promise<LLMResponse> {
    const maxIterations = opts.maxIterations ?? 10;
    let cumulativeCost = 0;
    let messages = [...request.messages];

    let lastResponseCost = 0;

    for (let iteration = 0; iteration <= maxIterations; iteration++) {
      if (opts.abortSignal?.aborted) {
        throw new ToolLoopAbortedError('factory');
      }

      // Pre-flight cost guard: use the previous iteration's cost as an
      // estimate for the next one.  This prevents obvious overshoots where
      // a single expensive response would blow past the cap.  The cap is
      // still soft (±1 iteration tolerance) because the actual cost is
      // only known after the response.
      if (opts.maxCostUSD !== undefined && iteration > 0) {
        const projectedCost = cumulativeCost + lastResponseCost;
        if (projectedCost > opts.maxCostUSD) {
          throw new ToolLoopLimitError(
            'factory',
            `Tool loop would exceed max cost ${opts.maxCostUSD} (projected ${projectedCost.toFixed(4)})`
          );
        }
      }

      const response = await this.generateResponse({ ...request, messages });
      lastResponseCost = response.usage.cost;
      cumulativeCost += lastResponseCost;

      if (opts.maxCostUSD !== undefined && cumulativeCost > opts.maxCostUSD) {
        throw new ToolLoopLimitError(
          'factory',
          `Tool loop exceeded max cost ${opts.maxCostUSD}`
        );
      }

      if (!response.toolCalls || response.toolCalls.length === 0) {
        return {
          ...response,
          metadata: {
            ...response.metadata,
            cumulativeCost,
            toolIterations: iteration
          }
        };
      }

      if (iteration >= maxIterations) {
        throw new ToolLoopLimitError('factory', `Tool loop exceeded ${maxIterations} iterations`);
      }

      const toolResults = [];
      for (const toolCall of response.toolCalls) {
        if (opts.abortSignal?.aborted) {
          throw new ToolLoopAbortedError('factory');
        }

        let parsedArguments: unknown;
        try {
          parsedArguments = JSON.parse(toolCall.function.arguments);
        } catch {
          parsedArguments = toolCall.function.arguments;
        }

        try {
          const output = await toolExecutor.execute(toolCall.function.name, parsedArguments);
          toolResults.push({
            id: toolCall.id,
            output: typeof output === 'string' ? output : JSON.stringify(output)
          });
        } catch (error) {
          toolResults.push({
            id: toolCall.id,
            output: '',
            error: (error as Error).message
          });
        }
      }

      messages = [
        ...messages,
        {
          role: 'assistant',
          content: response.message,
          toolCalls: response.toolCalls
        },
        {
          role: 'user',
          content: '',
          toolResults
        }
      ];

      const state: ToolLoopState = {
        iteration: iteration + 1,
        cumulativeCost,
        messageCount: messages.length,
        lastToolCalls: response.toolCalls
      };
      await opts.onIteration?.(iteration + 1, state);
    }

    throw new ToolLoopLimitError('factory', `Tool loop exceeded ${maxIterations} iterations`);
  }

  async classify<T = unknown>(
    input: string | LLMRequest,
    options: ClassifyOptions<T> = {}
  ): Promise<ClassifyResult<T>> {
    const parser = options.schema && typeof (options.schema as { parse?: unknown }).parse === 'function'
      ? (options.schema as { parse(data: unknown): T }).parse
      : undefined;
    const schemaDescription = options.schema && !parser
      ? `\nJSON schema:\n${JSON.stringify(options.schema)}`
      : '';
    const systemPrompt = options.systemPrompt ||
      `Classify the input and return only valid JSON.${schemaDescription}`;
    const request: LLMRequest = typeof input === 'string'
      ? {
          messages: [{ role: 'user', content: input }],
          model: options.model,
          temperature: options.temperature ?? 0,
          maxTokens: options.maxTokens,
          response_format: { type: 'json_object' },
          systemPrompt,
          seed: options.seed
        }
      : {
          ...input,
          model: options.model ?? input.model,
          temperature: options.temperature ?? input.temperature ?? 0,
          maxTokens: options.maxTokens ?? input.maxTokens,
          response_format: { type: 'json_object' },
          systemPrompt: options.systemPrompt ?? input.systemPrompt ?? systemPrompt,
          seed: options.seed ?? input.seed
        };

    const response = await this.generateResponse(request);
    const parsed = this.parseJsonResponse(response.message);
    const data = parser ? parser(parsed) : parsed as T;
    const confidenceValue = (parsed as Record<string, unknown>)[options.confidenceField ?? 'confidence'];

    return {
      data,
      confidence: typeof confidenceValue === 'number' ? confidenceValue : undefined,
      response
    };
  }

  async analyzeImage(input: AnalyzeImageInput): Promise<LLMResponse> {
    return this.generateResponse({
      messages: [{ role: 'user', content: input.prompt }],
      images: [input.image],
      model: input.model ?? this.getDefaultVisionModel(),
      systemPrompt: input.systemPrompt,
      temperature: input.temperature,
      maxTokens: input.maxTokens,
      response_format: input.response_format,
      tenantId: input.tenantId,
      requestId: input.requestId,
      metadata: input.metadata
    });
  }

  async getProviderBalance(provider?: string): Promise<ProviderBalance | Record<string, ProviderBalance>> {
    if (provider) {
      const balance = await this.getSingleProviderBalance(provider);
      this.hooks.onProviderBalance?.({ provider, balance, timestamp: Date.now() });
      return balance;
    }

    const result: Record<string, ProviderBalance> = {};
    for (const providerName of this.providers.keys()) {
      const balance = await this.getSingleProviderBalance(providerName);
      result[providerName] = balance;
      this.hooks.onProviderBalance?.({ provider: providerName, balance, timestamp: Date.now() });
    }
    return result;
  }

  /**
   * Build provider chain based on request and configuration
   */
  private buildProviderChain(request: LLMRequest): string[] {
    const chain: string[] = [];

    // If specific provider requested, try it first
    if (request.model) {
      const providerForModel = this.getProviderForModel(request.model);
      if (providerForModel && this.providers.has(providerForModel)) {
        chain.push(providerForModel);
      }
    }

    // Add default provider if different from model provider
    const defaultProvider = this.config.defaultProvider || 'auto';
    if (defaultProvider !== 'auto' && !chain.includes(defaultProvider)) {
      if (this.providers.has(defaultProvider)) {
        chain.push(defaultProvider);
      }
    }

    // For 'auto' mode or as fallbacks, add providers by priority
    const prioritizedProviders = this.getPrioritizedProviders(request);
    for (const provider of prioritizedProviders) {
      if (!chain.includes(provider) && this.providers.has(provider)) {
        chain.push(provider);
      }
    }

    return chain;
  }

  /**
   * Get prioritized list of providers based on cost optimization and capabilities
   */
  private getPrioritizedProviders(request: LLMRequest): string[] {
    const visionOnly = (request.images?.length ?? 0) > 0;
    if (!this.config.costOptimization) {
      // Default priority: all configured providers, cheapest first
      return ['cloudflare', 'cerebras', 'groq', 'anthropic', 'openai']
        .filter(p => this.providers.has(p))
        .filter(p => !visionOnly || this.providerSupportsVision(p));
    }

    // Cost-optimized routing
    const providers = Array.from(this.providers.keys())
      .filter(p => !visionOnly || this.providerSupportsVision(p));
    const sortedProviders = [...providers].sort((a, b) => {
      const providerA = this.providers.get(a)!;
      const providerB = this.providers.get(b)!;
      const estimatedCostA = providerA.estimateCost(request);
      const estimatedCostB = providerB.estimateCost(request);

      if (estimatedCostA !== estimatedCostB) {
        return estimatedCostA - estimatedCostB;
      }

      // If estimates tie, prefer the provider with less accumulated spend.
      const trackedCostA = this.costTracker.getProviderCost(a);
      const trackedCostB = this.costTracker.getProviderCost(b);
      return trackedCostA - trackedCostB;
    });

    return sortedProviders;
  }

  /**
   * Get appropriate provider for a specific model
   */
  private getProviderForModel(model: string): string | null {
    // OpenAI models
    if (model.startsWith('gpt-')) {
      return 'openai';
    }

    // Anthropic models
    if (model.includes('claude')) {
      return 'anthropic';
    }

    // Cloudflare models
    if (model.startsWith('@cf/')) {
      return 'cloudflare';
    }

    // Groq models (openai/gpt-oss-120b is Groq-hosted, not @cf/ prefixed)
    if (model.includes('-versatile') || model.includes('-instant') || model === 'openai/gpt-oss-120b') {
      return 'groq';
    }

    // Cerebras models
    if (model.startsWith('llama-3.1-8b') || model.startsWith('llama-3.3-70b')
        || model.startsWith('zai-glm') || model.startsWith('qwen-3-235b')) {
      return 'cerebras';
    }

    return null;
  }

  /**
   * Check if we should fallback to another provider
   */
  private shouldFallback(error: Error): boolean {
    return this.getFallbackDecision(error).shouldFallback;
  }

  /**
   * Get fallback routing decision for an error.
   */
  private getFallbackDecision(error: Error): FallbackDecision {
    // Don't fallback for authentication errors
    if (error instanceof AuthenticationError) {
      return { shouldFallback: false };
    }

    // Don't fallback for configuration errors
    if (error instanceof ConfigurationError) {
      return { shouldFallback: false };
    }

    // Custom fallback rules can provide explicit provider/model routing.
    for (const rule of this.fallbackRules) {
      if (this.evaluateFallbackRule(rule, error)) {
        return {
          shouldFallback: true,
          fallbackProvider: rule.fallbackProvider,
          fallbackModel: rule.fallbackModel
        };
      }
    }

    // Fallback for circuit breaker, rate limits, and server errors
    if (error instanceof CircuitBreakerOpenError ||
        error instanceof RateLimitError) {
      return { shouldFallback: true };
    }

    if (error instanceof LLMProviderError) {
      if (error.code === 'SERVER_ERROR' ||
          error.code === 'NETWORK_ERROR' ||
          error.code === 'TIMEOUT') {
        return { shouldFallback: true };
      }
    }

    return { shouldFallback: false };
  }

  /**
   * Evaluate a fallback rule against an error
   */
  private evaluateFallbackRule(rule: FallbackRule, error: Error): boolean {
    switch (rule.condition) {
      case 'error':
        return true; // Any error triggers fallback

      case 'rate_limit':
        return error instanceof RateLimitError;

      case 'cost':
        // Check if cost threshold exceeded
        if (rule.threshold && this.config.costOptimization) {
          const totalCost = this.costTracker.getTotalCost();
          return totalCost > rule.threshold;
        }
        return false;

      case 'latency':
        // Would need to track latency to implement this
        return false;

      default:
        return false;
    }
  }

  /**
   * Get default fallback rules
   */
  private getDefaultFallbackRules(): FallbackRule[] {
    return [
      {
        condition: 'rate_limit',
        fallbackProvider: 'cloudflare' // Fallback to Cloudflare for rate limits
      },
      {
        condition: 'cost',
        threshold: 10, // $10 threshold
        fallbackProvider: 'cloudflare' // Use cheaper provider when cost is high
      },
      {
        condition: 'error',
        fallbackProvider: 'anthropic' // General error fallback
      }
    ];
  }

  /**
   * Get provider instance by name
   */
  getProvider(name: string): LLMProvider | undefined {
    return this.providers.get(name);
  }

  /**
   * Get all available providers
   */
  getAvailableProviders(): string[] {
    return Array.from(this.providers.keys());
  }

  /**
   * Get provider health status
   */
  async getProviderHealth(): Promise<Record<string, ProviderHealthEntry>> {
    const health: Record<string, ProviderHealthEntry> = {};

    for (const [name, provider] of this.providers) {
      try {
        const isHealthy = await provider.healthCheck();
        const metrics = provider.getMetrics();
        const circuitState = this.config.enableCircuitBreaker
          ? defaultCircuitBreakerManager.getBreaker(name).getState()
          : null;

        health[name] = {
          healthy: isHealthy,
          metrics,
          circuitBreaker: circuitState,
          models: provider.getModels(),
          capabilities: {
            streaming: provider.supportsStreaming,
            tools: provider.supportsTools,
            batching: provider.supportsBatching
          }
        };
      } catch (error) {
        health[name] = {
          healthy: false,
          error: (error as Error).message
        };
      }
    }

    return health;
  }

  /**
   * Get cost analytics
   */
  getCostAnalytics(): CostAnalytics {
    if (!this.config.costOptimization) {
      return { message: 'Cost optimization not enabled' };
    }

    return {
      breakdown: this.costTracker.getCostBreakdown(),
      total: this.costTracker.getTotalCost(),
      recommendations: this.getCostRecommendations()
    };
  }

  /**
   * Get cost optimization recommendations
   */
  private getCostRecommendations(): string[] {
    const breakdown = this.costTracker.getCostBreakdown();
    const recommendations: string[] = [];

    // Check for expensive providers
    const totalCost = this.costTracker.getTotalCost();
    for (const [provider, data] of Object.entries(breakdown)) {
      const percentage = (data.cost / totalCost) * 100;
      if (percentage > 60) {
        recommendations.push(
          `Consider reducing usage of ${provider} (${percentage.toFixed(1)}% of total cost)`
        );
      }
    }

    // Recommend Cloudflare for cost savings
    if (!this.providers.has('cloudflare')) {
      recommendations.push('Consider adding Cloudflare AI provider for significant cost savings');
    }

    return recommendations;
  }

  /**
   * Get latency histogram data for all providers
   */
  getLatencyHistogram(): Record<string, import('./utils/latency-histogram').LatencySummary> {
    return defaultLatencyHistogram.allSummaries();
  }

  /**
   * Get currently exhausted providers
   */
  getExhaustedProviders(): string[] {
    return defaultExhaustionRegistry.getExhaustedProviders();
  }

  /**
   * Reset all provider metrics, circuit breakers, exhaustion, and histograms
   */
  reset(): void {
    for (const [name, provider] of this.providers) {
      provider.resetMetrics();
      if (this.config.enableCircuitBreaker) {
        defaultCircuitBreakerManager.reset(name);
      }
    }

    if (this.config.costOptimization || this.config.ledger) {
      this.costTracker.reset();
    }

    defaultExhaustionRegistry.reset();
    defaultLatencyHistogram.reset();
  }

  /**
   * Update factory configuration
   */
  updateConfig(config: Partial<ProviderFactoryConfig>): void {
    this.config = { ...this.config, ...config };

    if ('ledger' in config) {
      this.costTracker = config.ledger
        ? new CostTracker({}, config.ledger, this.logger)
        : defaultCostTracker;
    }

    if (config.fallbackRules) {
      this.fallbackRules = config.fallbackRules;
    }

    // Re-initialize providers if configs changed
    if (
      config.openai ||
      config.anthropic ||
      config.cloudflare ||
      config.cerebras ||
      config.groq ||
      config.enableRetries !== undefined
    ) {
      this.providers.clear();
      this.initializeProviders();
    }
  }

  private async openStreamWithFirstChunk(
    provider: LLMProvider,
    request: LLMRequest
  ): Promise<{ reader: ReadableStreamDefaultReader<string>; firstChunk?: string; done: boolean }> {
    if (!provider.streamResponse) {
      throw new ConfigurationError(provider.name, 'Provider does not support streaming');
    }

    const stream = await provider.streamResponse(request);
    const reader = stream.getReader();
    const first = await reader.read();
    return {
      reader,
      firstChunk: first.value,
      done: first.done
    };
  }

  private buildFactoryStream(
    reader: ReadableStreamDefaultReader<string>,
    firstChunk: string | undefined,
    firstDone: boolean,
    providerName: string,
    model: string,
    request: LLMRequest,
    startTime: number,
    estimatedCost: number
  ): ReadableStream<string> {
    return new ReadableStream<string>({
      start: async (controller) => {
        try {
          if (!firstDone && firstChunk !== undefined) {
            controller.enqueue(firstChunk);
          }

          if (!firstDone) {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              if (value !== undefined) controller.enqueue(value);
            }
          }

          const usage = { inputTokens: 0, outputTokens: 0, totalTokens: 0, cost: estimatedCost };
          this.hooks.onRequestEnd?.({
            provider: providerName,
            model,
            requestId: request.requestId,
            tenantId: request.tenantId,
            durationMs: Date.now() - startTime,
            usage,
            finishReason: 'stop',
            timestamp: Date.now(),
          });
          this.recordQuotaInput({
            tenantId: request.tenantId,
            provider: providerName,
            model,
            actualCost: estimatedCost,
            metadata: request.metadata
          });
          controller.close();
        } catch (error) {
          controller.error(error);
        } finally {
          reader.releaseLock();
        }
      }
    });
  }

  private async checkQuota(
    providerName: string,
    provider: LLMProvider,
    request: LLMRequest,
    model: string
  ): Promise<number> {
    const estimatedCost = provider.estimateCost(request);
    if (!this.config.quotaHook) {
      return estimatedCost;
    }

    const input: QuotaCheckInput = {
      tenantId: request.tenantId,
      provider: providerName,
      model,
      estimatedCost,
      metadata: request.metadata
    };

    try {
      const result = await this.config.quotaHook.check(input);
      this.hooks.onQuotaCheck?.({ input, result, timestamp: Date.now() });
      if (!result.allowed) {
        this.hooks.onQuotaDenied?.({ input, reason: result.reason, timestamp: Date.now() });
        throw new QuotaExceededError(providerName, result.reason || 'Quota hook denied request');
      }
    } catch (error) {
      if (error instanceof QuotaExceededError) {
        throw error;
      }

      if ((this.config.quotaFailPolicy ?? 'closed') === 'open') {
        this.logger.warn(`[LLMProviderFactory] Quota check failed open for ${providerName}:`, (error as Error).message);
        return estimatedCost;
      }

      const reason = (error as Error).message;
      this.hooks.onQuotaDenied?.({ input, reason, timestamp: Date.now() });
      throw new QuotaExceededError(providerName, reason);
    }

    return estimatedCost;
  }

  private recordQuota(providerName: string, response: LLMResponse, request: LLMRequest): void {
    this.recordQuotaInput({
      tenantId: request.tenantId,
      provider: providerName,
      model: response.model,
      actualCost: response.usage.cost,
      inputTokens: response.usage.inputTokens,
      outputTokens: response.usage.outputTokens,
      metadata: request.metadata
    });
  }

  private recordQuotaInput(input: QuotaRecordInput): void {
    if (!this.config.quotaHook) return;

    void this.config.quotaHook.record(input).catch(error => {
      this.logger.warn(`[LLMProviderFactory] Quota record failed for ${input.provider}:`, (error as Error).message);
    });
  }

  private parseJsonResponse(message: string): unknown {
    try {
      return JSON.parse(message);
    } catch {
      // Strip markdown fences (```json ... ``` or ``` ... ```) before
      // falling back to brace extraction so fenced JSON parses cleanly.
      const fenced = message.replace(/^```(?:json)?\s*\n?/m, '').replace(/\n?```\s*$/m, '');
      try {
        return JSON.parse(fenced);
      } catch {
        // Last resort: extract outermost braces.
        const start = fenced.indexOf('{');
        const end = fenced.lastIndexOf('}');
        if (start >= 0 && end > start) {
          return JSON.parse(fenced.slice(start, end + 1));
        }
      }
      throw new ConfigurationError('factory', 'Classification response was not valid JSON');
    }
  }

  private getDefaultVisionModel(): string | undefined {
    if (this.config.defaultVisionModel) return this.config.defaultVisionModel;
    if (this.providers.has('anthropic')) return 'claude-haiku-4-5-20251001';
    if (this.providers.has('openai')) return 'gpt-4o-mini';
    return undefined;
  }

  private providerSupportsVision(providerName: string): boolean {
    return this.providers.get(providerName)?.supportsVision === true;
  }

  private async getSingleProviderBalance(providerName: string): Promise<ProviderBalance> {
    const ledgerBalance = this.getLedgerBalance(providerName);
    if (ledgerBalance) {
      return ledgerBalance;
    }

    const provider = this.providers.get(providerName);
    if (!provider) {
      return {
        provider: providerName,
        status: 'error',
        source: 'not_supported',
        message: `Provider '${providerName}' is not configured`
      };
    }

    if (provider.getProviderBalance) {
      return provider.getProviderBalance();
    }

    return {
      provider: providerName,
      status: 'unavailable',
      source: 'not_supported',
      message: `Provider '${providerName}' does not expose balance reporting`
    };
  }

  private getLedgerBalance(providerName: string): ProviderBalance | undefined {
    const acc = this.config.ledger?.getProviderAccumulator(providerName);
    if (!acc) return undefined;

    const rateLimits: ProviderBalance['rateLimits'] = {};
    for (const [dimension, window] of Object.entries(acc.rateLimits)) {
      rateLimits[dimension] = {
        limit: window.limit,
        used: window.used,
        remaining: Math.max(window.limit - window.used, 0)
      };
    }

    return {
      provider: providerName,
      status: 'available',
      source: 'ledger',
      currentSpend: acc.spend,
      monthlyBudget: acc.budget ?? undefined,
      remainingBudget: acc.budget === null ? undefined : acc.budget - acc.spend,
      usedTokens: acc.inputTokens + acc.outputTokens,
      requestCount: acc.requestCount,
      rateLimits
    };
  }

  private isLedgerLimited(providerName: string): boolean {
    if (!this.config.ledger) return false;

    for (const dimension of ['rpm', 'rpd', 'tpm', 'tpd'] as const) {
      const check = this.config.ledger.checkRateLimit(providerName, dimension);
      if (!check.allowed) {
        this.logger.warn(
          `[LLMProviderFactory] Rate limit (${dimension}) exceeded for ${providerName} (${check.used}/${check.limit}), skipping`
        );
        return true;
      }
    }

    return false;
  }

  private requestForProvider(
    request: LLMRequest,
    providerName: string,
    providerModels: Map<string, string>
  ): LLMRequest {
    const model = providerModels.get(providerName);
    if (!model) {
      return request;
    }

    return { ...request, model };
  }

  private applyFallbackDecision(
    decision: FallbackDecision,
    failedProvider: string,
    providerChain: string[],
    currentIndex: number,
    providerModels: Map<string, string>
  ): void {
    const targetProvider = decision.fallbackProvider;
    if (!targetProvider || targetProvider === failedProvider || !this.providers.has(targetProvider)) {
      return;
    }

    if (decision.fallbackModel) {
      providerModels.set(targetProvider, decision.fallbackModel);
    }

    const nextIndex = currentIndex + 1;
    const firstIndex = providerChain.indexOf(targetProvider);
    if (firstIndex >= 0 && firstIndex <= currentIndex) {
      return;
    }

    const existingIndex = providerChain.indexOf(targetProvider, nextIndex);
    if (existingIndex >= 0) {
      providerChain.splice(existingIndex, 1);
    }

    providerChain.splice(nextIndex, 0, targetProvider);
  }
}

/**
 * Create a provider factory with common configurations
 */
export function createLLMProviderFactory(config: ProviderFactoryConfig): LLMProviderFactory {
  return new LLMProviderFactory(config);
}

/**
 * Create a cost-optimized provider factory
 */
export function createCostOptimizedFactory(config: ProviderFactoryConfig): LLMProviderFactory {
  return new LLMProviderFactory({
    ...config,
    defaultProvider: 'auto',
    costOptimization: true,
    enableCircuitBreaker: true,
    enableRetries: true,
    fallbackRules: [
      { condition: 'cost', threshold: 5, fallbackProvider: 'cloudflare' },
      { condition: 'rate_limit', fallbackProvider: 'cloudflare' },
      { condition: 'error', fallbackProvider: 'anthropic' }
    ]
  });
}
