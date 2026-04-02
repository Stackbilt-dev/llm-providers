/**
 * LLM Provider Factory
 * Creates and manages LLM provider instances with intelligent fallback logic
 */

import type {
  LLMProvider,
  LLMConfig,
  LLMRequest,
  LLMResponse,
  OpenAIConfig,
  AnthropicConfig,
  CloudflareConfig,
  CerebrasConfig,
  GroqConfig,
  FallbackRule,
  ProviderMetrics,
  CircuitBreakerState
} from './types';

import type { Logger } from './utils/logger';
import { noopLogger } from './utils/logger';
import { OpenAIProvider } from './providers/openai';
import { AnthropicProvider } from './providers/anthropic';
import { CloudflareProvider } from './providers/cloudflare';
import { CerebrasProvider } from './providers/cerebras';
import { GroqProvider } from './providers/groq';
import { CostTracker, defaultCostTracker } from './utils/cost-tracker';
import type { ProviderCostBreakdownEntry } from './utils/cost-tracker';
import type { CreditLedger } from './utils/credit-ledger';
import { defaultCircuitBreakerManager } from './utils/circuit-breaker';
import {
  LLMProviderError,
  ConfigurationError,
  CircuitBreakerOpenError,
  AuthenticationError,
  RateLimitError
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
  logger?: Logger;
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

export class LLMProviderFactory {
  private providers: Map<string, LLMProvider> = new Map();
  private config: ProviderFactoryConfig;
  private costTracker: CostTracker;
  private fallbackRules: FallbackRule[];
  private logger: Logger;

  constructor(config: ProviderFactoryConfig) {
    this.config = config;
    this.logger = config.logger ?? noopLogger;
    this.costTracker = defaultCostTracker;
    this.fallbackRules = config.fallbackRules || this.getDefaultFallbackRules();

    this.initializeProviders();
  }

  /**
   * Initialize all configured providers
   */
  private initializeProviders(): void {
    // Initialize OpenAI provider
    if (this.config.openai) {
      try {
        const provider = new OpenAIProvider({ ...this.config.openai, logger: this.logger });
        if (provider.validateConfig()) {
          this.providers.set('openai', provider);
          this.logger.info('[LLMProviderFactory] OpenAI provider initialized');
        }
      } catch (error) {
        this.logger.warn('[LLMProviderFactory] Failed to initialize OpenAI provider:', (error as Error).message);
      }
    }

    // Initialize Anthropic provider
    if (this.config.anthropic) {
      try {
        const provider = new AnthropicProvider({ ...this.config.anthropic, logger: this.logger });
        if (provider.validateConfig()) {
          this.providers.set('anthropic', provider);
          this.logger.info('[LLMProviderFactory] Anthropic provider initialized');
        }
      } catch (error) {
        this.logger.warn('[LLMProviderFactory] Failed to initialize Anthropic provider:', (error as Error).message);
      }
    }

    // Initialize Cloudflare provider
    if (this.config.cloudflare) {
      try {
        const provider = new CloudflareProvider({ ...this.config.cloudflare, logger: this.logger });
        if (provider.validateConfig()) {
          this.providers.set('cloudflare', provider);
          this.logger.info('[LLMProviderFactory] Cloudflare provider initialized');
        }
      } catch (error) {
        this.logger.warn('[LLMProviderFactory] Failed to initialize Cloudflare provider:', (error as Error).message);
      }
    }

    // Initialize Cerebras provider
    if (this.config.cerebras) {
      try {
        const provider = new CerebrasProvider({ ...this.config.cerebras, logger: this.logger });
        if (provider.validateConfig()) {
          this.providers.set('cerebras', provider);
          this.logger.info('[LLMProviderFactory] Cerebras provider initialized');
        }
      } catch (error) {
        this.logger.warn('[LLMProviderFactory] Failed to initialize Cerebras provider:', (error as Error).message);
      }
    }

    // Initialize Groq provider
    if (this.config.groq) {
      try {
        const provider = new GroqProvider({ ...this.config.groq, logger: this.logger });
        if (provider.validateConfig()) {
          this.providers.set('groq', provider);
          this.logger.info('[LLMProviderFactory] Groq provider initialized');
        }
      } catch (error) {
        this.logger.warn('[LLMProviderFactory] Failed to initialize Groq provider:', (error as Error).message);
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
    let lastError: Error | null = null;

    for (const providerName of providerChain) {
      try {
        const provider = this.providers.get(providerName);
        if (!provider) continue;

        // Check circuit breaker
        if (this.config.enableCircuitBreaker) {
          const breaker = defaultCircuitBreakerManager.getBreaker(providerName);
          if (breaker.isOpen()) {
            this.logger.warn(`[LLMProviderFactory] Circuit breaker open for ${providerName}, skipping`);
            continue;
          }
        }

        // Check rate limits if ledger is configured
        if (this.config.ledger) {
          const rpmCheck = this.config.ledger.checkRateLimit(providerName, 'rpm');
          if (!rpmCheck.allowed) {
            this.logger.warn(`[LLMProviderFactory] Rate limit (rpm) exceeded for ${providerName} (${rpmCheck.used}/${rpmCheck.limit}), skipping`);
            continue;
          }
          const rpdCheck = this.config.ledger.checkRateLimit(providerName, 'rpd');
          if (!rpdCheck.allowed) {
            this.logger.warn(`[LLMProviderFactory] Rate limit (rpd) exceeded for ${providerName} (${rpdCheck.used}/${rpdCheck.limit}), skipping`);
            continue;
          }
        }

        this.logger.debug(`[LLMProviderFactory] Trying provider: ${providerName}`);

        const response = await provider.generateResponse(request);

        // Track cost if enabled
        if (this.config.costOptimization) {
          this.costTracker.trackCost(providerName, response);
        }

        this.logger.debug(`[LLMProviderFactory] Successfully used provider: ${providerName}`);
        return response;

      } catch (error) {
        lastError = error as Error;
        this.logger.warn(`[LLMProviderFactory] Provider ${providerName} failed:`, (error as Error).message);

        // Check if we should continue trying other providers
        if (!this.shouldFallback(error as Error)) {
          throw error;
        }
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
    if (!this.config.costOptimization) {
      // Default priority: all configured providers, cheapest first
      return ['cloudflare', 'cerebras', 'groq', 'anthropic', 'openai']
        .filter(p => this.providers.has(p));
    }

    // Cost-optimized routing
    const providers = Array.from(this.providers.keys());
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
    // Don't fallback for authentication errors
    if (error instanceof AuthenticationError) {
      return false;
    }

    // Don't fallback for configuration errors
    if (error instanceof ConfigurationError) {
      return false;
    }

    // Fallback for circuit breaker, rate limits, and server errors
    if (error instanceof CircuitBreakerOpenError ||
        error instanceof RateLimitError) {
      return true;
    }

    if (error instanceof LLMProviderError) {
      if (error.code === 'SERVER_ERROR' ||
          error.code === 'NETWORK_ERROR' ||
          error.code === 'TIMEOUT') {
        return true;
      }
    }

    // Check custom fallback rules
    for (const rule of this.fallbackRules) {
      if (this.evaluateFallbackRule(rule, error)) {
        return true;
      }
    }

    return false;
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
   * Reset all provider metrics and circuit breakers
   */
  reset(): void {
    for (const [name, provider] of this.providers) {
      provider.resetMetrics();
      if (this.config.enableCircuitBreaker) {
        defaultCircuitBreakerManager.reset(name);
      }
    }

    if (this.config.costOptimization) {
      this.costTracker.reset();
    }
  }

  /**
   * Update factory configuration
   */
  updateConfig(config: Partial<ProviderFactoryConfig>): void {
    this.config = { ...this.config, ...config };

    if (config.fallbackRules) {
      this.fallbackRules = config.fallbackRules;
    }

    // Re-initialize providers if configs changed
    if (config.openai || config.anthropic || config.cloudflare || config.cerebras || config.groq) {
      this.providers.clear();
      this.initializeProviders();
    }
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
