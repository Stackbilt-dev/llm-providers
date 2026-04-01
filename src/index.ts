/**
 * LLM Providers Package
 * Unified interface for OpenAI, Anthropic, and Cloudflare AI providers
 * with intelligent fallback logic and cost optimization
 */

// Core types
export type {
  LLMProvider,
  LLMRequest,
  LLMResponse,
  LLMMessage,
  LLMConfig,
  TokenUsage,
  Tool,
  ToolCall,
  ToolResult,
  ProviderConfig,
  OpenAIConfig,
  AnthropicConfig,
  CloudflareConfig,
  CerebrasConfig,
  GroqConfig,
  ModelCapabilities,
  ProviderCapabilities,
  ProviderMetrics,
  FallbackRule,
  CircuitBreakerConfig,
  CircuitBreakerState,
  RetryConfig,
  CostConfig,
  LLMError,
  StreamChunk,
  StreamResponse,
  BatchRequest,
  BatchResponse,
  BatchJob
} from './types';

// Provider implementations
export { BaseProvider } from './providers/base';
export { OpenAIProvider } from './providers/openai';
export { AnthropicProvider } from './providers/anthropic';
export { CloudflareProvider } from './providers/cloudflare';
export { CerebrasProvider } from './providers/cerebras';
export { GroqProvider } from './providers/groq';

// Factory pattern
export {
  LLMProviderFactory,
  createLLMProviderFactory,
  createCostOptimizedFactory
} from './factory';
export type { ProviderFactoryConfig } from './factory';

// Local imports for use within this file
import { LLMProviderFactory } from './factory';
import type { ProviderFactoryConfig } from './factory';
import type { LLMProvider, LLMRequest, LLMResponse } from './types';
import { createCostOptimizedFactory } from './factory';
import { ConfigurationError } from './errors';

// Error classes
export {
  LLMProviderError,
  RateLimitError,
  QuotaExceededError,
  AuthenticationError,
  InvalidRequestError,
  ModelNotFoundError,
  TimeoutError,
  NetworkError,
  ServerError,
  ContentFilterError,
  TokenLimitError,
  ConfigurationError,
  CircuitBreakerOpenError,
  LLMErrorFactory
} from './errors';

// Utility classes
export { RetryManager, defaultRetryManager, withRetry, retry } from './utils/retry';
export { 
  CircuitBreaker, 
  CircuitBreakerManager, 
  defaultCircuitBreakerManager 
} from './utils/circuit-breaker';
export {
  CostTracker,
  CostOptimizer,
  defaultCostTracker
} from './utils/cost-tracker';
export type {
  ProviderCostEntry,
  ProviderCostBreakdownEntry
} from './utils/cost-tracker';
export { CreditLedger } from './utils/credit-ledger';
export type {
  CreditLedgerSnapshot,
  LedgerEvent,
  ThresholdEvent,
  DepletionEvent,
  LedgerListener,
  BudgetConfig,
  ThresholdConfig,
  ThresholdTier,
  DepletionTier,
  ProviderAccumulator,
  ModelAccumulator,
  RateLimitDimension,
  RateLimitCheck,
  SpendEntry,
  BurnRate,
  DepletionEstimate,
  SpendSummary,
} from './utils/credit-ledger';

/**
 * Overrides for `LLMProviders.fromEnv()` auto-discovery.
 */
export interface FromEnvOverrides {
  defaultProvider?: ProviderFactoryConfig['defaultProvider'];
  costOptimization?: boolean;
  enableCircuitBreaker?: boolean;
  enableRetries?: boolean;
  fallbackRules?: ProviderFactoryConfig['fallbackRules'];
  ledger?: ProviderFactoryConfig['ledger'];
}

/**
 * Main LLMProviders class for easy usage
 */
export class LLMProviders {
  private factory: LLMProviderFactory;

  constructor(config: ProviderFactoryConfig) {
    this.factory = new LLMProviderFactory(config);
  }

  /**
   * Auto-discover providers from a Cloudflare Worker `env` object.
   *
   * Scans for known API-key environment variables and bindings, configures
   * only the providers whose keys are present, and returns a ready-to-use
   * `LLMProviders` instance.
   *
   * @throws ConfigurationError if no providers are detected
   */
  static fromEnv(
    env: Record<string, unknown>,
    overrides: FromEnvOverrides = {}
  ): LLMProviders {
    const config: ProviderFactoryConfig = {};

    if (typeof env.ANTHROPIC_API_KEY === 'string' && env.ANTHROPIC_API_KEY) {
      config.anthropic = { apiKey: env.ANTHROPIC_API_KEY };
    }
    if (typeof env.OPENAI_API_KEY === 'string' && env.OPENAI_API_KEY) {
      config.openai = { apiKey: env.OPENAI_API_KEY };
    }
    if (typeof env.GROQ_API_KEY === 'string' && env.GROQ_API_KEY) {
      config.groq = { apiKey: env.GROQ_API_KEY };
    }
    if (typeof env.CEREBRAS_API_KEY === 'string' && env.CEREBRAS_API_KEY) {
      config.cerebras = { apiKey: env.CEREBRAS_API_KEY };
    }
    if (env.AI != null && typeof env.AI === 'object') {
      config.cloudflare = { ai: env.AI as Ai };
    }

    const detected = [
      config.anthropic && 'anthropic',
      config.openai && 'openai',
      config.groq && 'groq',
      config.cerebras && 'cerebras',
      config.cloudflare && 'cloudflare',
    ].filter(Boolean) as string[];

    if (detected.length === 0) {
      throw new ConfigurationError(
        'fromEnv',
        'No LLM providers detected in env. Expected at least one of: ' +
          'ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, CEREBRAS_API_KEY, or AI binding.'
      );
    }

    if (overrides.defaultProvider !== undefined) {
      config.defaultProvider = overrides.defaultProvider;
    }
    if (overrides.costOptimization !== undefined) {
      config.costOptimization = overrides.costOptimization;
    }
    if (overrides.enableCircuitBreaker !== undefined) {
      config.enableCircuitBreaker = overrides.enableCircuitBreaker;
    }
    if (overrides.enableRetries !== undefined) {
      config.enableRetries = overrides.enableRetries;
    }
    if (overrides.fallbackRules !== undefined) {
      config.fallbackRules = overrides.fallbackRules;
    }
    if (overrides.ledger !== undefined) {
      config.ledger = overrides.ledger;
    }

    return new LLMProviders(config);
  }

  /**
   * Generate response with automatic provider selection and fallback
   */
  async generateResponse(request: LLMRequest): Promise<LLMResponse> {
    return this.factory.generateResponse(request);
  }

  /**
   * Get specific provider instance
   */
  getProvider(name: string): LLMProvider | undefined {
    return this.factory.getProvider(name);
  }

  /**
   * Get all available providers
   */
  getAvailableProviders(): string[] {
    return this.factory.getAvailableProviders();
  }

  /**
   * Get provider health status
   */
  async getHealth(): Promise<Record<string, any>> {
    return this.factory.getProviderHealth();
  }

  /**
   * Get cost analytics
   */
  getCostAnalytics(): any {
    return this.factory.getCostAnalytics();
  }

  /**
   * Reset all metrics and circuit breakers
   */
  reset(): void {
    this.factory.reset();
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<ProviderFactoryConfig>): void {
    this.factory.updateConfig(config);
  }
}

/**
 * Create LLMProviders instance with cost optimization
 */
export function createCostOptimizedLLMProviders(
  config: ProviderFactoryConfig
): LLMProviders {
  const factory = createCostOptimizedFactory(config);
  return new LLMProviders(config);
}

/**
 * Create LLMProviders instance with basic configuration
 */
export function createLLMProviders(config: ProviderFactoryConfig): LLMProviders {
  return new LLMProviders(config);
}

/**
 * Default export for convenience
 */
export default LLMProviders;

/**
 * Version and metadata
 */
export const VERSION = '0.1.0';
export const SUPPORTED_PROVIDERS = ['openai', 'anthropic', 'cloudflare', 'cerebras', 'groq'] as const;

/**
 * Common model mappings for easy reference
 */
export const MODELS = {
  // OpenAI models
  GPT_4O: 'gpt-4o',
  GPT_4O_MINI: 'gpt-4o-mini',
  GPT_4_TURBO: 'gpt-4-turbo',
  GPT_4: 'gpt-4',
  GPT_3_5_TURBO: 'gpt-3.5-turbo',

  // Anthropic models
  CLAUDE_3_5_SONNET: 'claude-3-5-sonnet-20241022',
  CLAUDE_3_5_HAIKU: 'claude-3-5-haiku-20241022',
  CLAUDE_3_OPUS: 'claude-3-opus-20240229',
  CLAUDE_3_SONNET: 'claude-3-sonnet-20240229',
  CLAUDE_3_HAIKU: 'claude-3-haiku-20240307',

  // Cloudflare models
  LLAMA_3_1_8B: '@cf/meta/llama-3.1-8b-instruct',
  LLAMA_3_1_70B: '@cf/meta/llama-3.1-70b-instruct',
  LLAMA_3_8B: '@cf/meta/llama-3-8b-instruct',
  MISTRAL_7B: '@cf/mistral/mistral-7b-instruct-v0.1',
  TINY_LLAMA: '@cf/tinyllama/tinyllama-1.1b-chat-v1.0',

  // Cerebras models
  CEREBRAS_LLAMA_3_1_8B: 'llama-3.1-8b',
  CEREBRAS_LLAMA_3_3_70B: 'llama-3.3-70b',
  CEREBRAS_ZAI_GLM_4_7: 'zai-glm-4.7',
  CEREBRAS_QWEN_3_235B: 'qwen-3-235b-a22b-instruct-2507',

  // Groq models
  GROQ_LLAMA_3_3_70B_VERSATILE: 'llama-3.3-70b-versatile',
  GROQ_LLAMA_3_1_8B_INSTANT: 'llama-3.1-8b-instant'
} as const;

/**
 * Model recommendations by use case
 */
export const MODEL_RECOMMENDATIONS = {
  // Cost-optimized models
  COST_EFFECTIVE: [
    MODELS.CEREBRAS_LLAMA_3_1_8B,
    MODELS.TINY_LLAMA,
    MODELS.CLAUDE_3_5_HAIKU,
    MODELS.GPT_4O_MINI
  ],

  // High-performance models
  HIGH_PERFORMANCE: [
    MODELS.CEREBRAS_ZAI_GLM_4_7,
    MODELS.GPT_4O,
    MODELS.CLAUDE_3_5_SONNET,
    MODELS.CEREBRAS_QWEN_3_235B,
    MODELS.LLAMA_3_1_70B
  ],

  // Balanced models
  BALANCED: [
    MODELS.GPT_3_5_TURBO,
    MODELS.CLAUDE_3_HAIKU,
    MODELS.LLAMA_3_1_8B
  ],

  // Tool/function calling
  TOOL_CALLING: [
    MODELS.GPT_4O,
    MODELS.CLAUDE_3_5_SONNET,
    MODELS.CEREBRAS_ZAI_GLM_4_7,
    MODELS.CEREBRAS_QWEN_3_235B,
    MODELS.GPT_4_TURBO
  ],

  // Long context tasks
  LONG_CONTEXT: [
    MODELS.CLAUDE_3_5_SONNET,
    MODELS.GPT_4_TURBO,
    MODELS.CLAUDE_3_OPUS
  ]
} as const;

/**
 * Utility function to get recommended model for a use case
 */
export function getRecommendedModel(
  useCase: keyof typeof MODEL_RECOMMENDATIONS,
  availableProviders: string[]
): string {
  const recommendations = MODEL_RECOMMENDATIONS[useCase];
  
  for (const model of recommendations) {
    // Check if we have the provider for this model
    if (model.startsWith('gpt-') && availableProviders.includes('openai')) {
      return model;
    }
    if (model.includes('claude') && availableProviders.includes('anthropic')) {
      return model;
    }
    if (model.startsWith('@cf/') && availableProviders.includes('cloudflare')) {
      return model;
    }
    if ((model.startsWith('llama-3.1-8b') || model.startsWith('llama-3.3-70b')
        || model.startsWith('zai-glm') || model.startsWith('qwen-3-235b')) && availableProviders.includes('cerebras')) {
      return model;
    }
    if ((model.includes('-versatile') || model.includes('-instant')) && availableProviders.includes('groq')) {
      return model;
    }
  }

  // Fallback to first available provider's default model
  if (availableProviders.includes('cloudflare')) {
    return MODELS.LLAMA_3_1_8B;
  }
  if (availableProviders.includes('anthropic')) {
    return MODELS.CLAUDE_3_HAIKU;
  }
  if (availableProviders.includes('openai')) {
    return MODELS.GPT_3_5_TURBO;
  }

  throw new Error('No available providers configured');
}
