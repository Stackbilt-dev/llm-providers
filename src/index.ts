/**
 * LLM Providers Package
 * Unified interface for OpenAI, Anthropic, and Cloudflare AI providers
 * with intelligent fallback logic and cost optimization
 */

// Core types
export type {
  LLMProvider,
  LLMImageInput,
  GatewayMetadata,
  CacheHints,
  CacheObservability,
  ResponseCacheAdapter,
  LLMRequest,
  LLMResponse,
  LLMMessage,
  LLMConfig,
  TokenUsage,
  Tool,
  BuiltInTool,
  BuiltInToolType,
  BuiltInToolResult,
  ToolCall,
  ToolResult,
  ProviderConfig,
  OpenAIConfig,
  AnthropicConfig,
  CloudflareAIGatewayOptions,
  CloudflareConfig,
  CerebrasConfig,
  GroqConfig,
  NvidiaConfig,
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
  QuotaHook,
  QuotaCheckInput,
  QuotaCheckResult,
  QuotaRecordInput,
  ToolExecutor,
  ToolLoopAbortSignal,
  ToolLoopOptions,
  ToolLoopState,
  ClassifyOptions,
  ClassifyResult,
  AnalyzeImageInput,
  ProviderBalance,
  RateLimitBalance,
  BatchRequest,
  BatchResponse,
  BatchJob
} from './types.js';

// Canonical provider contract
export {
  canonicalToLLMRequest,
  normalizeLLMRequest,
  normalizeLLMResponse,
} from './canonical.js';
export type {
  CanonicalDegradation,
  CanonicalFallbackHop,
  CanonicalLLMRequest,
  CanonicalLLMResponse,
  CanonicalMediaPart,
  CanonicalMessage,
  CanonicalOutputFormat,
  CanonicalProviderOptions,
  CanonicalRequestMetadata,
  CanonicalRequirements,
  CanonicalResponseError,
  CanonicalRoutingMetadata,
  CanonicalSamplingOptions,
  CanonicalTool,
  CanonicalToolMode,
} from './canonical.js';

// Gateway / Worker route planning
export { getGatewayRoutePlan } from './gateway-routing.js';
export type {
  GatewayRouteCachePlan,
  GatewayRouteCapabilityReport,
  GatewayRoutePlan,
  GatewayRouteRequirements,
} from './gateway-routing.js';

// Provider implementations
export { BaseProvider } from './providers/base.js';
export { OpenAIProvider } from './providers/openai.js';
export { AnthropicProvider } from './providers/anthropic.js';
export { CloudflareProvider } from './providers/cloudflare.js';
export { CerebrasProvider } from './providers/cerebras.js';
export { GroqProvider } from './providers/groq.js';
export { NvidiaProvider } from './providers/nvidia.js';

// Factory pattern
export {
  LLMProviderFactory,
  createLLMProviderFactory,
  createCostOptimizedFactory
} from './factory.js';
export type { ProviderFactoryConfig, CostAnalytics, ProviderHealthEntry } from './factory.js';
export {
  MODEL_CATALOG,
  MODEL_RECOMMENDATIONS,
  PROVIDER_FALLBACK_ORDER,
  getCatalogEntry,
  getProviderDefaultModel,
  getProviderDefaultModelForWorkload,
  getProviderForCatalogModel,
  getProvidersForCatalogModel,
  modelSupportsBuiltInTools,
  getRecommendedModel,
  getRecommendedModelForWorkload,
  getRoutingInfo,
  inferUseCaseFromRequest,
  normalizeModelWorkload,
  rankModels,
} from './model-catalog.js';
export type {
  ModelCatalogEntry,
  ModelLifecycle,
  ModelPreferenceMap,
  ModelRecommendationUseCase,
  ModelSelectionContext,
  ModelWorkloadClass,
  ProviderName,
  RoutingInfo,
} from './model-catalog.js';

// Local imports for use within this file
import { LLMProviderFactory } from './factory.js';
import type { ProviderFactoryConfig, CostAnalytics, ProviderHealthEntry } from './factory.js';
import type {
  AnalyzeImageInput,
  ClassifyOptions,
  ClassifyResult,
  LLMProvider,
  LLMRequest,
  LLMResponse,
  ProviderBalance,
  ToolExecutor,
  ToolLoopOptions
} from './types.js';
import type { ModelRecommendationUseCase } from './model-catalog.js';
import { ConfigurationError } from './errors.js';

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
  SchemaDriftError,
  ToolLoopLimitError,
  ToolLoopAbortedError,
  LLMErrorFactory
} from './errors.js';

// Schema validator (for custom provider authors)
export { validateSchema } from './utils/schema-validator.js';
export type { SchemaField, SchemaFieldType } from './utils/schema-validator.js';

// Schema drift canary (for integration test suites and cron canaries)
export { extractShape, compareShapes, runCanaryCheck } from './utils/schema-canary.js';
export type { ShapeMap, CanaryDiff, CanaryReport } from './utils/schema-canary.js';

// Image generation
export { ImageProvider, normalizeAiResponse } from './image/index.js';
export type { ImageProviderConfig, ImageRequest, ImageResponse, ImageModelConfig, ImageModelInputFormat } from './image/index.js';
export { IMAGE_MODELS, getImageModel } from './image/index.js';

// Logger
export { noopLogger, consoleLogger } from './utils/logger.js';
export type { Logger } from './utils/logger.js';

// Observability hooks
export { noopHooks, composeHooks } from './utils/hooks.js';
export type {
  ObservabilityHooks,
  RequestStartEvent,
  RequestEndEvent,
  CacheEvent,
  RequestErrorEvent,
  RetryEvent,
  FallbackEvent,
  CircuitStateChangeEvent,
  QuotaExhaustedEvent,
  BudgetThresholdEvent,
  QuotaCheckEvent,
  QuotaDeniedEvent,
  ProviderBalanceEvent,
  SchemaDriftEvent,
} from './utils/hooks.js';

// Exhaustion registry
export { ExhaustionRegistry, defaultExhaustionRegistry } from './utils/exhaustion.js';
export type { ExhaustionEntry, ExhaustionRegistrySnapshot } from './utils/exhaustion.js';

// Latency histogram
export { LatencyHistogram, defaultLatencyHistogram } from './utils/latency-histogram.js';
export type { LatencySummary } from './utils/latency-histogram.js';

// Utility classes
export { RetryManager, defaultRetryManager, withRetry, retry } from './utils/retry.js';
export {
  CircuitBreaker,
  CircuitBreakerManager,
  defaultCircuitBreakerManager
} from './utils/circuit-breaker.js';
export type { CircuitBreakerManagerSnapshot, CircuitBreakerSnapshot } from './utils/circuit-breaker.js';
export {
  CostTracker,
  CostOptimizer,
  defaultCostTracker
} from './utils/cost-tracker.js';
export type {
  ProviderCostEntry,
  ProviderCostBreakdownEntry
} from './utils/cost-tracker.js';
export { CreditLedger } from './utils/credit-ledger.js';
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
} from './utils/credit-ledger.js';

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
  quotaHook?: ProviderFactoryConfig['quotaHook'];
  quotaFailPolicy?: ProviderFactoryConfig['quotaFailPolicy'];
  hooks?: ProviderFactoryConfig['hooks'];
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
    if (typeof env.NVIDIA_API_KEY === 'string' && env.NVIDIA_API_KEY) {
      config.nvidia = { apiKey: env.NVIDIA_API_KEY };
    }
    if (env.AI != null && typeof env.AI === 'object') {
      config.cloudflare = { ai: env.AI as Ai };
    }

    const detected = [
      config.anthropic && 'anthropic',
      config.openai && 'openai',
      config.groq && 'groq',
      config.cerebras && 'cerebras',
      config.nvidia && 'nvidia',
      config.cloudflare && 'cloudflare',
    ].filter(Boolean) as string[];

    if (detected.length === 0) {
      throw new ConfigurationError(
        'fromEnv',
        'No LLM providers detected in env. Expected at least one of: ' +
          'ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, CEREBRAS_API_KEY, NVIDIA_API_KEY, or AI binding.'
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
    if (overrides.quotaHook !== undefined) {
      config.quotaHook = overrides.quotaHook;
    }
    if (overrides.quotaFailPolicy !== undefined) {
      config.quotaFailPolicy = overrides.quotaFailPolicy;
    }
    if (overrides.hooks !== undefined) {
      config.hooks = overrides.hooks;
    }

    return new LLMProviders(config);
  }

  /**
   * Generate response with automatic provider selection and fallback
   */
  async generateResponse(request: LLMRequest): Promise<LLMResponse> {
    return this.factory.generateResponse(request);
  }

  async generateResponseStream(request: LLMRequest): Promise<ReadableStream<string>> {
    return this.factory.generateResponseStream(request);
  }

  async generateResponseWithTools(
    request: LLMRequest,
    toolExecutor: ToolExecutor,
    opts?: ToolLoopOptions
  ): Promise<LLMResponse> {
    return this.factory.generateResponseWithTools(request, toolExecutor, opts);
  }

  async classify<T = unknown>(
    input: string | LLMRequest,
    options?: ClassifyOptions<T>
  ): Promise<ClassifyResult<T>> {
    return this.factory.classify(input, options);
  }

  async analyzeImage(input: AnalyzeImageInput): Promise<LLMResponse> {
    return this.factory.analyzeImage(input);
  }

  async getProviderBalance(provider?: string): Promise<ProviderBalance | Record<string, ProviderBalance>> {
    return this.factory.getProviderBalance(provider);
  }

  getRecommendedModel(
    request: LLMRequest,
    useCase?: ModelRecommendationUseCase
  ): string {
    return this.factory.getRecommendedModel(request, useCase);
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
  async getHealth(): Promise<Record<string, ProviderHealthEntry>> {
    return this.factory.getProviderHealth();
  }

  /**
   * Get cost analytics
   */
  getCostAnalytics(): CostAnalytics {
    return this.factory.getCostAnalytics();
  }

  /**
   * Get latency histogram summaries for all providers
   */
  getLatencyHistogram(): Record<string, import('./utils/latency-histogram').LatencySummary> {
    return this.factory.getLatencyHistogram();
  }

  /**
   * Get currently quota-exhausted providers
   */
  getExhaustedProviders(): string[] {
    return this.factory.getExhaustedProviders();
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
  return new LLMProviders({
    ...config,
    defaultProvider: 'auto',
    costOptimization: true,
    enableCircuitBreaker: true,
    enableRetries: true,
    fallbackRules: config.fallbackRules ?? [
      { condition: 'cost', threshold: 5, fallbackProvider: 'cloudflare' },
      { condition: 'rate_limit', fallbackProvider: 'cloudflare' },
      { condition: 'error', fallbackProvider: 'anthropic' }
    ]
  });
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
export { VERSION } from './version.js';
export const SUPPORTED_PROVIDERS = ['openai', 'anthropic', 'cloudflare', 'cerebras', 'groq', 'nvidia'] as const;

/**
 * Common model mappings for easy reference
 */
export const MODELS = {
  // OpenAI models
  /** @deprecated Retired by OpenAI on 2026-04-03. Use GPT_4O_MINI or a current GPT-4 successor. */
  GPT_4O: 'gpt-4o',
  GPT_4O_MINI: 'gpt-4o-mini',
  GPT_4_TURBO: 'gpt-4-turbo',
  GPT_4: 'gpt-4',
  GPT_3_5_TURBO: 'gpt-3.5-turbo',

  // Anthropic models
  CLAUDE_OPUS_4_6: 'claude-opus-4-6-20250618',
  CLAUDE_SONNET_4_6: 'claude-sonnet-4-6-20250618',
  CLAUDE_OPUS_4: 'claude-opus-4-20250618',
  CLAUDE_SONNET_4: 'claude-sonnet-4-20250514',
  CLAUDE_HAIKU_4_5: 'claude-haiku-4-5-20251001',
  CLAUDE_3_7_SONNET: 'claude-3-7-sonnet-20250219',
  CLAUDE_3_5_SONNET: 'claude-3-5-sonnet-20241022',
  CLAUDE_3_5_HAIKU: 'claude-3-5-haiku-20241022',
  CLAUDE_3_OPUS: 'claude-3-opus-20240229',
  CLAUDE_3_SONNET: 'claude-3-sonnet-20240229',
  /** @deprecated Retires 2026-04-19. Use CLAUDE_HAIKU_4_5 or CLAUDE_3_5_HAIKU. */
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
  GROQ_LLAMA_3_1_8B_INSTANT: 'llama-3.1-8b-instant',
  GROQ_GPT_OSS_120B: 'openai/gpt-oss-120b',
  GROQ_COMPOUND: 'groq/compound',
  GROQ_COMPOUND_MINI: 'groq/compound-mini',

  // NVIDIA NIM models
  NVIDIA_LLAMA_3_3_70B: 'meta/llama-3.3-70b-instruct',
  NVIDIA_LLAMA_4_MAVERICK: 'meta/llama-4-maverick-17b-128e-instruct',
  NVIDIA_NEMOTRON_70B: 'nvidia/llama-3.1-nemotron-70b-instruct',
  NVIDIA_NEMOTRON_SUPER_49B: 'nvidia/llama-3.3-nemotron-super-49b-v1',
  NVIDIA_NEMOTRON_ULTRA_253B: 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  NVIDIA_LLAMA_3_1_70B: 'meta/llama-3.1-70b-instruct',
  NVIDIA_MISTRAL_LARGE_2: 'mistralai/mistral-large-2-instruct',
  NVIDIA_DEEPSEEK_V4_FLASH: 'deepseek-ai/deepseek-v4-flash',
  NVIDIA_DEEPSEEK_V4_PRO: 'deepseek-ai/deepseek-v4-pro',
} as const;

