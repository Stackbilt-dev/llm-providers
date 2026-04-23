import type { CreditLedger } from './utils/credit-ledger';
import type { CircuitBreakerState, LLMRequest, ModelCapabilities } from './types';

export type ProviderName = 'openai' | 'anthropic' | 'cloudflare' | 'cerebras' | 'groq';
export type ModelLifecycle = 'active' | 'compatibility' | 'retired';
export type ModelRecommendationUseCase =
  | 'COST_EFFECTIVE'
  | 'HIGH_PERFORMANCE'
  | 'BALANCED'
  | 'TOOL_CALLING'
  | 'LONG_CONTEXT'
  | 'VISION';

export interface ModelCatalogEntry {
  provider: ProviderName;
  model: string;
  lifecycle: ModelLifecycle;
  useCases: ModelRecommendationUseCase[];
  capabilities: ModelCapabilities;
  speedScore: number;
  qualityScore: number;
  costScore: number;
}

type SelectionHealthEntry = {
  healthy?: boolean;
  circuitBreaker?: CircuitBreakerState | null;
};

export interface ModelSelectionContext {
  request?: Partial<LLMRequest>;
  providerHealth?: Partial<Record<ProviderName, SelectionHealthEntry>>;
  ledger?: Pick<CreditLedger, 'getDepletionEstimate' | 'utilizationPct'>;
}

export const PROVIDER_FALLBACK_ORDER: ProviderName[] = [
  'cloudflare',
  'cerebras',
  'groq',
  'anthropic',
  'openai',
];

function entry(
  provider: ProviderName,
  model: string,
  lifecycle: ModelLifecycle,
  useCases: ModelRecommendationUseCase[],
  capabilities: ModelCapabilities,
  scores: { speed: number; quality: number; cost: number },
): ModelCatalogEntry {
  return {
    provider,
    model,
    lifecycle,
    useCases,
    capabilities,
    speedScore: scores.speed,
    qualityScore: scores.quality,
    costScore: scores.cost,
  };
}

export const MODEL_CATALOG: readonly ModelCatalogEntry[] = [
  entry('openai', 'gpt-4o', 'retired', ['HIGH_PERFORMANCE', 'TOOL_CALLING'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: true,
    supportsVision: true,
    supportsBatching: false,
    inputTokenCost: 0.005,
    outputTokenCost: 0.015,
    description: 'Retired GPT-4o compatibility entry'
  }, { speed: 4, quality: 5, cost: 2 }),
  entry('openai', 'gpt-4o-mini', 'active', ['COST_EFFECTIVE', 'BALANCED', 'TOOL_CALLING', 'VISION', 'LONG_CONTEXT', 'HIGH_PERFORMANCE'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: true,
    supportsVision: true,
    supportsBatching: false,
    inputTokenCost: 0.00015,
    outputTokenCost: 0.0006,
    description: 'Modern OpenAI baseline'
  }, { speed: 5, quality: 4, cost: 5 }),
  entry('openai', 'gpt-4-turbo', 'compatibility', ['HIGH_PERFORMANCE', 'LONG_CONTEXT'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0.01,
    outputTokenCost: 0.03,
    description: 'Compatibility OpenAI GPT-4 Turbo'
  }, { speed: 3, quality: 4, cost: 2 }),
  entry('openai', 'gpt-4', 'compatibility', ['HIGH_PERFORMANCE'], {
    maxContextLength: 8192,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0.03,
    outputTokenCost: 0.06,
    description: 'Compatibility GPT-4'
  }, { speed: 2, quality: 4, cost: 1 }),
  entry('openai', 'gpt-3.5-turbo', 'compatibility', ['COST_EFFECTIVE'], {
    maxContextLength: 16385,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0.0005,
    outputTokenCost: 0.0015,
    description: 'Compatibility GPT-3.5 Turbo'
  }, { speed: 4, quality: 2, cost: 3 }),
  entry('openai', 'gpt-3.5-turbo-16k', 'compatibility', ['LONG_CONTEXT'], {
    maxContextLength: 16385,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0.001,
    outputTokenCost: 0.002,
    description: 'Compatibility GPT-3.5 Turbo 16k'
  }, { speed: 3, quality: 2, cost: 3 }),

  entry('anthropic', 'claude-opus-4-6-20250618', 'active', ['HIGH_PERFORMANCE', 'LONG_CONTEXT', 'VISION', 'TOOL_CALLING'], {
    maxContextLength: 200000,
    supportsStreaming: true,
    supportsTools: true,
    supportsVision: true,
    supportsBatching: false,
    inputTokenCost: 0.015,
    outputTokenCost: 0.075,
    description: 'Claude Opus 4.6'
  }, { speed: 2, quality: 5, cost: 1 }),
  entry('anthropic', 'claude-sonnet-4-6-20250618', 'active', ['HIGH_PERFORMANCE', 'LONG_CONTEXT', 'VISION', 'TOOL_CALLING', 'BALANCED'], {
    maxContextLength: 200000,
    supportsStreaming: true,
    supportsTools: true,
    supportsVision: true,
    supportsBatching: false,
    inputTokenCost: 0.003,
    outputTokenCost: 0.015,
    description: 'Claude Sonnet 4.6'
  }, { speed: 3, quality: 5, cost: 3 }),
  entry('anthropic', 'claude-opus-4-20250618', 'compatibility', ['HIGH_PERFORMANCE', 'LONG_CONTEXT'], {
    maxContextLength: 200000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0.015,
    outputTokenCost: 0.075,
    description: 'Claude Opus 4'
  }, { speed: 2, quality: 5, cost: 1 }),
  entry('anthropic', 'claude-sonnet-4-20250514', 'active', ['HIGH_PERFORMANCE', 'BALANCED', 'TOOL_CALLING', 'LONG_CONTEXT'], {
    maxContextLength: 200000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0.003,
    outputTokenCost: 0.015,
    description: 'Claude Sonnet 4'
  }, { speed: 3, quality: 4, cost: 3 }),
  entry('anthropic', 'claude-haiku-4-5-20251001', 'active', ['COST_EFFECTIVE', 'BALANCED', 'TOOL_CALLING', 'VISION', 'LONG_CONTEXT'], {
    maxContextLength: 200000,
    supportsStreaming: true,
    supportsTools: true,
    supportsVision: true,
    supportsBatching: false,
    inputTokenCost: 0.001,
    outputTokenCost: 0.005,
    description: 'Claude Haiku 4.5'
  }, { speed: 4, quality: 4, cost: 5 }),
  entry('anthropic', 'claude-3-7-sonnet-20250219', 'compatibility', ['HIGH_PERFORMANCE', 'LONG_CONTEXT'], {
    maxContextLength: 200000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0.003,
    outputTokenCost: 0.015,
    description: 'Claude 3.7 Sonnet'
  }, { speed: 3, quality: 4, cost: 3 }),
  entry('anthropic', 'claude-3-5-sonnet-20241022', 'compatibility', ['BALANCED', 'LONG_CONTEXT'], {
    maxContextLength: 200000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0.003,
    outputTokenCost: 0.015,
    description: 'Claude 3.5 Sonnet'
  }, { speed: 3, quality: 4, cost: 3 }),
  entry('anthropic', 'claude-3-5-haiku-20241022', 'compatibility', ['COST_EFFECTIVE', 'BALANCED'], {
    maxContextLength: 200000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0.00025,
    outputTokenCost: 0.00125,
    description: 'Claude 3.5 Haiku'
  }, { speed: 4, quality: 3, cost: 4 }),
  entry('anthropic', 'claude-3-opus-20240229', 'compatibility', ['HIGH_PERFORMANCE'], {
    maxContextLength: 200000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0.015,
    outputTokenCost: 0.075,
    description: 'Claude 3 Opus'
  }, { speed: 2, quality: 4, cost: 1 }),
  entry('anthropic', 'claude-3-sonnet-20240229', 'compatibility', ['BALANCED'], {
    maxContextLength: 200000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0.003,
    outputTokenCost: 0.015,
    description: 'Claude 3 Sonnet'
  }, { speed: 3, quality: 3, cost: 2 }),

  entry('cloudflare', '@cf/meta/llama-3.1-8b-instruct', 'compatibility', ['COST_EFFECTIVE'], {
    maxContextLength: 8192,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: true,
    inputTokenCost: 0.0000001,
    outputTokenCost: 0.0000001,
    description: 'Workers AI Llama 3.1 8B'
  }, { speed: 5, quality: 2, cost: 5 }),
  entry('cloudflare', '@cf/meta/llama-3.1-70b-instruct', 'compatibility', ['HIGH_PERFORMANCE'], {
    maxContextLength: 8192,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: true,
    inputTokenCost: 0.0000005,
    outputTokenCost: 0.0000005,
    description: 'Workers AI Llama 3.1 70B'
  }, { speed: 3, quality: 4, cost: 4 }),
  entry('cloudflare', '@cf/meta/llama-3-8b-instruct', 'compatibility', ['COST_EFFECTIVE'], {
    maxContextLength: 8192,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: true,
    inputTokenCost: 0.0000001,
    outputTokenCost: 0.0000001,
    description: 'Workers AI Llama 3 8B'
  }, { speed: 4, quality: 2, cost: 5 }),
  entry('cloudflare', '@cf/meta/llama-2-7b-chat-int8', 'compatibility', ['COST_EFFECTIVE'], {
    maxContextLength: 4096,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: true,
    inputTokenCost: 0.00000005,
    outputTokenCost: 0.00000005,
    description: 'Workers AI Llama 2 7B'
  }, { speed: 4, quality: 1, cost: 5 }),
  entry('cloudflare', '@cf/microsoft/phi-2', 'compatibility', ['COST_EFFECTIVE'], {
    maxContextLength: 2048,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: true,
    inputTokenCost: 0.00000002,
    outputTokenCost: 0.00000002,
    description: 'Workers AI Phi-2'
  }, { speed: 4, quality: 1, cost: 5 }),
  entry('cloudflare', '@cf/mistral/mistral-7b-instruct-v0.1', 'compatibility', ['BALANCED'], {
    maxContextLength: 8192,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: true,
    inputTokenCost: 0.0000001,
    outputTokenCost: 0.0000001,
    description: 'Workers AI Mistral 7B'
  }, { speed: 4, quality: 2, cost: 5 }),
  entry('cloudflare', '@cf/openchat/openchat-3.5-0106', 'compatibility', ['BALANCED'], {
    maxContextLength: 8192,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: true,
    inputTokenCost: 0.0000001,
    outputTokenCost: 0.0000001,
    description: 'Workers AI OpenChat 3.5'
  }, { speed: 4, quality: 2, cost: 5 }),
  entry('cloudflare', '@cf/openai/gpt-oss-120b', 'active', ['TOOL_CALLING', 'HIGH_PERFORMANCE'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: true,
    toolCalling: true,
    supportsBatching: true,
    inputTokenCost: 0.0000008,
    outputTokenCost: 0.0000008,
    description: 'Workers AI GPT-OSS 120B'
  }, { speed: 4, quality: 5, cost: 4 }),
  entry('cloudflare', '@cf/tinyllama/tinyllama-1.1b-chat-v1.0', 'compatibility', ['COST_EFFECTIVE'], {
    maxContextLength: 2048,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: true,
    inputTokenCost: 0.00000001,
    outputTokenCost: 0.00000001,
    description: 'Workers AI TinyLlama'
  }, { speed: 5, quality: 1, cost: 5 }),
  entry('cloudflare', '@cf/qwen/qwen1.5-0.5b-chat', 'compatibility', ['COST_EFFECTIVE'], {
    maxContextLength: 2048,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: true,
    inputTokenCost: 0.00000001,
    outputTokenCost: 0.00000001,
    description: 'Workers AI Qwen 1.5 0.5B'
  }, { speed: 5, quality: 1, cost: 5 }),
  entry('cloudflare', '@cf/qwen/qwen1.5-1.8b-chat', 'compatibility', ['COST_EFFECTIVE'], {
    maxContextLength: 4096,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: true,
    inputTokenCost: 0.00000002,
    outputTokenCost: 0.00000002,
    description: 'Workers AI Qwen 1.5 1.8B'
  }, { speed: 5, quality: 1, cost: 5 }),
  entry('cloudflare', '@cf/qwen/qwen1.5-14b-chat-awq', 'compatibility', ['BALANCED'], {
    maxContextLength: 8192,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: true,
    inputTokenCost: 0.0000002,
    outputTokenCost: 0.0000002,
    description: 'Workers AI Qwen 1.5 14B'
  }, { speed: 4, quality: 3, cost: 4 }),
  entry('cloudflare', '@cf/qwen/qwen1.5-7b-chat-awq', 'compatibility', ['BALANCED'], {
    maxContextLength: 8192,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: true,
    inputTokenCost: 0.0000001,
    outputTokenCost: 0.0000001,
    description: 'Workers AI Qwen 1.5 7B'
  }, { speed: 4, quality: 2, cost: 4 }),
  entry('cloudflare', '@cf/google/gemma-4-26b-a4b-it', 'active', ['BALANCED', 'COST_EFFECTIVE', 'VISION', 'TOOL_CALLING', 'LONG_CONTEXT'], {
    maxContextLength: 256000,
    supportsStreaming: true,
    supportsTools: true,
    toolCalling: true,
    supportsVision: true,
    supportsBatching: true,
    inputTokenCost: 0.0000001,
    outputTokenCost: 0.0000003,
    description: 'Workers AI Gemma 4 26B'
  }, { speed: 4, quality: 4, cost: 5 }),
  entry('cloudflare', '@cf/meta/llama-4-scout-17b-16e-instruct', 'active', ['HIGH_PERFORMANCE', 'VISION', 'TOOL_CALLING', 'LONG_CONTEXT'], {
    maxContextLength: 131000,
    supportsStreaming: true,
    supportsTools: true,
    toolCalling: true,
    supportsVision: true,
    supportsBatching: true,
    inputTokenCost: 0.0000003,
    outputTokenCost: 0.0000009,
    description: 'Workers AI Llama 4 Scout'
  }, { speed: 4, quality: 5, cost: 4 }),
  entry('cloudflare', '@cf/meta/llama-3.2-11b-vision-instruct', 'active', ['VISION'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: false,
    supportsVision: true,
    supportsBatching: true,
    inputTokenCost: 0.0000005,
    outputTokenCost: 0.0000005,
    description: 'Workers AI Llama 3.2 Vision'
  }, { speed: 4, quality: 3, cost: 4 }),

  entry('cerebras', 'llama-3.1-8b', 'active', ['COST_EFFECTIVE', 'BALANCED'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: false,
    inputTokenCost: 0.0001,
    outputTokenCost: 0.0001,
    description: 'Cerebras Llama 3.1 8B'
  }, { speed: 5, quality: 3, cost: 5 }),
  entry('cerebras', 'llama-3.3-70b', 'active', ['HIGH_PERFORMANCE', 'BALANCED'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: false,
    inputTokenCost: 0.0006,
    outputTokenCost: 0.0006,
    description: 'Cerebras Llama 3.3 70B'
  }, { speed: 5, quality: 4, cost: 4 }),
  entry('cerebras', 'zai-glm-4.7', 'active', ['HIGH_PERFORMANCE', 'TOOL_CALLING', 'LONG_CONTEXT'], {
    maxContextLength: 131000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0.00225,
    outputTokenCost: 0.00275,
    description: 'Cerebras ZAI-GLM 4.7'
  }, { speed: 4, quality: 5, cost: 2 }),
  entry('cerebras', 'qwen-3-235b-a22b-instruct-2507', 'active', ['TOOL_CALLING', 'BALANCED', 'HIGH_PERFORMANCE', 'LONG_CONTEXT'], {
    maxContextLength: 131000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0.0006,
    outputTokenCost: 0.0012,
    description: 'Cerebras Qwen 3 235B MoE'
  }, { speed: 4, quality: 4, cost: 3 }),

  entry('groq', 'llama-3.3-70b-versatile', 'active', ['HIGH_PERFORMANCE', 'TOOL_CALLING', 'BALANCED'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0.00059,
    outputTokenCost: 0.00079,
    description: 'Groq Llama 3.3 70B Versatile'
  }, { speed: 5, quality: 4, cost: 4 }),
  entry('groq', 'llama-3.1-8b-instant', 'active', ['COST_EFFECTIVE', 'BALANCED'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: false,
    inputTokenCost: 0.00005,
    outputTokenCost: 0.00008,
    description: 'Groq Llama 3.1 8B Instant'
  }, { speed: 5, quality: 2, cost: 5 }),
  entry('groq', 'openai/gpt-oss-120b', 'active', ['TOOL_CALLING', 'BALANCED', 'HIGH_PERFORMANCE'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0.00015,
    outputTokenCost: 0.0006,
    description: 'Groq GPT-OSS 120B'
  }, { speed: 5, quality: 4, cost: 4 }),
] as const;

function isStructuredRequest(request: Partial<LLMRequest> | undefined): boolean {
  return request?.response_format?.type === 'json_object';
}

function usesTools(request: Partial<LLMRequest> | undefined): boolean {
  if (!request) return false;
  return (request.tools?.length ?? 0) > 0 || (request.messages ?? []).some(message =>
    (message.toolCalls?.length ?? 0) > 0 || (message.toolResults?.length ?? 0) > 0
  );
}

function usesVision(request: Partial<LLMRequest> | undefined): boolean {
  return (request?.images?.length ?? 0) > 0;
}

function totalMessageLength(request: Partial<LLMRequest> | undefined): number {
  return (request?.messages ?? []).reduce((sum, message) => sum + message.content.length, 0);
}

function providerRank(provider: ProviderName): number {
  const index = PROVIDER_FALLBACK_ORDER.indexOf(provider);
  return index >= 0 ? index : PROVIDER_FALLBACK_ORDER.length;
}

function scoreUseCase(entry: ModelCatalogEntry, useCase: ModelRecommendationUseCase): number {
  const weights = {
    COST_EFFECTIVE: { cost: 6, speed: 3, quality: 1 },
    HIGH_PERFORMANCE: { cost: 1, speed: 2, quality: 7 },
    BALANCED: { cost: 4, speed: 3, quality: 4 },
    TOOL_CALLING: { cost: 2, speed: 3, quality: 5 },
    LONG_CONTEXT: { cost: 2, speed: 2, quality: 4 },
    VISION: { cost: 2, speed: 2, quality: 4 },
  } as const;

  const weight = weights[useCase];
  let score =
    entry.costScore * weight.cost +
    entry.speedScore * weight.speed +
    entry.qualityScore * weight.quality;

  if (entry.useCases.includes(useCase)) {
    score += 10;
  }

  if (useCase === 'LONG_CONTEXT') {
    score += entry.capabilities.maxContextLength / 20_000;
  }

  return score;
}

function scoreHealth(entry: ModelCatalogEntry, context: ModelSelectionContext): number {
  const providerHealth = context.providerHealth?.[entry.provider];
  if (!providerHealth) return 0;

  let score = providerHealth.healthy === false ? -15 : 5;
  const breakerState = providerHealth.circuitBreaker?.state;

  if (breakerState === 'OPEN') return Number.NEGATIVE_INFINITY;
  if (breakerState === 'DEGRADED') score -= 35;
  if (breakerState === 'RECOVERING') score -= 15;

  const primaryTrafficPct = providerHealth.circuitBreaker?.primaryTrafficPct;
  if (primaryTrafficPct !== undefined && primaryTrafficPct < 1) {
    score -= (1 - primaryTrafficPct) * 20;
  }

  return score;
}

function scoreLedger(entry: ModelCatalogEntry, context: ModelSelectionContext): number {
  if (!context.ledger) return 0;

  let score = 0;
  const providerUtilization = context.ledger.utilizationPct(entry.provider);
  score -= providerUtilization * 35;

  const estimate = context.ledger.getDepletionEstimate(entry.provider);
  if (estimate?.daysRemaining !== null && estimate?.daysRemaining !== undefined) {
    if (estimate.daysRemaining <= 1) score -= 50;
    else if (estimate.daysRemaining <= 3) score -= 30;
    else if (estimate.daysRemaining <= 7) score -= 15;
  }

  return score;
}

function isCompatible(entry: ModelCatalogEntry, context: ModelSelectionContext): boolean {
  const request = context.request;

  if (entry.lifecycle === 'retired') return false;
  if (!request) return true;

  if (usesVision(request) && !entry.capabilities.supportsVision) return false;
  if (usesTools(request) && !entry.capabilities.supportsTools) return false;

  if (isStructuredRequest(request) && !entry.capabilities.supportsTools && entry.provider === 'cloudflare') {
    return false;
  }

  return true;
}

export function inferUseCaseFromRequest(request: Partial<LLMRequest>): ModelRecommendationUseCase {
  if (usesVision(request)) return 'VISION';
  if (usesTools(request) || isStructuredRequest(request)) return 'TOOL_CALLING';

  const messageLength = totalMessageLength(request);
  if ((request.maxTokens ?? 0) >= 4000 || messageLength >= 20_000) {
    return 'LONG_CONTEXT';
  }

  return 'BALANCED';
}

export function rankModels(
  useCase: ModelRecommendationUseCase,
  availableProviders: string[],
  context: ModelSelectionContext = {},
): ModelCatalogEntry[] {
  const allowedProviders = new Set(availableProviders as ProviderName[]);
  const precedencePenalty = useCase === 'COST_EFFECTIVE' ? 2 : 0.01;

  return MODEL_CATALOG
    .filter(entry => allowedProviders.has(entry.provider))
    .filter(entry => isCompatible(entry, context))
    .map(entry => ({
      entry,
      score:
        scoreUseCase(entry, useCase) +
        scoreHealth(entry, context) +
        scoreLedger(entry, context) -
        providerRank(entry.provider) * precedencePenalty,
    }))
    .filter(result => Number.isFinite(result.score))
    .sort((a, b) => b.score - a.score)
    .map(result => result.entry);
}

export function getRecommendedModel(
  useCase: ModelRecommendationUseCase,
  availableProviders: string[],
  context: ModelSelectionContext = {},
): string {
  const ranked = rankModels(useCase, availableProviders, context);

  if (ranked.length === 0) {
    throw new Error('No available providers configured');
  }

  return ranked[0].model;
}

export function getProviderDefaultModel(
  provider: ProviderName,
  request: Partial<LLMRequest> = {},
  context: Omit<ModelSelectionContext, 'request'> = {},
): string {
  const useCase = inferUseCaseFromRequest(request);
  return getRecommendedModel(useCase, [provider], { ...context, request });
}

export function getCatalogEntry(model: string): ModelCatalogEntry | undefined {
  return MODEL_CATALOG.find(entry => entry.model === model);
}

export function getProviderForCatalogModel(model: string): ProviderName | null {
  const entry = getCatalogEntry(model);
  return entry?.provider ?? null;
}

function activeModelsFor(useCase: ModelRecommendationUseCase): string[] {
  return rankModels(useCase, PROVIDER_FALLBACK_ORDER)
    .filter(entry => entry.lifecycle === 'active')
    .map(entry => entry.model);
}

export const MODEL_RECOMMENDATIONS = {
  COST_EFFECTIVE: activeModelsFor('COST_EFFECTIVE'),
  HIGH_PERFORMANCE: activeModelsFor('HIGH_PERFORMANCE'),
  BALANCED: activeModelsFor('BALANCED'),
  TOOL_CALLING: activeModelsFor('TOOL_CALLING'),
  LONG_CONTEXT: activeModelsFor('LONG_CONTEXT'),
  VISION: activeModelsFor('VISION'),
} as const;
