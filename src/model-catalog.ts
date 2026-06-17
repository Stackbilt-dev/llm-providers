import type { CreditLedger } from './utils/credit-ledger.js';
import type { CircuitBreakerState, LLMRequest, ModelCapabilities } from './types.js';

export type ProviderName = 'openai' | 'anthropic' | 'cloudflare' | 'cerebras' | 'groq' | 'nvidia';
export type ModelLifecycle = 'active' | 'compatibility' | 'retired';
export type ModelRecommendationUseCase =
  | 'COST_EFFECTIVE'
  | 'HIGH_PERFORMANCE'
  | 'BALANCED'
  | 'TOOL_CALLING'
  | 'LONG_CONTEXT'
  | 'VISION'
  | 'RESEARCH';

export type ModelWorkloadClass =
  | 'summary'
  | 'planning'
  | 'code_draft'
  | 'long_context'
  | 'tool_loop'
  | 'vision'
  | 'research'
  | 'cost_effective'
  | 'balanced'
  | 'high_performance';

export type ModelPreferenceMap = Partial<
  Record<ModelRecommendationUseCase | ModelWorkloadClass, Partial<Record<ProviderName, string>>>
>;

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
  modelPreferences?: ModelPreferenceMap;
}

const WORKLOAD_USE_CASES: Record<ModelWorkloadClass, ModelRecommendationUseCase> = {
  summary: 'COST_EFFECTIVE',
  planning: 'BALANCED',
  code_draft: 'HIGH_PERFORMANCE',
  long_context: 'LONG_CONTEXT',
  tool_loop: 'TOOL_CALLING',
  vision: 'VISION',
  research: 'RESEARCH',
  cost_effective: 'COST_EFFECTIVE',
  balanced: 'BALANCED',
  high_performance: 'HIGH_PERFORMANCE',
};

const USE_CASES = new Set<ModelRecommendationUseCase>([
  'COST_EFFECTIVE',
  'HIGH_PERFORMANCE',
  'BALANCED',
  'TOOL_CALLING',
  'LONG_CONTEXT',
  'VISION',
  'RESEARCH',
]);

export const PROVIDER_FALLBACK_ORDER: ProviderName[] = [
  'cloudflare',
  'cerebras',
  'groq',
  'nvidia',
  'anthropic',
  'openai',
];

// Quality score scale (1–7):
//   7 = True frontier        (Claude Opus 4.6, top-tier proprietary)
//   6 = Near-frontier MoE    (gpt-oss-120b ≈ o4-mini, kimi-k2.6/k2.7, deepseek-v4-pro, full GLM-4.7)
//   5 = Strong large/MoE     (llama-3.3-70b, qwen3-30b-a3b, qwen2.5-coder-32b, glm-4.7-flash)
//   4 = Capable mid          (llama-4-scout, gpt-oss-20b, haiku-class)
//   3 = Capable small        (mistral-small-24b, llama 8B class)
//   2 = Limited small        (llama 3B class)
//   1 = Toy                  (<2B params)
// Speed/cost also 1–5 (5=fastest/cheapest). Scores inform rankModels() weighted selection.
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
  }, { speed: 2, quality: 7, cost: 1 }),
  entry('anthropic', 'claude-sonnet-4-6-20250618', 'active', ['HIGH_PERFORMANCE', 'LONG_CONTEXT', 'VISION', 'TOOL_CALLING', 'BALANCED'], {
    maxContextLength: 200000,
    supportsStreaming: true,
    supportsTools: true,
    supportsVision: true,
    supportsBatching: false,
    inputTokenCost: 0.003,
    outputTokenCost: 0.015,
    description: 'Claude Sonnet 4.6'
  }, { speed: 3, quality: 6, cost: 3 }),
  entry('anthropic', 'claude-opus-4-20250618', 'compatibility', ['HIGH_PERFORMANCE', 'LONG_CONTEXT'], {
    maxContextLength: 200000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0.015,
    outputTokenCost: 0.075,
    description: 'Claude Opus 4'
  }, { speed: 2, quality: 7, cost: 1 }),
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
    description: 'Workers AI GPT-OSS 120B — OpenAI open-weight 117B/5.1B-active MoE, near o4-mini class (not Phi-4)'
  }, { speed: 4, quality: 6, cost: 4 }),
  entry('cloudflare', '@cf/moonshotai/kimi-k2.6', 'active', ['HIGH_PERFORMANCE', 'TOOL_CALLING', 'LONG_CONTEXT', 'VISION', 'BALANCED'], {
    maxContextLength: 262100,
    supportsStreaming: true,
    supportsTools: true,
    toolCalling: true,
    supportsVision: true,
    supportsBatching: true,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'Workers AI Kimi K2.6 — 1T/32B-active MoE, frontier-scale agent model; vision, 262K context, tool calling (Moonshot self-reported benchmarks)'
  }, { speed: 3, quality: 6, cost: 5 }),
  entry('cloudflare', '@cf/meta/llama-3.3-70b-instruct-fp8-fast', 'active', ['COST_EFFECTIVE', 'BALANCED', 'HIGH_PERFORMANCE', 'TOOL_CALLING'], {
    maxContextLength: 131072,
    supportsStreaming: true,
    supportsTools: true,
    toolCalling: true,
    supportsBatching: true,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'Workers AI Llama 3.3 70B FP8 Fast — primary COST_EFFECTIVE choice; fast FP8 inference, full tool calling'
  }, { speed: 5, quality: 5, cost: 5 }),
  entry('cloudflare', '@cf/openai/gpt-oss-20b', 'active', ['COST_EFFECTIVE', 'TOOL_CALLING'], {
    maxContextLength: 131072,
    supportsStreaming: true,
    supportsTools: true,
    toolCalling: true,
    supportsBatching: true,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'Workers AI GPT-OSS 20B — lightweight tool-calling model; cost-effective complement to gpt-oss-120b'
  }, { speed: 5, quality: 4, cost: 5 }),
  entry('cloudflare', '@cf/qwen/qwen2.5-coder-32b-instruct', 'active', ['COST_EFFECTIVE', 'BALANCED', 'TOOL_CALLING'], {
    maxContextLength: 32768,
    supportsStreaming: true,
    supportsTools: true,
    toolCalling: true,
    supportsBatching: true,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'Workers AI Qwen 2.5 Coder 32B — purpose-built for code generation and completion'
  }, { speed: 4, quality: 5, cost: 5 }),
  entry('cloudflare', '@cf/mistralai/mistral-small-3.1-24b-instruct', 'active', ['BALANCED', 'TOOL_CALLING'], {
    maxContextLength: 131072,
    supportsStreaming: true,
    supportsTools: true,
    toolCalling: true,
    supportsVision: true,
    supportsBatching: true,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'Workers AI Mistral Small 3.1 24B — vision + tool calling; weakest of the active CF set'
  }, { speed: 4, quality: 3, cost: 5 }),
  entry('cloudflare', '@cf/qwen/qwen3-30b-a3b-fp8', 'active', ['BALANCED', 'HIGH_PERFORMANCE'], {
    maxContextLength: 40960,
    supportsStreaming: true,
    supportsTools: true,
    toolCalling: true,
    supportsBatching: false,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'Workers AI Qwen3 30B FP8 — state-of-the-art Qwen3 for balanced/high-performance tasks'
  }, { speed: 4, quality: 5, cost: 5 }),
  entry('cloudflare', '@cf/meta/llama-3.2-1b-instruct', 'active', ['COST_EFFECTIVE'], {
    maxContextLength: 131072,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: true,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'Workers AI Llama 3.2 1B — ultra-cheap tiny model for simple classification and summary'
  }, { speed: 5, quality: 2, cost: 5 }),
  entry('cloudflare', '@cf/meta/llama-3.2-3b-instruct', 'active', ['COST_EFFECTIVE'], {
    maxContextLength: 131072,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: true,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'Workers AI Llama 3.2 3B — cheap small model, step up from 1B'
  }, { speed: 5, quality: 3, cost: 5 }),
  entry('cloudflare', '@cf/moonshotai/kimi-k2.7-code', 'active', ['BALANCED', 'TOOL_CALLING'], {
    maxContextLength: 131072,
    supportsStreaming: true,
    supportsTools: true,
    toolCalling: true,
    supportsBatching: false,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'Workers AI Kimi K2.7 Code — K2.6 retuned for code, 30% fewer thinking tokens; SWE-bench 78.2 (Moonshot self-reported)'
  }, { speed: 3, quality: 6, cost: 5 }),
  entry('cloudflare', '@cf/zai-org/glm-4.7-flash', 'active', ['RESEARCH'], {
    maxContextLength: 131072,
    supportsStreaming: true,
    supportsTools: true,
    toolCalling: true,
    supportsBatching: true,
    inputTokenCost: 0,
    outputTokenCost: 0,
    thinkingModel: true,
    description: 'Workers AI GLM-4.7-Flash — 30B/3B-active MoE CoT model; emits <think> reasoning traces, RESEARCH-only routing'
  }, { speed: 5, quality: 5, cost: 5 }),
  entry('cloudflare', 'deepseek/deepseek-v4-pro', 'active', ['HIGH_PERFORMANCE', 'BALANCED'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: true,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'Workers AI DeepSeek V4 Pro — frontier-class reasoning and coding, near gpt-oss-120b tier'
  }, { speed: 3, quality: 6, cost: 5 }),
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
  entry('cloudflare', '@cf/google/gemma-4-26b-a4b-it', 'active', ['COST_EFFECTIVE', 'VISION', 'TOOL_CALLING', 'LONG_CONTEXT'], {
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
    description: 'Workers AI Llama 4 Scout — 109B/17B-active MoE; multimodal + 10M ctx, capability ≈ Llama 3.3 70B dense'
  }, { speed: 4, quality: 4, cost: 4 }),
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
  entry('cloudflare', '@cf/nvidia/nemotron-3-120b-a12b', 'active', ['HIGH_PERFORMANCE', 'TOOL_CALLING', 'LONG_CONTEXT'], {
    maxContextLength: 256000,
    supportsStreaming: true,
    supportsTools: true,
    toolCalling: true,
    supportsBatching: true,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'Workers AI NVIDIA Nemotron-3 120B — hybrid MoE, parallel function calling, 256K context'
  }, { speed: 3, quality: 5, cost: 5 }),
  entry('cloudflare', '@cf/deepseek-ai/deepseek-r1-distill-qwen-32b', 'active', ['RESEARCH'], {
    maxContextLength: 80000,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: true,
    inputTokenCost: 0,
    outputTokenCost: 0,
    thinkingModel: true,
    description: 'Workers AI DeepSeek-R1-Distill-Qwen-32B — chain-of-thought reasoning model; outputs thinking traces'
  }, { speed: 3, quality: 5, cost: 5 }),
  entry('cloudflare', '@cf/qwen/qwq-32b', 'active', ['RESEARCH'], {
    maxContextLength: 24000,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: true,
    inputTokenCost: 0,
    outputTokenCost: 0,
    thinkingModel: true,
    description: 'Workers AI QwQ-32B — native thinking/reasoning model; outputs chain-of-thought traces'
  }, { speed: 4, quality: 5, cost: 5 }),
  // Third-party model via CF binding — requires external Anthropic billing through CF dashboard,
  // NOT a native CF Workers AI model. Excluded from standard routing tags (no HIGH_PERFORMANCE)
  // so auto-routing never selects it. Only reached by explicit model name request.
  entry('cloudflare', 'anthropic/claude-opus-4.8', 'active', ['LONG_CONTEXT'], {
    maxContextLength: 1000000,
    supportsStreaming: false,
    supportsTools: false,
    supportsBatching: false,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'Cloudflare-managed Anthropic Claude Opus 4.8 — 1M context, external billing via CF dashboard; not a native CF model'
  }, { speed: 2, quality: 7, cost: 3 }),
  entry('cloudflare', '@cf/zai-org/glm-5.2', 'active', ['HIGH_PERFORMANCE', 'TOOL_CALLING', 'LONG_CONTEXT'], {
    maxContextLength: 262144,
    supportsStreaming: true,
    supportsTools: true,
    toolCalling: true,
    supportsBatching: true,
    inputTokenCost: 0.0000014,
    outputTokenCost: 0.0000044,
    description: 'Workers AI GLM-5.2 — Z.ai flagship agentic coder; 262K ctx, direct-response (not CoT), function calling'
  }, { speed: 4, quality: 6, cost: 3 }),
  entry('cloudflare', '@cf/moonshotai/kimi-k2.5', 'retired', ['HIGH_PERFORMANCE', 'TOOL_CALLING', 'LONG_CONTEXT', 'VISION'], {
    maxContextLength: 256000,
    supportsStreaming: true,
    supportsTools: true,
    toolCalling: true,
    supportsVision: true,
    supportsBatching: true,
    inputTokenCost: 0.0000006,
    outputTokenCost: 0.000003,
    description: 'Workers AI Kimi K2.5 — EOL 2026-05-30; predecessor to K2.6, 256K ctx, vision + tool calling'
  }, { speed: 3, quality: 5, cost: 4 }),
  entry('cloudflare', '@cf/google/gemma-3-12b-it', 'retired', ['COST_EFFECTIVE', 'VISION', 'LONG_CONTEXT'], {
    maxContextLength: 80000,
    supportsStreaming: true,
    supportsTools: false,
    supportsVision: true,
    supportsBatching: true,
    inputTokenCost: 0.00000035,
    outputTokenCost: 0.00000056,
    description: 'Workers AI Gemma 3 12B IT — EOL 2026-05-30; multimodal, 140+ languages, LoRA support'
  }, { speed: 4, quality: 3, cost: 4 }),

  entry('cerebras', 'llama-3.1-8b', 'compatibility', ['COST_EFFECTIVE', 'BALANCED'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: false,
    inputTokenCost: 0.0001,
    outputTokenCost: 0.0001,
    description: 'Cerebras Llama 3.1 8B (deprecated 2026-05-27)'
  }, { speed: 5, quality: 3, cost: 5 }),
  entry('cerebras', 'llama-3.3-70b', 'retired', ['HIGH_PERFORMANCE', 'BALANCED'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: false,
    inputTokenCost: 0.0006,
    outputTokenCost: 0.0006,
    description: 'Cerebras Llama 3.3 70B (retired)'
  }, { speed: 5, quality: 4, cost: 4 }),
  entry('cerebras', 'zai-glm-4.7', 'active', ['HIGH_PERFORMANCE', 'TOOL_CALLING', 'LONG_CONTEXT'], {
    maxContextLength: 131072,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    supportsPromptCache: true,
    inputTokenCost: 0.00225,
    outputTokenCost: 0.00275,
    description: 'Cerebras ZAI-GLM 4.7 — full (non-flash) GLM-4.7; reasoning, tool calling, predicted outputs; patch task default'
  }, { speed: 4, quality: 6, cost: 2 }),
  entry('cerebras', 'qwen-3-235b-a22b-instruct-2507', 'compatibility', ['TOOL_CALLING', 'BALANCED', 'HIGH_PERFORMANCE', 'LONG_CONTEXT'], {
    maxContextLength: 131072,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    supportsPromptCache: true,
    inputTokenCost: 0.0006,
    outputTokenCost: 0.0012,
    description: 'Cerebras Qwen 3 235B MoE (deprecated 2026-05-27)'
  }, { speed: 4, quality: 4, cost: 3 }),
  entry('cerebras', 'openai/gpt-oss-120b', 'active', ['TOOL_CALLING', 'BALANCED', 'HIGH_PERFORMANCE'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    supportsPromptCache: true,
    inputTokenCost: 0.00015,
    outputTokenCost: 0.0006,
    description: 'Cerebras GPT-OSS 120B — OpenAI open-weight 117B/5.1B-active MoE; tool calling, predicted outputs'
  }, { speed: 5, quality: 6, cost: 4 }),

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
    // Groq-hosted gpt-oss runs built-in browser_search + code_interpreter
    // server-side (issue #69). The same model string on Cerebras does NOT —
    // this capability is what disambiguates the cross-provider catalog collision.
    supportsBuiltInTools: ['web_search', 'code_interpreter'],
    description: 'Groq GPT-OSS 120B'
  }, { speed: 5, quality: 4, cost: 4 }),
  // Groq Compound systems (issue #69): server-side agentic systems that route
  // between backing models and natively run all five built-in tools. Same
  // `/chat/completions` endpoint — no new HTTP path. Token costs are an estimate
  // (roll-up of Llama 4 Scout 17B / GPT-OSS 120B); built-in surcharges
  // (web_search $5/1k req, code_interpreter $0.18/hr, browser_automation
  // $0.08/hr) are billed separately and are NOT token-tracked here.
  //
  // Tagged RESEARCH-only — deliberately NOT TOOL_CALLING/HIGH_PERFORMANCE/etc.
  // despite supporting function tools. Compound autonomously runs web_search as
  // part of its routing, so any generic traffic that landed here risks
  // unpredictable per-search surcharges. Keeping them out of the generic
  // recommendation pools means they are only auto-selected when a caller asks
  // for RESEARCH explicitly (or pins the model, the intended path).
  //
  // costScore 1 (most expensive) on both reflects that surcharge-inflated
  // effective cost — and structurally keeps them below the generic groq lineup
  // in every non-RESEARCH pool, including LONG_CONTEXT where no groq model
  // carries a tag bonus and the slightly larger window would otherwise edge them
  // ahead. Full Compound outranks Mini for RESEARCH via its higher qualityScore.
  entry('groq', 'groq/compound', 'active', ['RESEARCH'], {
    maxContextLength: 131072,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0.00015,
    outputTokenCost: 0.0006,
    supportsBuiltInTools: ['web_search', 'visit_website', 'browser_automation', 'code_interpreter', 'wolfram_alpha'],
    supportsVision: false,
    description: 'Groq Compound — agentic system routing Llama 4 Scout (17B) / GPT-OSS 120B by query complexity; runs all five built-in tools server-side. Token cost estimated; built-in tool surcharges billed separately, not token-tracked.'
  }, { speed: 4, quality: 5, cost: 1 }),
  entry('groq', 'groq/compound-mini', 'active', ['RESEARCH'], {
    maxContextLength: 131072,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0.0001,
    outputTokenCost: 0.0004,
    supportsBuiltInTools: ['web_search', 'visit_website', 'browser_automation', 'code_interpreter', 'wolfram_alpha'],
    supportsVision: false,
    description: 'Groq Compound Mini — lower-latency single-pass variant of Compound; same five built-in tools. Token cost estimated; built-in tool surcharges billed separately, not token-tracked.'
  }, { speed: 5, quality: 4, cost: 1 }),

  // NVIDIA NIM — costs are zero placeholders (dev-tier credit-based; production
  // pricing varies by model). Update inputTokenCost/outputTokenCost via CreditLedger.
  entry('nvidia', 'meta/llama-3.3-70b-instruct', 'active', ['HIGH_PERFORMANCE', 'TOOL_CALLING', 'BALANCED', 'LONG_CONTEXT'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'Meta Llama 3.3 70B Instruct on NVIDIA NIM'
  }, { speed: 3, quality: 4, cost: 3 }),
  entry('nvidia', 'meta/llama-4-maverick-17b-128e-instruct', 'active', ['HIGH_PERFORMANCE', 'TOOL_CALLING', 'LONG_CONTEXT'], {
    maxContextLength: 1048576,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'Meta Llama 4 Maverick 17B 128E MoE Instruct on NVIDIA NIM'
  }, { speed: 4, quality: 5, cost: 3 }),
  entry('nvidia', 'nvidia/llama-3.1-nemotron-70b-instruct', 'active', ['HIGH_PERFORMANCE', 'TOOL_CALLING', 'BALANCED'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'NVIDIA Llama 3.1 Nemotron 70B Instruct — NVIDIA-optimized accuracy'
  }, { speed: 3, quality: 5, cost: 3 }),
  entry('nvidia', 'nvidia/llama-3.3-nemotron-super-49b-v1', 'active', ['BALANCED', 'TOOL_CALLING', 'HIGH_PERFORMANCE'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'NVIDIA Llama 3.3 Nemotron Super 49B v1 — efficiency-optimized'
  }, { speed: 4, quality: 4, cost: 3 }),
  entry('nvidia', 'deepseek-ai/deepseek-v4-flash', 'active', ['COST_EFFECTIVE', 'BALANCED'], {
    maxContextLength: 65536,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: false,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'DeepSeek V4 Flash on NVIDIA NIM'
  }, { speed: 5, quality: 3, cost: 4 }),
  entry('nvidia', 'meta/llama-3.1-70b-instruct', 'active', ['HIGH_PERFORMANCE', 'TOOL_CALLING', 'BALANCED'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'Meta Llama 3.1 70B Instruct on NVIDIA NIM'
  }, { speed: 3, quality: 4, cost: 3 }),
  entry('nvidia', 'nvidia/llama-3.1-nemotron-ultra-253b-v1', 'active', ['HIGH_PERFORMANCE', 'LONG_CONTEXT'], {
    maxContextLength: 128000,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'NVIDIA Llama 3.1 Nemotron Ultra 253B v1 — maximum quality'
  }, { speed: 1, quality: 5, cost: 3 }),
  entry('nvidia', 'mistralai/mistral-large-2-instruct', 'active', ['HIGH_PERFORMANCE', 'TOOL_CALLING', 'BALANCED'], {
    maxContextLength: 131072,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'Mistral Large 2 Instruct on NVIDIA NIM'
  }, { speed: 3, quality: 4, cost: 3 }),
  entry('nvidia', 'deepseek-ai/deepseek-v4-pro', 'active', ['HIGH_PERFORMANCE', 'BALANCED'], {
    maxContextLength: 65536,
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: false,
    inputTokenCost: 0,
    outputTokenCost: 0,
    description: 'DeepSeek V4 Pro on NVIDIA NIM'
  }, { speed: 2, quality: 5, cost: 3 }),
] as const;

function isStructuredRequest(request: Partial<LLMRequest> | undefined): boolean {
  const type = request?.response_format?.type;
  return type === 'json_object' || type === 'json_schema';
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
    // Research favours model quality strongly over cost/speed — the workload is
    // authoritative-source gathering, where a better backing model and richer
    // built-in-tool routing matter more than latency. Quality-dominant so full
    // Compound ranks above Compound Mini for the same RESEARCH request.
    RESEARCH: { cost: 1, speed: 2, quality: 6 },
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

function isCompatible(entry: ModelCatalogEntry, context: ModelSelectionContext, useCase?: ModelRecommendationUseCase): boolean {
  const request = context.request;

  if (entry.lifecycle === 'retired') return false;

  // Thinking/reasoning models output chain-of-thought traces rather than direct
  // responses. Exclude them from every pool except RESEARCH, where the caller
  // explicitly handles raw reasoning output.
  if (entry.capabilities.thinkingModel && useCase !== 'RESEARCH') return false;

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
    .filter(entry => isCompatible(entry, context, useCase))
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

export function normalizeModelWorkload(
  workload: ModelRecommendationUseCase | ModelWorkloadClass
): ModelRecommendationUseCase {
  if (USE_CASES.has(workload as ModelRecommendationUseCase)) {
    return workload as ModelRecommendationUseCase;
  }

  const normalized = workload.toLowerCase() as ModelWorkloadClass;
  const useCase = WORKLOAD_USE_CASES[normalized];
  if (!useCase) {
    throw new Error(`Unknown model workload class: ${workload}`);
  }

  return useCase;
}

function getPreferredModelForWorkload(
  workload: ModelRecommendationUseCase | ModelWorkloadClass,
  useCase: ModelRecommendationUseCase,
  availableProviders: string[],
  context: ModelSelectionContext,
): string | undefined {
  const preferences = context.modelPreferences;
  if (!preferences) return undefined;

  const workloadPrefs = preferences[workload];
  const useCasePrefs = preferences[useCase];
  for (const provider of availableProviders as ProviderName[]) {
    const preferred = workloadPrefs?.[provider] ?? useCasePrefs?.[provider];
    if (preferred) return preferred;
  }

  return undefined;
}

export function getRecommendedModelForWorkload(
  workload: ModelRecommendationUseCase | ModelWorkloadClass,
  availableProviders: string[],
  context: ModelSelectionContext = {},
): string {
  const useCase = normalizeModelWorkload(workload);
  const preferred = getPreferredModelForWorkload(workload, useCase, availableProviders, context);
  return preferred ?? getRecommendedModel(useCase, availableProviders, context);
}

export function getProviderDefaultModelForWorkload(
  provider: ProviderName,
  workload: ModelRecommendationUseCase | ModelWorkloadClass,
  context: Omit<ModelSelectionContext, 'request'> = {},
): string {
  return getRecommendedModelForWorkload(workload, [provider], context);
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

/**
 * All providers that serve this exact model string, ordered by
 * `PROVIDER_FALLBACK_ORDER`. Some model strings are hosted by more than one
 * provider (e.g. `openai/gpt-oss-120b` runs on both Cerebras and Groq); the
 * singular `getProviderForCatalogModel` returns only the first catalog match,
 * which hides the collision. Routing uses this plural form to choose a
 * configured / capability-matching provider deterministically.
 */
export function getProvidersForCatalogModel(model: string): ProviderName[] {
  const providers = MODEL_CATALOG
    .filter(entry => entry.model === model)
    .map(entry => entry.provider);
  const unique = [...new Set(providers)];
  return unique.sort(
    (a, b) => PROVIDER_FALLBACK_ORDER.indexOf(a) - PROVIDER_FALLBACK_ORDER.indexOf(b),
  );
}

/**
 * Whether a specific (model, provider) pair advertises a given built-in tool,
 * or any built-in tool when `tool` is omitted. Used to steer a built-in-tools
 * request to the capable provider when a model string is hosted by several.
 */
export function modelSupportsBuiltInTools(
  model: string,
  provider: ProviderName,
  tool?: string,
): boolean {
  const entry = MODEL_CATALOG.find(e => e.model === model && e.provider === provider);
  const supported = entry?.capabilities.supportsBuiltInTools;
  if (!supported || supported.length === 0) return false;
  // `tool` is an open-ended string (request- or response-side identifier);
  // widen the typed array rather than narrowing the argument.
  return tool === undefined ? true : (supported as readonly string[]).includes(tool);
}

/**
 * Pre-flight routing snapshot for a request.
 *
 * Gateways and agent orchestrators call this BEFORE dispatching to understand
 * what the catalog engine would do: which use-case tier the request falls into,
 * which model/provider would be selected, estimated token load, and any
 * lifecycle warnings on the target model. Pair with `metadata.useCase` on the
 * `LLMRequest` to pass the gateway's classification directly into the factory's
 * `resolveUseCase()` path so model selection stays catalog-driven at runtime.
 */
export interface RoutingInfo {
  /** Use case inferred from the request's tools, vision, and message length */
  useCase: ModelRecommendationUseCase;
  /** Provider that would serve the request */
  provider: ProviderName;
  /** Model string that would be selected */
  model: string;
  /** Full catalog entry for the selected model, or undefined if not in catalog */
  catalogEntry: ModelCatalogEntry | undefined;
  /** Heuristic input token estimate (~4 chars/token for English text) */
  estimatedInputTokens: number;
  /** True if the request has tools attached or tool history in messages */
  requiresTools: boolean;
  /** True if the request has image inputs */
  requiresVision: boolean;
  /** True if request.stream is set */
  requestsStreaming: boolean;
  /** Catalog lifecycle of the target model */
  modelLifecycle: ModelLifecycle | 'unknown';
  /**
   * Human-readable deprecation warning if the target model is not on active
   * lifecycle, or if its catalog description contains a deprecation date.
   * Undefined when the model is active with no known end-of-life date.
   */
  deprecationWarning: string | undefined;
}

function deprecationNotice(entry: ModelCatalogEntry): string | undefined {
  if (entry.lifecycle === 'active') {
    // Some active-lifecycle entries carry a near-term deprecation date in their
    // description (e.g. "deprecated 2026-05-27"). Surface it as a warning.
    const dateMatch = entry.capabilities.description.match(/deprecated\s+([\d]{4}-[\d]{2}-[\d]{2})/i);
    return dateMatch ? `${entry.model} deprecates ${dateMatch[1]} — plan migration` : undefined;
  }
  const dateMatch = entry.capabilities.description.match(/deprecated\s+([\d]{4}-[\d]{2}-[\d]{2})/i);
  if (dateMatch) return `${entry.model} deprecates ${dateMatch[1]} — plan migration`;
  if (entry.lifecycle === 'retired') return `${entry.model} is retired — update your model configuration`;
  return `${entry.model} is in compatibility mode — prefer an active model for new deployments`;
}

/**
 * Return a routing snapshot for a request without dispatching it.
 *
 * @param request - The request to analyse (may be partial; model optional).
 * @param availableProviders - Providers to consider. Defaults to the full
 *   fallback order. Pass the list of actually-configured providers for an
 *   accurate recommendation.
 * @param context - Optional health and ledger context for ranking.
 */
export function getRoutingInfo(
  request: Partial<LLMRequest>,
  availableProviders: ProviderName[] = [...PROVIDER_FALLBACK_ORDER],
  context: ModelSelectionContext = {}
): RoutingInfo {
  const requiresTools = usesTools(request);
  const requiresVision = usesVision(request);
  const requestsStreaming = request.stream === true;
  const useCase = inferUseCaseFromRequest(request);

  const charCount = totalMessageLength(request) + (request.systemPrompt?.length ?? 0);
  const estimatedInputTokens = Math.max(1, Math.ceil(charCount / 4));

  // When a specific model is pinned, report on that model rather than recommending.
  if (request.model) {
    const entry = getCatalogEntry(request.model);
    const provider = (entry?.provider ??
      getProviderForCatalogModel(request.model) ??
      availableProviders[0]) as ProviderName;
    return {
      useCase,
      provider,
      model: request.model,
      catalogEntry: entry,
      estimatedInputTokens,
      requiresTools,
      requiresVision,
      requestsStreaming,
      modelLifecycle: entry?.lifecycle ?? 'unknown',
      deprecationWarning: entry ? deprecationNotice(entry) : undefined,
    };
  }

  const ranked = rankModels(useCase, availableProviders, { ...context, request });
  const best = ranked[0];

  if (!best) {
    return {
      useCase,
      provider: availableProviders[0] ?? ('anthropic' as ProviderName),
      model: '',
      catalogEntry: undefined,
      estimatedInputTokens,
      requiresTools,
      requiresVision,
      requestsStreaming,
      modelLifecycle: 'unknown',
      deprecationWarning: undefined,
    };
  }

  return {
    useCase,
    provider: best.provider,
    model: best.model,
    catalogEntry: best,
    estimatedInputTokens,
    requiresTools,
    requiresVision,
    requestsStreaming,
    modelLifecycle: best.lifecycle,
    deprecationWarning: deprecationNotice(best),
  };
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
  RESEARCH: activeModelsFor('RESEARCH'),
} as const;
