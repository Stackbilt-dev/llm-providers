/**
 * Per-model mapping of `LLMRequest.reasoning.effort` onto Workers AI inputs.
 *
 * Workers AI reasoning models do not share one effort vocabulary: each model page
 * documents its own `reasoning_effort` levels (and aliases that can map a "lower"
 * value to the model's maximum), and some models use `chat_template_kwargs`
 * instead. Every entry here cites the model page it was read from. Models without
 * an entry get nothing sent and a not-applied report.
 */
import type {
  ReasoningEffortCapability,
  ReasoningEffortLevel,
  ReasoningEffortReport,
} from '../types.js';

/** Request fields a rule adds to the Workers AI chat input. */
export interface CloudflareReasoningInput {
  reasoning_effort?: string;
  chat_template_kwargs?: Record<string, boolean>;
}

interface EffortRule {
  /** Native level label reported back to the caller. */
  native: string;
  /** Fields sent to Workers AI; omitted when the provider default already matches. */
  input?: CloudflareReasoningInput;
  exact: boolean;
}

interface ModelReasoningMapping {
  source: string;
  verifiedAt: string;
  levels: Partial<Record<ReasoningEffortLevel, EffortRule>>;
}

const VERIFIED_AT = '2026-11-06';
const page = (slug: string) => `https://developers.cloudflare.com/workers-ai/models/${slug}/`;

const effort = (native: string, exact: boolean): EffortRule => ({
  native,
  input: { reasoning_effort: native },
  exact,
});

// gpt-oss: model page lists low / medium (default) / high; Cloudflare's own
// workers-ai-provider (3.1.12) forwards top-level `reasoning_effort` for chat input.
const GPT_OSS: Omit<ModelReasoningMapping, 'source'> = {
  verifiedAt: VERIFIED_AT,
  levels: { low: effort('low', true), medium: effort('medium', true), high: effort('high', true) },
};

// GLM-5.3 / 5.3 Flash: low / high / max (default). Reasoning cannot be disabled;
// the documented aliases map none and medium to max, so never forward those.
const GLM_5_3: Omit<ModelReasoningMapping, 'source'> = {
  verifiedAt: VERIFIED_AT,
  levels: { low: effort('low', true), medium: effort('high', false), high: effort('high', true) },
};

// DeepSeek V4 Flash / V4.1 Flash / V4 Pro: none / low / high (default) / max.
const DEEPSEEK_V4: Omit<ModelReasoningMapping, 'source'> = {
  verifiedAt: VERIFIED_AT,
  levels: {
    none: effort('none', true),
    low: effort('low', true),
    medium: effort('high', false),
    high: effort('high', true),
  },
};

const MAPPINGS: Record<string, ModelReasoningMapping> = {
  '@cf/openai/gpt-oss-20b': { ...GPT_OSS, source: page('gpt-oss-20b') },
  '@cf/openai/gpt-oss-120b': { ...GPT_OSS, source: page('gpt-oss-120b') },
  '@cf/zai-org/glm-5.3': { ...GLM_5_3, source: page('glm-5.3') },
  '@cf/zai-org/glm-5.3-flash': { ...GLM_5_3, source: page('glm-5.3-flash') },
  // GLM-5.2: none / high / max (default); none turns reasoning off. Nothing lower than
  // high exists, so low and medium send high (still below the max default).
  '@cf/zai-org/glm-5.2': {
    source: page('glm-5.2'),
    verifiedAt: VERIFIED_AT,
    levels: {
      none: effort('none', true),
      low: effort('high', false),
      medium: effort('high', false),
      high: effort('high', true),
    },
  },
  '@cf/deepseek-ai/deepseek-v4-flash-0731': { ...DEEPSEEK_V4, source: page('deepseek-v4-flash-0731') },
  '@cf/deepseek-ai/deepseek-v4.1-flash-1017': {
    ...DEEPSEEK_V4, source: page('deepseek-v4.1-flash-1017'), verifiedAt: '2026-11-08',
  },
  '@cf/deepseek-ai/deepseek-v4-pro-0813': { ...DEEPSEEK_V4, source: page('deepseek-v4-pro-0813') },
  // Kimi K2.6: none / high (default) only.
  '@cf/moonshotai/kimi-k2.6': {
    source: page('kimi-k2.6'),
    verifiedAt: VERIFIED_AT,
    levels: {
      none: effort('none', true),
      low: effort('high', false),
      medium: effort('high', false),
      high: effort('high', true),
    },
  },
  // Qwen 3.8 27B: low / medium / xhigh (default); no off level.
  '@cf/qwen/qwen3.8-27b': {
    source: page('qwen3.8-27b'),
    verifiedAt: VERIFIED_AT,
    levels: { low: effort('low', true), medium: effort('medium', true), high: effort('xhigh', false) },
  },
  // Nemotron 3: ignores top-level reasoning_effort; modes are chat-template flags.
  // The page documents three modes: normal reasoning, low reasoning (enable_thinking
  // plus low_effort) and reasoning off (enable_thinking false). It does not say which
  // is the default, so medium and high send normal reasoning explicitly; there is one
  // level above low, so neither is exact.
  '@cf/nvidia/nemotron-3-120b-a12b': {
    source: page('nemotron-3-120b-a12b'),
    verifiedAt: VERIFIED_AT,
    levels: {
      none: { native: 'enable_thinking=false', input: { chat_template_kwargs: { enable_thinking: false } }, exact: true },
      low: {
        native: 'low_effort',
        input: { chat_template_kwargs: { enable_thinking: true, low_effort: true } },
        exact: true,
      },
      medium: { native: 'enable_thinking=true', input: { chat_template_kwargs: { enable_thinking: true } }, exact: false },
      high: { native: 'enable_thinking=true', input: { chat_template_kwargs: { enable_thinking: true } }, exact: false },
    },
  },
};

const LEVEL_ORDER: readonly ReasoningEffortLevel[] = ['none', 'low', 'medium', 'high'];

/** Public capability view of the mapping for one model, if any. */
export function cloudflareReasoningEffortCapability(model: string): ReasoningEffortCapability | undefined {
  if (!Object.hasOwn(MAPPINGS, model)) return undefined;
  const mapping = MAPPINGS[model];
  return {
    exact: LEVEL_ORDER.filter(level => mapping.levels[level]?.exact === true),
    source: mapping.source,
    verifiedAt: mapping.verifiedAt,
  };
}

export interface ResolvedCloudflareReasoning {
  input?: CloudflareReasoningInput;
  report: ReasoningEffortReport;
}

/** Resolve a requested effort for a model. Returns undefined when no effort was requested. */
export function resolveCloudflareReasoningEffort(
  model: string,
  requested: ReasoningEffortLevel | undefined,
): ResolvedCloudflareReasoning | undefined {
  if (requested === undefined) return undefined;
  const mapping = Object.hasOwn(MAPPINGS, model) ? MAPPINGS[model] : undefined;
  const rule = mapping && LEVEL_ORDER.includes(requested) && Object.hasOwn(mapping.levels, requested)
    ? mapping.levels[requested]
    : undefined;
  if (!rule) {
    return { report: { requested, applied: null, exact: false } };
  }
  return {
    ...(rule.input ? { input: rule.input } : {}),
    report: { requested, applied: rule.native, exact: rule.exact },
  };
}
