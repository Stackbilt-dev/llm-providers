/**
 * Per-model output-token facts for Workers AI models, read from each model's API
 * schema: the cloudflare-docs source `src/content/workers-ai-models/<slug>.json`,
 * rendered at https://developers.cloudflare.com/workers-ai/models/<slug>/
 * (checked 2026-11-10).
 *
 * - `defaultMaxTokens` is the documented default of `max_tokens` for the Prompt and
 *   Messages inputs, which is what Workers AI applies when a request sends none.
 *   It is absent where the schema states no default. (The Async Batch input, which
 *   this adapter doesn't use, can differ: 256 on Qwen3 30B and SEA-LION.)
 * - `outputTokensParameter` is `max_completion_tokens` where the schema marks
 *   `max_tokens` "Deprecated in favor of max_completion_tokens". The adapter still
 *   sends `max_tokens`, which those schemas accept.
 *
 * No schema states a maximum output; the only documented ceiling is the context
 * window. Models without an entry have no verified output facts.
 */
import type { ModelCapabilities } from '../types.js';

export type CloudflareOutputLimits = Pick<ModelCapabilities, 'defaultMaxTokens' | 'outputTokensParameter'>;

const LEGACY_256: CloudflareOutputLimits = { defaultMaxTokens: 256, outputTokensParameter: 'max_tokens' };
const LEGACY_2000: CloudflareOutputLimits = { defaultMaxTokens: 2000, outputTokensParameter: 'max_tokens' };
// "Deprecated in favor of max_completion_tokens"; neither field has a default.
const COMPLETION_TOKENS: CloudflareOutputLimits = { outputTokensParameter: 'max_completion_tokens' };

const OUTPUT_LIMITS: Readonly<Record<string, CloudflareOutputLimits>> = {
  '@cf/openai/gpt-oss-120b': LEGACY_256,
  '@cf/openai/gpt-oss-20b': LEGACY_256,
  '@cf/meta/llama-3.3-70b-instruct-fp8-fast': LEGACY_256,
  '@cf/qwen/qwen2.5-coder-32b-instruct': LEGACY_256,
  '@cf/mistralai/mistral-small-3.1-24b-instruct': LEGACY_256,
  '@cf/meta/llama-3.2-1b-instruct': LEGACY_256,
  '@cf/meta/llama-3.2-3b-instruct': LEGACY_256,
  '@cf/meta/llama-3.1-8b-instruct-fp8': LEGACY_256,
  '@cf/ibm-granite/granite-4.0-h-micro': LEGACY_256,
  '@cf/meta/llama-4-scout-17b-16e-instruct': LEGACY_256,
  '@cf/meta/llama-3.2-11b-vision-instruct': LEGACY_256,
  '@cf/deepseek-ai/deepseek-r1-distill-qwen-32b': LEGACY_256,
  '@cf/qwen/qwq-32b': LEGACY_256,
  '@cf/qwen/qwen3-30b-a3b-fp8': LEGACY_2000,
  '@cf/aisingapore/gemma-sea-lion-v4-27b-it': LEGACY_2000,
  '@cf/moonshotai/kimi-k2.6': COMPLETION_TOKENS,
  '@cf/moonshotai/kimi-k2.7-code': COMPLETION_TOKENS,
  '@cf/qwen/qwen3.8-27b': COMPLETION_TOKENS,
  '@cf/zai-org/glm-4.7-flash': COMPLETION_TOKENS,
  '@cf/zai-org/glm-5.2': COMPLETION_TOKENS,
  '@cf/zai-org/glm-5.3': COMPLETION_TOKENS,
  '@cf/zai-org/glm-5.3-flash': COMPLETION_TOKENS,
  '@cf/deepseek-ai/deepseek-v4-flash-0731': COMPLETION_TOKENS,
  '@cf/deepseek-ai/deepseek-v4.1-flash-1017': COMPLETION_TOKENS,
  '@cf/deepseek-ai/deepseek-v4-pro-0813': COMPLETION_TOKENS,
  '@cf/google/gemma-4-26b-a4b-it': COMPLETION_TOKENS,
  // Has both fields, but doesn't mark max_tokens deprecated; no default for either.
  '@cf/nvidia/nemotron-3-120b-a12b': { outputTokensParameter: 'max_tokens' },
};

/** The documented output-token facts for a Workers AI model, or undefined if none are verified. */
export function cloudflareOutputLimits(model: string): CloudflareOutputLimits | undefined {
  const limits = OUTPUT_LIMITS[model];
  return limits ? { ...limits } : undefined;
}

/** Defaults at or below this many tokens get a warning when a request sets no `maxTokens`. */
export const SMALL_DEFAULT_MAX_TOKENS = 1024;
