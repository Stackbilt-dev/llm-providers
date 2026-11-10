import type { ModelCatalogEntry, ModelRecommendationUseCase } from '../model-catalog.js';
import { cloudflareReasoningEffortCapability } from '../providers/cloudflare-reasoning.js';
import type { LLMRequest } from '../types.js';

/**
 * For `contextWindowModelSwitch`: the highest-ranked model of `provider` whose context
 * window holds `requiredTokens` (the estimated input plus `maxTokens`), or undefined
 * if it has none.
 *
 * `ranked` should be `rankModels` for the provider, with the request and the use case
 * its planned model was chosen for, so every candidate already passed the request's
 * compatibility checks: vision, tools, structured output, and no retired,
 * thinking-only or explicit-only models. A candidate must also:
 * - be `active` (not a deprecated `compatibility` model);
 * - suit the use case or be tagged `LONG_CONTEXT`, so a plain request isn't moved
 *   onto, say, a research-only model;
 * - for a `reasoning.strict` request, accept the requested effort exactly, which only
 *   Cloudflare models can.
 */
export function largerOwnModel(
  ranked: readonly ModelCatalogEntry[],
  provider: string,
  useCase: ModelRecommendationUseCase,
  requiredTokens: number,
  request: Pick<LLMRequest, 'reasoning'>,
): ModelCatalogEntry | undefined {
  const strictEffort = request.reasoning?.strict === true ? request.reasoning.effort : undefined;
  return ranked.find(entry =>
    entry.provider === provider
    && entry.lifecycle === 'active'
    && requiredTokens <= entry.capabilities.maxContextLength
    && (entry.useCases.includes(useCase) || entry.useCases.includes('LONG_CONTEXT'))
    && (strictEffort === undefined
      || (entry.provider === 'cloudflare' && (cloudflareReasoningEffortCapability(entry.model)?.exact.includes(strictEffort) ?? false)))
  );
}

/** Output tokens a switch target must also hold: a valid `maxTokens`, else none. */
export function requestedOutputTokens(maxTokens: unknown): number {
  return typeof maxTokens === 'number' && Number.isSafeInteger(maxTokens) && maxTokens > 0 ? maxTokens : 0;
}
