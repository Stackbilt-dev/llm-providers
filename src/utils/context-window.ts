import { fitsContextWindow, type ModelCatalogEntry, type ModelRecommendationUseCase } from '../model-catalog.js';
import { cloudflareReasoningEffortCapability } from '../providers/cloudflare-reasoning.js';
import type { LLMRequest } from '../types.js';

/**
 * For `contextWindowModelSwitch`: the highest-ranked model of `provider` whose context
 * window holds the estimated input, or undefined if it has none.
 *
 * `ranked` is the plan's ranking (`rankModels` with the request), so every candidate
 * already passed the request's compatibility checks: vision, tools, structured
 * output, and no retired, thinking-only or explicit-only models. A candidate must
 * also be suited to the use case or tagged `LONG_CONTEXT`, so a plain request isn't
 * moved onto, say, a research-only model. For a `reasoning.strict` request it must
 * accept the requested effort exactly, which only Cloudflare models can.
 */
export function largerOwnModel(
  ranked: readonly ModelCatalogEntry[],
  provider: string,
  useCase: ModelRecommendationUseCase,
  estimatedInputTokens: number,
  request: Pick<LLMRequest, 'reasoning'>,
): string | undefined {
  const strictEffort = request.reasoning?.strict === true ? request.reasoning.effort : undefined;
  return ranked.find(entry =>
    entry.provider === provider
    && fitsContextWindow(entry, estimatedInputTokens)
    && (entry.useCases.includes(useCase) || entry.useCases.includes('LONG_CONTEXT'))
    && (strictEffort === undefined
      || (entry.provider === 'cloudflare' && (cloudflareReasoningEffortCapability(entry.model)?.exact.includes(strictEffort) ?? false)))
  )?.model;
}
