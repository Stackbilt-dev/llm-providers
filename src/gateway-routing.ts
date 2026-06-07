import {
  canonicalToLLMRequest,
  normalizeLLMRequest,
  type CanonicalDegradation,
  type CanonicalLLMRequest,
} from './canonical.js';
import {
  getRoutingInfo,
  type ModelSelectionContext,
  type ProviderName,
  type RoutingInfo,
} from './model-catalog.js';
import type { BuiltInToolType, CacheHints, LLMRequest, ModelCapabilities } from './types.js';

export interface GatewayRouteRequirements {
  streaming: boolean;
  toolCalling: boolean;
  structuredOutput: boolean;
  vision: boolean;
  builtInTools: BuiltInToolType[];
  lora: boolean;
}

export interface GatewayRouteCapabilityReport {
  supportsStreaming: boolean;
  supportsTools: boolean;
  supportsVision: boolean;
  supportsPromptCache: boolean;
  supportsBuiltInTools: BuiltInToolType[];
  maxContextLength: number | undefined;
}

export interface GatewayRouteCachePlan {
  strategy: CacheHints['strategy'];
  providerPromptCache: boolean;
  responseCache: boolean;
  key: string | undefined;
  ttl: CacheHints['ttl'];
  sessionId: string | undefined;
  cacheablePrefix: CacheHints['cacheablePrefix'];
}

export interface GatewayRoutePlan {
  canonicalRequest: CanonicalLLMRequest;
  legacyRequest: LLMRequest;
  routing: RoutingInfo;
  selectedProvider: ProviderName;
  selectedModel: string;
  useCase: RoutingInfo['useCase'];
  estimatedInputTokens: number;
  requirements: GatewayRouteRequirements;
  capabilities: GatewayRouteCapabilityReport;
  cache: GatewayRouteCachePlan;
  degradations: CanonicalDegradation[];
  warnings: string[];
}

export function getGatewayRoutePlan(
  request: LLMRequest | CanonicalLLMRequest,
  availableProviders?: ProviderName[],
  context: ModelSelectionContext = {}
): GatewayRoutePlan {
  const canonicalRequest = normalizeLLMRequest(request);
  const legacyRequest = canonicalToLLMRequest(canonicalRequest);
  const routing = getRoutingInfo(legacyRequest, availableProviders, context);
  const requirements = routeRequirements(canonicalRequest);
  const capabilities = capabilityReport(routing.catalogEntry?.capabilities);
  const cache = cachePlan(canonicalRequest.metadata?.cache, capabilities);
  const degradations = routeDegradations(requirements, capabilities, routing.provider);
  const warnings = routeWarnings(routing, requirements, capabilities, cache);

  return {
    canonicalRequest,
    legacyRequest,
    routing,
    selectedProvider: routing.provider,
    selectedModel: routing.model,
    useCase: routing.useCase,
    estimatedInputTokens: routing.estimatedInputTokens,
    requirements,
    capabilities,
    cache,
    degradations,
    warnings,
  };
}

function routeRequirements(request: CanonicalLLMRequest): GatewayRouteRequirements {
  const builtInTools = request.builtInTools?.map(tool => tool.type) ?? [];

  return {
    streaming: request.stream === true,
    toolCalling: request.requirements?.toolCalling === true || (request.tools?.length ?? 0) > 0,
    structuredOutput: request.requirements?.structuredOutput === true || request.output?.kind === 'json_object' || request.output?.kind === 'json_schema',
    vision: request.requirements?.vision === true || (request.media?.length ?? 0) > 0,
    builtInTools,
    lora: request.providerOptions?.cloudflare?.lora !== undefined,
  };
}

function capabilityReport(capabilities: ModelCapabilities | undefined): GatewayRouteCapabilityReport {
  return {
    supportsStreaming: capabilities?.supportsStreaming ?? false,
    supportsTools: capabilities?.supportsTools ?? false,
    supportsVision: capabilities?.supportsVision ?? false,
    supportsPromptCache: capabilities?.supportsPromptCache ?? false,
    supportsBuiltInTools: capabilities?.supportsBuiltInTools ?? [],
    maxContextLength: capabilities?.maxContextLength,
  };
}

function cachePlan(
  hints: CacheHints | undefined,
  capabilities: GatewayRouteCapabilityReport
): GatewayRouteCachePlan {
  const strategy = hints ? hints.strategy ?? 'provider-prefix' : undefined;
  return {
    strategy,
    providerPromptCache: strategy === 'provider-prefix' || strategy === 'both'
      ? capabilities.supportsPromptCache
      : false,
    responseCache: strategy === 'response' || strategy === 'both',
    key: hints?.key,
    ttl: hints?.ttl,
    sessionId: hints?.sessionId,
    cacheablePrefix: hints?.cacheablePrefix,
  };
}

function routeDegradations(
  requirements: GatewayRouteRequirements,
  capabilities: GatewayRouteCapabilityReport,
  provider: ProviderName
): CanonicalDegradation[] {
  const degradations: CanonicalDegradation[] = [];

  if (requirements.streaming && !capabilities.supportsStreaming) {
    degradations.push({
      capability: 'streaming',
      action: 'failed',
      reason: 'selected model does not advertise streaming support',
    });
  }

  if (requirements.toolCalling && !capabilities.supportsTools) {
    degradations.push({
      capability: 'toolCalling',
      action: 'failed',
      reason: 'selected model does not advertise function-tool support',
    });
  }

  if (requirements.structuredOutput && !capabilities.supportsTools) {
    degradations.push({
      capability: 'structuredOutput',
      action: 'emulated',
      reason: 'selected model may require prompt-level JSON instruction instead of native schema enforcement',
    });
  }

  if (requirements.vision && !capabilities.supportsVision) {
    degradations.push({
      capability: 'vision',
      action: 'failed',
      reason: 'selected model does not advertise vision input support',
    });
  }

  const missingBuiltIns = requirements.builtInTools.filter(tool => !capabilities.supportsBuiltInTools.includes(tool));
  if (missingBuiltIns.length > 0) {
    degradations.push({
      capability: 'builtInTools',
      action: 'failed',
      reason: `selected model does not advertise built-in tools: ${missingBuiltIns.join(', ')}`,
    });
  }

  if (requirements.lora && provider !== 'cloudflare') {
    degradations.push({
      capability: 'lora',
      action: 'stripped',
      reason: 'LoRA adapters are provider-specific Cloudflare options and will not apply to this route',
    });
  }

  return degradations;
}

function routeWarnings(
  routing: RoutingInfo,
  requirements: GatewayRouteRequirements,
  capabilities: GatewayRouteCapabilityReport,
  cache: GatewayRouteCachePlan
): string[] {
  const warnings: string[] = [];

  if (routing.deprecationWarning) {
    warnings.push(routing.deprecationWarning);
  }

  if (cache.strategy !== 'off' && !cache.providerPromptCache && (cache.strategy === 'provider-prefix' || cache.strategy === 'both')) {
    warnings.push('provider prompt cache requested but selected model does not advertise prompt-cache support');
  }

  if (capabilities.maxContextLength !== undefined && routing.estimatedInputTokens > capabilities.maxContextLength) {
    warnings.push(`estimated input tokens exceed selected model context length (${routing.estimatedInputTokens} > ${capabilities.maxContextLength})`);
  }

  if (requirements.lora && routing.provider === 'cloudflare') {
    warnings.push('LoRA adapter identifiers are forwarded to Workers AI and are not validated by llm-providers');
  }

  return warnings;
}
