import type { LLMRequest, LLMResponse } from '../types.js';

// Reject values JSON would silently omit/coerce, rather than aliasing requests.
function stableJson(value: unknown, ancestors = new Set<object>()): string {
  if (value === null) return 'null';
  if (typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
  if (typeof value === 'number' && Number.isFinite(value)) return JSON.stringify(value);
  if (typeof value !== 'object' || ancestors.has(value)) throw new Error('Non-JSON cache identity');
  const prototype = Object.getPrototypeOf(value);
  if (!Array.isArray(value) && prototype !== Object.prototype && prototype !== null) {
    throw new Error('Non-JSON cache identity');
  }
  if (Object.getOwnPropertySymbols(value).length) throw new Error('Non-JSON cache identity');
  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      return '[' + Array.from(value, item => stableJson(item, ancestors)).join(',') + ']';
    }
    const record = value as Record<string, unknown>;
    return '{' + Object.keys(record).filter(key => record[key] !== undefined).sort()
      .map(key => JSON.stringify(key) + ':' + stableJson(record[key], ancestors)).join(',') + '}';
  } finally {
    ancestors.delete(value);
  }
}

export async function deriveResponseCacheKey(
  request: LLMRequest,
  namespace: string,
  routing: unknown
): Promise<string | undefined> {
  if (request.cache?.strategy === 'off' || request.cache?.strategy === 'provider-prefix') return undefined;
  try {
    const { requestId, gatewayMetadata, cache, ...semanticRequest } = request;
    // Custom gateway metadata may influence routing; exclude only tracing/cache controls.
    const identity = stableJson({
      version: 2, namespace, routing, request: semanticRequest,
      callerKey: cache?.key ?? null,
      gatewayCustomMetadata: gatewayMetadata?.customMetadata ?? null,
    });
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(identity));
    return 'llmp:v2:' + Array.from(new Uint8Array(digest))
      .map(byte => byte.toString(16).padStart(2, '0')).join('');
  } catch {
    return undefined;
  }
}

export function isCachedResponse(value: unknown): value is LLMResponse {
  if (!value || typeof value !== 'object') return false;
  const response = value as Partial<LLMResponse>;
  const usage = response.usage;
  return typeof response.message === 'string' && typeof response.model === 'string'
    && [response.id, response.reasoning, response.content]
      .every(field => field === undefined || typeof field === 'string')
    && [response.metadata, response.cache]
      .every(field => field === undefined || (field !== null && typeof field === 'object' && !Array.isArray(field)))
    && (response.finishReason === undefined || ['stop', 'length', 'tool_calls', 'content_filter'].includes(response.finishReason))
    && typeof response.provider === 'string' && typeof response.responseTime === 'number'
    && Number.isFinite(response.responseTime) && response.responseTime >= 0
    && !!usage && typeof usage === 'object'
    && [usage.inputTokens, usage.outputTokens, usage.totalTokens, usage.cost]
      .every(number => typeof number === 'number' && Number.isFinite(number) && number >= 0)
    && (response.toolCalls === undefined || (Array.isArray(response.toolCalls)
      && response.toolCalls.every(call => call && typeof call.id === 'string'
        && call.type === 'function' && call.function && typeof call.function.name === 'string'
        && typeof call.function.arguments === 'string')));
}
