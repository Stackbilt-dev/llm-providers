import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  LLMProviders, normalizeCanonicalRequest, normalizeLegacyRequest, normalizeLLMRequest,
  canonicalToLLMRequest, getRoutingInfo, getGatewayRoutePlan, NoRouteError, ResilienceContext,
} from '../index';
import { GroqProvider } from '../providers/groq';

afterEach(() => vi.restoreAllMocks());
const messages = [{ role: 'user' as const, content: 'hello' }];

describe('explicit normalization boundaries', () => {
  it('preserves canonical identity and cache hints on minimal metadata-only input', () => {
    const request = { messages, metadata: {
      tenantId: 'tenant', requestId: 'request', cache: { strategy: 'off' as const }, custom: { trace: 'id' },
    } };
    const normalized = normalizeCanonicalRequest(request);
    expect(normalized.metadata).toEqual(request.metadata);
    expect(canonicalToLLMRequest(normalized)).toMatchObject({
      tenantId: 'tenant', requestId: 'request', cache: { strategy: 'off' }, metadata: { trace: 'id' },
    });
  });

  it('keeps nested legacy metadata as custom data instead of inferring authority', () => {
    const request = { messages, tenantId: 'authoritative', metadata: { tenantId: 'custom-value', requestId: 'custom-id' } };
    expect(normalizeLegacyRequest(request).metadata).toMatchObject({
      tenantId: 'authoritative', custom: request.metadata,
    });
    expect(normalizeLLMRequest(request)).toEqual(normalizeLegacyRequest(request));
  });

  it('preserves multiple system messages in order and defines explicit prompt precedence', () => {
    const request = { messages: [
      { role: 'system' as const, content: 'first' }, { role: 'system' as const, content: 'second' }, ...messages,
    ] };
    expect(normalizeLegacyRequest(request)).toMatchObject({ system: 'first\n\nsecond', messages });
    expect(normalizeLegacyRequest({ ...request, systemPrompt: 'override' }).system).toBe('override');
  });

  it('explicit boundaries keep mixed shapes deterministic', () => {
    const mixed = { messages, systemPrompt: 'legacy', system: 'canonical',
      tenantId: 'legacy-tenant', metadata: { tenantId: 'canonical-tenant' } };
    expect(normalizeLegacyRequest(mixed)).toMatchObject({ system: 'legacy', metadata: { tenantId: 'legacy-tenant' } });
    expect(normalizeCanonicalRequest(mixed)).toMatchObject({ system: 'canonical', metadata: { tenantId: 'canonical-tenant' } });
    expect(normalizeLLMRequest(mixed)).toEqual(normalizeCanonicalRequest(mixed));
  });
});

describe('configured-provider routing snapshots', () => {
  it('reports Groq for a Groq-only shared model, including the Groq catalog entry', () => {
    const info = getRoutingInfo({ messages, model: 'openai/gpt-oss-120b' }, ['groq']);
    expect(info.provider).toBe('groq');
    expect(info.catalogEntry?.provider).toBe('groq');
  });

  it('selects the host capable of every requested built-in tool', () => {
    const info = getRoutingInfo({ messages, model: 'openai/gpt-oss-120b', builtInTools: [{ type: 'web_search' }] }, ['cerebras', 'groq']);
    expect(info.provider).toBe('groq');
  });

  it('keeps the public catalog helpers non-throwing for unconfigured hosts (1.x contract)', () => {
    expect(getRoutingInfo({ model: 'gpt-4o-mini' }, ['groq'])).toMatchObject({ provider: 'openai', model: 'gpt-4o-mini' });
    expect(getRoutingInfo({ model: 'gpt-4o-mini' }, ['cloudflare']).provider).toBe('openai');
    expect(() => getRoutingInfo({ messages }, [])).not.toThrow();
    expect(() => getRoutingInfo({ messages, model: 'openai/gpt-oss-120b', builtInTools: [{ type: 'browser_automation' }] }, ['groq']))
      .not.toThrow();
    const plan = getGatewayRoutePlan({ messages, model: 'gpt-4o-mini' }, ['cloudflare']);
    expect(plan.selectedProvider).toBe('openai');
    expect(plan.executable).toBe(false);
    expect(getGatewayRoutePlan({ messages, model: 'gpt-4o-mini' }, ['openai']).executable).toBe(true);
  });

  it('the client snapshot only reports configured providers, unlike the catalog helper', () => {
    const llm = new LLMProviders({ groq: { apiKey: 'test' }, stateScope: 'client' });
    expect(llm.getRoutingInfo({ messages, model: 'gpt-4o-mini' }).provider).toBe('groq');
  });

  it('the client snapshot uses the same plan as actual dispatch', async () => {
    const llm = new LLMProviders({ groq: { apiKey: 'test' }, stateScope: 'client' });
    const request = { messages, model: 'openai/gpt-oss-120b' };
    const snapshot = llm.getRoutingInfo(request);
    const generate = vi.spyOn(GroqProvider.prototype, 'generateResponse').mockResolvedValue({
      message: 'ok', provider: 'groq', model: snapshot.model, responseTime: 1,
      usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2, cost: 0 },
    });
    expect(await llm.generateResponse(request)).toMatchObject({ provider: snapshot.provider, model: snapshot.model });
    expect(generate).toHaveBeenCalledWith(expect.objectContaining({ model: snapshot.model }));
  });

  it('client introspection cannot offer an exhausted account', () => {
    const context = new ResilienceContext();
    const llm = new LLMProviders({ groq: { apiKey: 'test' }, resilienceContext: context });
    context.exhaustion.markExhausted('groq');
    expect(() => llm.getRoutingInfo({ messages })).toThrow(NoRouteError);
  });
});
