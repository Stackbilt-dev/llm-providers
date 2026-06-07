import { describe, expect, it } from 'vitest';
import { getGatewayRoutePlan } from '../gateway-routing';
import type { CanonicalLLMRequest } from '../canonical';

describe('getGatewayRoutePlan', () => {
  it('builds a Worker-friendly route plan from a canonical request', () => {
    const request: CanonicalLLMRequest = {
      messages: [{ role: 'user', content: 'Call the weather tool and return JSON.' }],
      stream: true,
      tools: [{
        type: 'function',
        function: {
          name: 'weather',
          description: 'Fetch weather.',
          parameters: { type: 'object', properties: { city: { type: 'string' } } },
        },
      }],
      output: { kind: 'json_object' },
      metadata: {
        cache: {
          strategy: 'both',
          key: 'route-cache-key',
          ttl: 300,
          sessionId: 'agent-session',
        },
      },
    };

    const plan = getGatewayRoutePlan(request, ['groq', 'anthropic']);

    expect(plan.selectedProvider).toBeTruthy();
    expect(plan.selectedModel).toBeTruthy();
    expect(plan.useCase).toBe('TOOL_CALLING');
    expect(plan.requirements).toMatchObject({
      streaming: true,
      toolCalling: true,
      structuredOutput: true,
      vision: false,
      builtInTools: [],
      lora: false,
    });
    expect(plan.cache).toMatchObject({
      strategy: 'both',
      responseCache: true,
      key: 'route-cache-key',
      ttl: 300,
      sessionId: 'agent-session',
    });
    expect(plan.legacyRequest.response_format).toEqual({ type: 'json_object' });
  });

  it('does not warn about cache support when the request has no cache hints', () => {
    const plan = getGatewayRoutePlan({
      messages: [{ role: 'user', content: 'Hello.' }],
    }, ['groq']);

    expect(plan.cache.strategy).toBeUndefined();
    expect(plan.cache.providerPromptCache).toBe(false);
    expect(plan.cache.responseCache).toBe(false);
    expect(plan.warnings).not.toContain('provider prompt cache requested but selected model does not advertise prompt-cache support');
  });

  it('surfaces LoRA degradation when a Cloudflare adapter request routes elsewhere', () => {
    const plan = getGatewayRoutePlan({
      messages: [{ role: 'user', content: 'Review this patch.' }],
      lora: 'codebeast-reviewer-v1',
    }, ['groq']);

    expect(plan.requirements.lora).toBe(true);
    expect(plan.selectedProvider).toBe('groq');
    expect(plan.degradations).toContainEqual({
      capability: 'lora',
      action: 'stripped',
      reason: 'LoRA adapters are provider-specific Cloudflare options and will not apply to this route',
    });
  });

  it('keeps Cloudflare LoRA route visible when the selected provider can receive it', () => {
    const plan = getGatewayRoutePlan({
      messages: [{ role: 'user', content: 'Review this patch.' }],
      lora: 'codebeast-reviewer-v1',
    }, ['cloudflare']);

    expect(plan.selectedProvider).toBe('cloudflare');
    expect(plan.requirements.lora).toBe(true);
    expect(plan.degradations.some(degradation => degradation.capability === 'lora')).toBe(false);
    expect(plan.warnings).toContain('LoRA adapter identifiers are forwarded to Workers AI and are not validated by llm-providers');
  });

  it('reports built-in tool capability mismatches for gateway debug endpoints', () => {
    const plan = getGatewayRoutePlan({
      messages: [{ role: 'user', content: 'Search the web.' }],
      builtInTools: [{ type: 'web_search' }],
      model: 'llama-3.3-70b-versatile',
    }, ['groq']);

    expect(plan.requirements.builtInTools).toEqual(['web_search']);
    expect(plan.degradations).toContainEqual({
      capability: 'builtInTools',
      action: 'failed',
      reason: 'selected model does not advertise built-in tools: web_search',
    });
  });
});
