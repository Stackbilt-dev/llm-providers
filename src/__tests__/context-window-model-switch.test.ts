/**
 * #136: opt-in switching to a larger model of the same provider
 * (contextWindowModelSwitch: true). When a provider's planned model can't hold the
 * estimated input and the provider has a compatible model that can, the plan uses
 * that model instead. Off by default; pinned models are never switched.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getCatalogEntry, getProviderDefaultModel, MODEL_CATALOG, rankModels } from '../model-catalog';
import { LLMProviders } from '../index';
import type { ProviderFactoryConfig } from '../factory';
import { cloudflareReasoningEffortCapability } from '../providers/cloudflare-reasoning';
import { largerOwnModel } from '../utils/context-window';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { defaultExhaustionRegistry } from '../utils/exhaustion';
import type { LLMRequest, Tool } from '../types';

const NVIDIA_DEFAULT = 'deepseek-ai/deepseek-v4-flash';
const NVIDIA_LARGE = 'meta/llama-4-maverick-17b-128e-instruct';

/** A short request that the caller says is `tokens` input tokens. */
function counted(tokens: number, extra: Partial<LLMRequest> = {}): LLMRequest {
  return { messages: [{ role: 'user', content: 'Summarise the attached document.' }], maxTokens: 10, inputTokenCount: tokens, ...extra };
}

const lookupTool: Tool = {
  type: 'function',
  function: { name: 'lookup', description: 'Look something up', parameters: { type: 'object', properties: {} } },
};

function chatResult(content: string) {
  return {
    id: 'chatcmpl-test',
    choices: [{ index: 0, message: { role: 'assistant', content }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
  };
}

/** An OpenAI-compatible JSON response echoing the requested model. */
function compatibleResponse(init?: RequestInit, message: Record<string, unknown> = { role: 'assistant', content: 'from http' }, finish = 'stop') {
  const model = JSON.parse(String(init?.body ?? '{}')).model ?? 'unknown';
  return new Response(JSON.stringify({
    id: 'http-1', object: 'chat.completion', created: 1, model,
    choices: [{ index: 0, message, finish_reason: finish }],
    usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
  }), { status: 200, headers: { 'Content-Type': 'application/json' } });
}

function sentModels(fetchMock: ReturnType<typeof vi.fn>): string[] {
  return fetchMock.mock.calls.map(call => JSON.parse(String(call[1].body)).model);
}

function setup(config: Partial<ProviderFactoryConfig> = {}) {
  const run = vi.fn().mockResolvedValue(chatResult('from cloudflare'));
  const fetchMock = vi.fn().mockImplementation(async (_url: string, init?: RequestInit) => compatibleResponse(init));
  vi.stubGlobal('fetch', fetchMock);
  const llm = new LLMProviders({
    cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 },
    nvidia: { apiKey: 'test', maxRetries: 0 },
    preferredProvider: 'nvidia',
    stateScope: 'client',
    ...config,
  });
  return { llm, run, fetchMock };
}

type Plan = { chain: string[]; providerModels: Map<string, string>; contextSkips: Set<string> };
function planOf(llm: LLMProviders, request: LLMRequest): Plan {
  return (llm as unknown as { factory: { buildProviderPlan(request: LLMRequest): Plan } }).factory.buildProviderPlan(request);
}

function windowOf(provider: string, model: string): number {
  return MODEL_CATALOG.find(entry => entry.provider === provider && entry.model === model)!.capabilities.maxContextLength;
}

beforeEach(() => {
  defaultCircuitBreakerManager.resetAll();
  defaultExhaustionRegistry.reset();
  vi.spyOn(console, 'log').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('test assumptions', () => {
  it("NVIDIA's default model for a plain request is smaller than its largest compatible model", () => {
    expect(getProviderDefaultModel('nvidia', counted(300_000))).toBe(NVIDIA_DEFAULT);
    expect(windowOf('nvidia', NVIDIA_DEFAULT)).toBeLessThan(300_000);
    expect(windowOf('nvidia', NVIDIA_LARGE)).toBeGreaterThanOrEqual(300_000);
    expect(getCatalogEntry(NVIDIA_LARGE)?.useCases).toContain('LONG_CONTEXT');
  });
});

describe('largerOwnModel', () => {
  const ranked = (provider: string, request: LLMRequest = counted(300_000), useCase: Parameters<typeof rankModels>[0] = 'BALANCED') =>
    rankModels(useCase, [provider], { request });

  it("returns the provider's highest-ranked compatible model that fits", () => {
    expect(largerOwnModel(ranked('nvidia'), 'nvidia', 'BALANCED', 300_000, {})).toBe(NVIDIA_LARGE);
  });

  it('returns undefined when no model of the provider fits', () => {
    expect(largerOwnModel(ranked('nvidia'), 'nvidia', 'BALANCED', 2_000_000, {})).toBeUndefined();
    expect(largerOwnModel(ranked('groq'), 'groq', 'BALANCED', 140_000, {})).toBeUndefined();
  });

  it('only considers the named provider', () => {
    expect(largerOwnModel(ranked('cloudflare'), 'nvidia', 'BALANCED', 300_000, {})).toBeUndefined();
  });

  it('skips models suited to neither the use case nor long context', () => {
    // groq/compound has a larger window than Groq's default but is research-only.
    expect(windowOf('groq', 'groq/compound')).toBeGreaterThan(windowOf('groq', 'llama-3.3-70b-versatile'));
    expect(largerOwnModel(ranked('groq'), 'groq', 'BALANCED', 129_000, {})).toBeUndefined();
    expect(largerOwnModel(ranked('groq', counted(129_000), 'RESEARCH'), 'groq', 'RESEARCH', 129_000, {})).toMatch(/^groq\/compound/);
  });

  it('accepts a model tagged LONG_CONTEXT although not tagged for the use case', () => {
    const model = largerOwnModel(ranked('cerebras', counted(130_000)), 'cerebras', 'BALANCED', 130_000, {});
    expect(model).toBeDefined();
    expect(windowOf('cerebras', model!)).toBeGreaterThanOrEqual(130_000);
    expect(getCatalogEntry(model!)?.useCases).toContain('LONG_CONTEXT');
  });

  it('for strict reasoning, requires a Cloudflare model that accepts the effort exactly', () => {
    const strict = { reasoning: { effort: 'high' as const, strict: true } };
    const cloudflare = ranked('cloudflare');
    const model = largerOwnModel(cloudflare, 'cloudflare', 'BALANCED', 300_000, strict);
    expect(model).toBeDefined();
    expect(cloudflareReasoningEffortCapability(model!)?.exact).toContain('high');

    const inexact = cloudflare.filter(entry => !(cloudflareReasoningEffortCapability(entry.model)?.exact.includes('high') ?? false));
    expect(largerOwnModel(inexact, 'cloudflare', 'BALANCED', 300_000, {})).toBeDefined();
    expect(largerOwnModel(inexact, 'cloudflare', 'BALANCED', 300_000, strict)).toBeUndefined();
    expect(largerOwnModel(ranked('nvidia'), 'nvidia', 'BALANCED', 300_000, strict)).toBeUndefined();
    // Without strict, the effort doesn't restrict the choice.
    expect(largerOwnModel(ranked('nvidia'), 'nvidia', 'BALANCED', 300_000, { reasoning: { effort: 'high' } })).toBe(NVIDIA_LARGE);
  });
});

describe('factory with contextWindowModelSwitch', () => {
  it.each([undefined, false])('is off by default (%s): the planned model is sent', async contextWindowModelSwitch => {
    const { llm, fetchMock } = setup({ contextWindowModelSwitch });
    const response = await llm.generateResponse(counted(300_000));
    expect(response.provider).toBe('nvidia');
    expect(sentModels(fetchMock)).toEqual([NVIDIA_DEFAULT]);
  });

  it('sends the larger model of the same provider', async () => {
    const { llm, run, fetchMock } = setup({ contextWindowModelSwitch: true });
    const response = await llm.generateResponse(counted(300_000));
    expect(response.provider).toBe('nvidia');
    expect(response.model).toBe(NVIDIA_LARGE);
    expect(sentModels(fetchMock)).toEqual([NVIDIA_LARGE]);
    expect(run).not.toHaveBeenCalled();
  });

  it('with contextWindowRouting, a provider that switched is not skipped', async () => {
    const { llm, run, fetchMock } = setup({ contextWindowModelSwitch: true, contextWindowRouting: true });
    const response = await llm.generateResponse(counted(300_000));
    expect(response.provider).toBe('nvidia');
    expect(sentModels(fetchMock)).toEqual([NVIDIA_LARGE]);
    expect(response.fallback?.failedAttempts ?? []).toEqual([]);
    expect(run).not.toHaveBeenCalled();
  });

  it('control: contextWindowRouting alone skips the provider instead', async () => {
    const { llm, run, fetchMock } = setup({ contextWindowRouting: true });
    const response = await llm.generateResponse(counted(300_000));
    expect(response.provider).toBe('cloudflare');
    expect(run).toHaveBeenCalledTimes(1);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(response.fallback?.failedAttempts).toEqual([{ provider: 'nvidia', model: NVIDIA_DEFAULT, errorCode: 'CONTEXT_WINDOW' }]);
  });

  it('keeps a planned model that already fits', async () => {
    const { llm, fetchMock } = setup({ contextWindowModelSwitch: true });
    await llm.generateResponse(counted(1_000));
    expect(sentModels(fetchMock)).toEqual([NVIDIA_DEFAULT]);
  });

  it('never switches a pinned model', async () => {
    const { llm, fetchMock } = setup({ contextWindowModelSwitch: true, contextWindowRouting: true });
    await llm.generateResponse(counted(300_000, { model: NVIDIA_DEFAULT }));
    expect(sentModels(fetchMock)).toEqual([NVIDIA_DEFAULT]);
  });

  it('switches other providers in the chain when a model is pinned', () => {
    const { llm } = setup({ contextWindowModelSwitch: true, preferredProvider: 'cloudflare' });
    const plan = planOf(llm, counted(300_000, { model: '@cf/meta/llama-3.3-70b-instruct-fp8-fast' }));
    expect(plan.providerModels.get('cloudflare')).toBe('@cf/meta/llama-3.3-70b-instruct-fp8-fast');
    expect(plan.providerModels.get('nvidia')).toBe(NVIDIA_LARGE);
  });

  it("keeps the planned model when none of the provider's models fits", async () => {
    const { llm, fetchMock } = setup({ contextWindowModelSwitch: true });
    await llm.generateResponse(counted(1_100_000));
    expect(sentModels(fetchMock)).toEqual([NVIDIA_DEFAULT]);
  });

  it('does not move a plain request onto a research-only model', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('from cloudflare'));
    const fetchMock = vi.fn().mockImplementation(async (_url: string, init?: RequestInit) => compatibleResponse(init));
    vi.stubGlobal('fetch', fetchMock);
    const llm = new LLMProviders({
      cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 },
      groq: { apiKey: 'test', maxRetries: 0 },
      preferredProvider: 'groq',
      stateScope: 'client',
      contextWindowModelSwitch: true,
      contextWindowRouting: true,
    });
    const response = await llm.generateResponse(counted(129_000));
    expect(response.provider).toBe('cloudflare');
    expect(fetchMock).not.toHaveBeenCalled();
    expect(response.fallback?.failedAttempts).toEqual([{ provider: 'groq', model: 'llama-3.3-70b-versatile', errorCode: 'CONTEXT_WINDOW' }]);
  });

  it('is reported by client getRoutingInfo', () => {
    const { llm } = setup({ contextWindowModelSwitch: true });
    const routing = llm.getRoutingInfo(counted(300_000));
    expect(routing.provider).toBe('nvidia');
    expect(routing.model).toBe(NVIDIA_LARGE);
    expect(setup().llm.getRoutingInfo(counted(300_000)).model).toBe(NVIDIA_DEFAULT);
  });

  it('applies on the managed streaming path', async () => {
    const { llm } = setup({ contextWindowModelSwitch: true });
    const fetchMock = vi.fn().mockImplementation(async () => new Response(
      'data: {"id":"s1","choices":[{"index":0,"delta":{"content":"streamed"},"finish_reason":null}]}\n\n'
      + 'data: {"id":"s1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\n'
      + 'data: [DONE]\n\n',
      { status: 200, headers: { 'Content-Type': 'text/event-stream' } },
    ));
    vi.stubGlobal('fetch', fetchMock);
    const managed = await llm.generateManagedResponseStream(counted(300_000, { stream: true }));
    const reader = managed.stream.getReader();
    while (!(await reader.read()).done) { /* drain */ }
    const completion = await managed.completion;
    expect(completion.outcome).toBe('success');
    expect(sentModels(fetchMock)).toEqual([NVIDIA_LARGE]);
  });

  it('switches a later tool-loop turn once the tool result outgrows the planned model', async () => {
    const toolCall = { role: 'assistant', content: null,
      tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'lookup', arguments: '{}' } }] };
    const fetchMock = vi.fn()
      .mockImplementationOnce(async (_url: string, init?: RequestInit) => compatibleResponse(init, toolCall, 'tool_calls'))
      .mockImplementation(async (_url: string, init?: RequestInit) => compatibleResponse(init));
    vi.stubGlobal('fetch', fetchMock);
    const llm = new LLMProviders({
      cerebras: { apiKey: 'test', maxRetries: 0 },
      stateScope: 'client',
      contextWindowModelSwitch: true,
    });
    const first = getProviderDefaultModel('cerebras', { messages: [{ role: 'user', content: 'x' }], tools: [lookupTool] });
    expect(windowOf('cerebras', first)).toBeLessThan(130_000);
    const response = await llm.generateResponseWithTools(
      { messages: [{ role: 'user', content: 'Summarise the document.' }], tools: [lookupTool], maxTokens: 10 },
      { execute: async () => 'x'.repeat(520_000) },
    );
    expect(response.message).toBe('from http');
    const [firstModel, secondModel] = sentModels(fetchMock);
    expect(firstModel).toBe(first);
    expect(secondModel).not.toBe(first);
    expect(windowOf('cerebras', secondModel)).toBeGreaterThanOrEqual(130_000);
    expect(getCatalogEntry(secondModel)?.capabilities.supportsTools).toBe(true);
  });

  it('includes the option in the response cache key, leaving default keys unchanged', async () => {
    const keys: Record<string, string> = {};
    for (const [label, contextWindowModelSwitch] of [['unset', undefined], ['false', false], ['true', true]] as const) {
      const cache = { get: vi.fn().mockResolvedValue(null), put: vi.fn().mockResolvedValue(undefined) };
      const { llm } = setup({ contextWindowModelSwitch, responseCache: cache });
      await llm.generateResponse(counted(10));
      keys[label] = cache.get.mock.calls[0][0];
    }
    expect(keys.unset).toBe(keys.false);
    expect(keys.true).not.toBe(keys.unset);
  });

  it('turning it on with updateConfig takes effect', async () => {
    const { llm, fetchMock } = setup();
    llm.updateConfig({ contextWindowModelSwitch: true });
    await llm.generateResponse(counted(300_000));
    expect(sentModels(fetchMock)).toEqual([NVIDIA_LARGE]);
  });
});
