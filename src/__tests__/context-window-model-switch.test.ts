/**
 * #136: opt-in switching to a larger model of the same provider
 * (contextWindowModelSwitch: true). When a provider's planned model can't hold the
 * estimated input and the provider has an active, compatible model that can hold the
 * input plus maxTokens, the plan uses that model and the response reports
 * `contextWindowSwitch`. Off by default; pinned models are never switched.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getCatalogEntry, getProviderDefaultModel, MODEL_CATALOG, rankModels } from '../model-catalog';
import { LLMProviders } from '../index';
import type { ContextWindowSwitch } from '../index';
import type { ProviderFactoryConfig } from '../factory';
import { cloudflareReasoningEffortCapability } from '../providers/cloudflare-reasoning';
import { largerOwnModel, requestedOutputTokens } from '../utils/context-window';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { defaultExhaustionRegistry } from '../utils/exhaustion';
import type { LLMRequest, Tool } from '../types';

const NVIDIA_DEFAULT = 'deepseek-ai/deepseek-v4-flash';
const NVIDIA_LARGE = 'meta/llama-4-maverick-17b-128e-instruct';
const NVIDIA_128K = 'nvidia/llama-3.1-nemotron-70b-instruct';

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

function setup(config: Partial<ProviderFactoryConfig> = {}, run = vi.fn().mockResolvedValue(chatResult('from cloudflare'))) {
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

type Plan = { chain: string[]; providerModels: Map<string, string>; contextSkips: Set<string>; modelSwitches: Map<string, ContextWindowSwitch> };
function planOf(llm: LLMProviders, request: LLMRequest): Plan {
  return (llm as unknown as { factory: { buildProviderPlan(request: LLMRequest): Plan } }).factory.buildProviderPlan(request);
}

function windowOf(provider: string, model: string): number {
  return MODEL_CATALOG.find(entry => entry.provider === provider && entry.model === model)!.capabilities.maxContextLength;
}

async function drain(stream: ReadableStream<string>) {
  const reader = stream.getReader();
  while (!(await reader.read()).done) { /* drain */ }
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
  it("NVIDIA's default model for a plain request is smaller than its larger compatible models", () => {
    expect(getProviderDefaultModel('nvidia', counted(300_000))).toBe(NVIDIA_DEFAULT);
    expect(windowOf('nvidia', NVIDIA_DEFAULT)).toBe(65_536);
    expect(windowOf('nvidia', NVIDIA_128K)).toBe(128_000);
    expect(windowOf('nvidia', NVIDIA_LARGE)).toBe(1_048_576);
    expect(getProviderDefaultModel('cloudflare', counted(300_000))).toBe('@cf/zai-org/glm-5.3-flash');
  });
});

describe('largerOwnModel', () => {
  const ranked = (provider: string, useCase: Parameters<typeof rankModels>[0] = 'BALANCED', request: LLMRequest = counted(300_000)) =>
    rankModels(useCase, [provider], { request });

  it("returns the provider's highest-ranked compatible entry that fits", () => {
    expect(largerOwnModel(ranked('nvidia'), 'nvidia', 'BALANCED', 300_000, {})?.model).toBe(NVIDIA_LARGE);
    expect(largerOwnModel(ranked('nvidia'), 'nvidia', 'BALANCED', 100_000, {})?.model).toBe(NVIDIA_128K);
  });

  it('accepts exactly a full window and nothing past it', () => {
    expect(largerOwnModel(ranked('nvidia'), 'nvidia', 'BALANCED', 1_048_576, {})?.model).toBe(NVIDIA_LARGE);
    expect(largerOwnModel(ranked('nvidia'), 'nvidia', 'BALANCED', 1_048_577, {})).toBeUndefined();
  });

  it('returns undefined when no model of the provider fits, and only considers the named provider', () => {
    expect(largerOwnModel(ranked('groq'), 'groq', 'BALANCED', 140_000, {})).toBeUndefined();
    expect(largerOwnModel(ranked('cloudflare'), 'nvidia', 'BALANCED', 300_000, {})).toBeUndefined();
  });

  it('only picks active models, never a deprecated compatibility one', () => {
    const cerebras = ranked('cerebras', 'BALANCED', counted(130_000));
    const deprecated = cerebras.find(entry => entry.model === 'qwen-3-235b-a22b-instruct-2507');
    expect(deprecated?.lifecycle).toBe('compatibility');
    expect(cerebras.indexOf(deprecated!)).toBeLessThan(cerebras.findIndex(entry => entry.model === 'zai-glm-4.7'));
    const model = largerOwnModel(cerebras, 'cerebras', 'BALANCED', 130_000, {});
    expect(model?.model).toBe('zai-glm-4.7');
    expect(model?.lifecycle).toBe('active');
  });

  it('skips models suited to neither the use case nor long context', () => {
    // groq/compound has a larger window than Groq's default but is research-only.
    expect(windowOf('groq', 'groq/compound')).toBeGreaterThan(windowOf('groq', 'llama-3.3-70b-versatile'));
    expect(largerOwnModel(ranked('groq'), 'groq', 'BALANCED', 129_000, {})).toBeUndefined();
    expect(largerOwnModel(ranked('groq', 'RESEARCH'), 'groq', 'RESEARCH', 129_000, {})?.model).toMatch(/^groq\/compound/);
  });

  it('accepts a model tagged LONG_CONTEXT although not tagged for the use case', () => {
    expect(getCatalogEntry(NVIDIA_LARGE)?.useCases).not.toContain('BALANCED');
    expect(getCatalogEntry(NVIDIA_LARGE)?.useCases).toContain('LONG_CONTEXT');
    expect(largerOwnModel(ranked('nvidia'), 'nvidia', 'BALANCED', 300_000, {})?.model).toBe(NVIDIA_LARGE);
  });

  it('for strict reasoning, requires a Cloudflare model that accepts the effort exactly', () => {
    const strict = { reasoning: { effort: 'high' as const, strict: true } };
    const cloudflare = ranked('cloudflare');
    const model = largerOwnModel(cloudflare, 'cloudflare', 'BALANCED', 300_000, strict);
    expect(model).toBeDefined();
    expect(cloudflareReasoningEffortCapability(model!.model)?.exact).toContain('high');

    const inexact = cloudflare.filter(entry => !(cloudflareReasoningEffortCapability(entry.model)?.exact.includes('high') ?? false));
    expect(largerOwnModel(inexact, 'cloudflare', 'BALANCED', 300_000, {})).toBeDefined();
    expect(largerOwnModel(inexact, 'cloudflare', 'BALANCED', 300_000, strict)).toBeUndefined();
    expect(largerOwnModel(ranked('nvidia'), 'nvidia', 'BALANCED', 300_000, strict)).toBeUndefined();
    // Without strict, the effort doesn't restrict the choice.
    expect(largerOwnModel(ranked('nvidia'), 'nvidia', 'BALANCED', 300_000, { reasoning: { effort: 'high' } })?.model).toBe(NVIDIA_LARGE);
  });

  it('requestedOutputTokens counts only a positive safe-integer maxTokens', () => {
    expect(requestedOutputTokens(2_000)).toBe(2_000);
    for (const value of [undefined, 0, -5, 1.5, Number.NaN, Number.POSITIVE_INFINITY, '100']) {
      expect(requestedOutputTokens(value)).toBe(0);
    }
  });
});

describe('factory with contextWindowModelSwitch', () => {
  it.each([undefined, false])('is off by default (%s): the planned model is sent', async contextWindowModelSwitch => {
    const { llm, fetchMock } = setup({ contextWindowModelSwitch });
    const response = await llm.generateResponse(counted(300_000));
    expect(response.provider).toBe('nvidia');
    expect(sentModels(fetchMock)).toEqual([NVIDIA_DEFAULT]);
    expect(response.contextWindowSwitch).toBeUndefined();
  });

  it('sends the larger model of the same provider and reports the switch', async () => {
    const { llm, run, fetchMock } = setup({ contextWindowModelSwitch: true });
    const response = await llm.generateResponse(counted(300_000));
    expect(response.provider).toBe('nvidia');
    expect(response.model).toBe(NVIDIA_LARGE);
    expect(response.contextWindowSwitch).toEqual({ from: NVIDIA_DEFAULT, to: NVIDIA_LARGE });
    expect(response.fallback).toBeUndefined();
    expect(sentModels(fetchMock)).toEqual([NVIDIA_LARGE]);
    expect(run).not.toHaveBeenCalled();
  });

  it('picks the smallest step that ranks first, not always the largest window', async () => {
    const { llm, fetchMock } = setup({ contextWindowModelSwitch: true });
    await llm.generateResponse(counted(100_000));
    expect(sentModels(fetchMock)).toEqual([NVIDIA_128K]);
  });

  it('requires the target to hold the input plus maxTokens', async () => {
    const { llm, fetchMock } = setup({ contextWindowModelSwitch: true });
    const tight = await llm.generateResponse(counted(130_000, { maxTokens: 2_000 }));
    expect(tight.contextWindowSwitch?.to).toBe(NVIDIA_LARGE);
    // Fits the 1M window as input alone, but not with maxTokens (under 4000, so the use case stays BALANCED).
    const none = await llm.generateResponse(counted(1_046_000, { maxTokens: 3_000 }));
    expect(none.contextWindowSwitch).toBeUndefined();
    expect(sentModels(fetchMock)).toEqual([NVIDIA_LARGE, NVIDIA_DEFAULT]);
  });

  it('with contextWindowRouting, a provider that switched is not skipped', async () => {
    const { llm, run, fetchMock } = setup({ contextWindowModelSwitch: true, contextWindowRouting: true });
    const response = await llm.generateResponse(counted(300_000));
    expect(response.provider).toBe('nvidia');
    expect(sentModels(fetchMock)).toEqual([NVIDIA_LARGE]);
    expect(response.fallback).toBeUndefined();
    expect(run).not.toHaveBeenCalled();
  });

  it('with contextWindowRouting, does not switch when a reliable provider fits at least twice as much', async () => {
    const { llm, run, fetchMock } = setup({ contextWindowModelSwitch: true, contextWindowRouting: true });
    // 100k: NVIDIA's target would be 131k; Workers AI's planned model holds 1M.
    const response = await llm.generateResponse(counted(100_000));
    expect(response.provider).toBe('cloudflare');
    expect(response.contextWindowSwitch).toBeUndefined();
    expect(response.fallback?.failedAttempts).toEqual([{ provider: 'nvidia', model: NVIDIA_DEFAULT, errorCode: 'CONTEXT_WINDOW' }]);
    expect(run).toHaveBeenCalledTimes(1);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('with contextWindowRouting, still switches when the larger provider is unavailable', () => {
    const resilience = { circuitBreakers: { getAllStates: () => ({ cloudflare: { state: 'OPEN' } }) } };
    const { llm } = setup({ contextWindowModelSwitch: true, contextWindowRouting: true });
    const factory = (llm as unknown as { factory: { resilienceContext: { circuitBreakers: { getAllStates(): unknown } } } }).factory;
    vi.spyOn(factory.resilienceContext.circuitBreakers, 'getAllStates').mockReturnValue(resilience.circuitBreakers.getAllStates());
    const plan = planOf(llm, counted(100_000));
    expect(plan.providerModels.get('nvidia')).toBe(NVIDIA_128K);
    expect(plan.contextSkips.has('nvidia')).toBe(false);
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
    const response = await llm.generateResponse(counted(1_000));
    expect(sentModels(fetchMock)).toEqual([NVIDIA_DEFAULT]);
    expect(response.contextWindowSwitch).toBeUndefined();
  });

  it('never switches a pinned model', async () => {
    const { llm, fetchMock } = setup({ contextWindowModelSwitch: true, contextWindowRouting: true });
    const response = await llm.generateResponse(counted(300_000, { model: NVIDIA_DEFAULT }));
    expect(sentModels(fetchMock)).toEqual([NVIDIA_DEFAULT]);
    expect(response.contextWindowSwitch).toBeUndefined();
  });

  it('switches other providers in the chain when a model is pinned', () => {
    const { llm } = setup({ contextWindowModelSwitch: true, preferredProvider: 'cloudflare' });
    const plan = planOf(llm, counted(300_000, { model: '@cf/meta/llama-3.3-70b-instruct-fp8-fast' }));
    expect(plan.providerModels.get('cloudflare')).toBe('@cf/meta/llama-3.3-70b-instruct-fp8-fast');
    expect(plan.providerModels.get('nvidia')).toBe(NVIDIA_LARGE);
    expect(plan.modelSwitches.has('cloudflare')).toBe(false);
  });

  it("keeps the planned model when none of the provider's models fits", async () => {
    const { llm, fetchMock } = setup({ contextWindowModelSwitch: true });
    await llm.generateResponse(counted(1_100_000));
    expect(sentModels(fetchMock)).toEqual([NVIDIA_DEFAULT]);
  });

  it('with allowFallback: false and no pin, the first provider is served on its switched model', async () => {
    const { llm, run, fetchMock } = setup({ contextWindowModelSwitch: true, contextWindowRouting: true });
    const response = await llm.generateResponse(counted(300_000, { allowFallback: false }));
    expect(response.provider).toBe('nvidia');
    expect(sentModels(fetchMock)).toEqual([NVIDIA_LARGE]);
    expect(run).not.toHaveBeenCalled();
  });

  it("doesn't report a switch when a caller's fallback rule sent another model", async () => {
    const run = vi.fn().mockRejectedValue(new Error('binding exploded'));
    const { llm, fetchMock } = setup({
      contextWindowModelSwitch: true,
      preferredProvider: 'cloudflare',
      fallbackRules: [{ condition: 'error', fallbackProvider: 'nvidia', fallbackModel: 'meta/llama-3.1-70b-instruct' }],
    }, run);
    expect(planOf(llm, counted(300_000)).modelSwitches.get('nvidia')).toEqual({ from: NVIDIA_DEFAULT, to: NVIDIA_LARGE });
    // Workers AI's planned model holds 1M, so it isn't switched; make the request
    // large enough to need NVIDIA's switch, then let the rule override it.
    const response = await llm.generateResponse(counted(300_000));
    expect(response.provider).toBe('nvidia');
    expect(sentModels(fetchMock)).toEqual(['meta/llama-3.1-70b-instruct']);
    expect(response.contextWindowSwitch).toBeUndefined();
  });

  it('chooses with the same use case the planned model came from', async () => {
    // metadata.useCase only affects ranked plans; the preferred provider's planned
    // model comes from the inferred use case, so the switch must not jump to a
    // research-only model.
    const fetchMock = vi.fn().mockImplementation(async (_url: string, init?: RequestInit) => compatibleResponse(init));
    vi.stubGlobal('fetch', fetchMock);
    const llm = new LLMProviders({
      groq: { apiKey: 'test', maxRetries: 0 },
      preferredProvider: 'groq',
      stateScope: 'client',
      contextWindowModelSwitch: true,
    });
    await llm.generateResponse(counted(129_000, { metadata: { useCase: 'RESEARCH' } }));
    expect(sentModels(fetchMock)).toEqual(['llama-3.3-70b-versatile']);
  });

  it('switches a Workers AI research plan to a larger model, honouring strict reasoning', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('from cloudflare'));
    const llm = new LLMProviders({ cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 }, stateScope: 'client', contextWindowModelSwitch: true });
    const research = { metadata: { useCase: 'RESEARCH' } };
    const planned = planOf(llm, counted(1_000, research)).providerModels.get('cloudflare')!;
    expect(windowOf('cloudflare', planned)).toBeLessThan(300_000);

    const response = await llm.generateResponse(counted(300_000, research));
    expect(response.contextWindowSwitch?.from).toBe(planned);
    expect(windowOf('cloudflare', response.model)).toBeGreaterThanOrEqual(300_010);

    const strict = await llm.generateResponse(counted(300_000, { ...research, reasoning: { effort: 'high', strict: true } }));
    expect(cloudflareReasoningEffortCapability(strict.model)?.exact).toContain('high');
    expect(run.mock.calls.at(-1)?.[0]).toBe(strict.model);
  });

  it('is reported by client getRoutingInfo', () => {
    const { llm } = setup({ contextWindowModelSwitch: true });
    const routing = llm.getRoutingInfo(counted(300_000));
    expect(routing.provider).toBe('nvidia');
    expect(routing.model).toBe(NVIDIA_LARGE);
    expect(setup().llm.getRoutingInfo(counted(300_000)).model).toBe(NVIDIA_DEFAULT);
  });

  it('applies on the managed streaming path and reports the switch on the completion', async () => {
    const { llm } = setup({ contextWindowModelSwitch: true });
    const streamBody = () => new Response(
      'data: {"id":"s1","choices":[{"index":0,"delta":{"content":"streamed"},"finish_reason":null}]}\n\n'
      + 'data: {"id":"s1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\n'
      + 'data: [DONE]\n\n',
      { status: 200, headers: { 'Content-Type': 'text/event-stream' } },
    );
    const fetchMock = vi.fn().mockImplementation(async () => streamBody());
    vi.stubGlobal('fetch', fetchMock);
    const managed = await llm.generateManagedResponseStream(counted(300_000, { stream: true }));
    await drain(managed.stream);
    const completion = await managed.completion;
    expect(completion.outcome).toBe('success');
    expect(completion.model).toBe(NVIDIA_LARGE);
    expect(completion.contextWindowSwitch).toEqual({ from: NVIDIA_DEFAULT, to: NVIDIA_LARGE });

    const plain = await llm.generateManagedResponseStream(counted(1_000, { stream: true }));
    await drain(plain.stream);
    expect((await plain.completion).contextWindowSwitch).toBeUndefined();
    expect(sentModels(fetchMock)).toEqual([NVIDIA_LARGE, NVIDIA_DEFAULT]);
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
    expect(getCatalogEntry(secondModel)?.lifecycle).toBe('active');
  });

  it('only switches to a vision model for an image request', () => {
    const { llm } = setup({ contextWindowModelSwitch: true });
    const plan = planOf(llm, counted(300_000, { images: [{ data: 'aGk=', mimeType: 'image/png' }] }));
    for (const [provider, model] of plan.providerModels) {
      if (plan.modelSwitches.has(provider)) expect(getCatalogEntry(model)?.capabilities.supportsVision).toBe(true);
    }
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
