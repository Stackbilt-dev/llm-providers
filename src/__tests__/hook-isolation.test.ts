import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { LLMProviders, SchemaDriftError, ServerError } from '../index';
import { OpenAIProvider } from '../providers/openai';
import { GroqProvider } from '../providers/groq';
import { composeHooks, guardHooks, type ObservabilityHooks } from '../utils/hooks';
import type { LLMRequest, LLMResponse } from '../types';

const request: LLMRequest = { messages: [{ role: 'user', content: 'hello' }], model: 'gpt-4o-mini', tenantId: 't' };
const ok = (provider: string): LLMResponse => ({
  message: 'ok', provider, model: provider === 'openai' ? 'gpt-4o-mini' : 'llama', responseTime: 1,
  usage: { inputTokens: 5, outputTokens: 2, totalTokens: 7, cost: 0.001 },
});
const methods = [
  'onRequestStart', 'onRequestEnd', 'onCache', 'onRequestError', 'onRetry', 'onFallback',
  'onCircuitStateChange', 'onQuotaExhausted', 'onBudgetThreshold', 'onQuotaCheck', 'onQuotaDenied',
  'onProviderBalance', 'onSchemaDrift', 'onImageAnalysisAttempt', 'onImageAnalysisComplete',
] as const;
function failingHooks(mode: 'throw' | 'reject'): ObservabilityHooks & Record<string, ReturnType<typeof vi.fn>> {
  return Object.fromEntries(methods.map(method => [method, vi.fn(() => {
    if (mode === 'throw') throw new Error(`${method} failed`);
    return Promise.reject(new Error(`${method} rejected`));
  })])) as never;
}
function quota() {
  return { check: vi.fn().mockResolvedValue({ allowed: true }), record: vi.fn(async () => {}) };
}
/** Collects unhandled rejections raised while `run` executes, including on the next macrotask. */
async function unhandledDuring(run: () => Promise<void>): Promise<unknown[]> {
  const seen: unknown[] = [];
  const listener = (reason: unknown) => { seen.push(reason); };
  process.on('unhandledRejection', listener);
  try {
    await run();
    await new Promise(resolve => setTimeout(resolve, 0));
    await new Promise(resolve => setTimeout(resolve, 0));
  } finally {
    process.off('unhandledRejection', listener);
  }
  return seen;
}

beforeEach(() => { vi.restoreAllMocks(); });
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('lifecycle hooks cannot change request outcomes (#116)', () => {
  it('a throwing onRequestEnd alone does not replay a successful call on another provider', async () => {
    const openai = vi.spyOn(OpenAIProvider.prototype, 'generateResponse').mockResolvedValue(ok('openai'));
    const groq = vi.spyOn(GroqProvider.prototype, 'generateResponse').mockResolvedValue(ok('groq'));
    const quotaHook = quota();
    const onRequestEnd = vi.fn(() => { throw new Error('telemetry sink down'); });
    const llm = new LLMProviders({ openai: { apiKey: 'k' }, groq: { apiKey: 'k' }, stateScope: 'client',
      hooks: { onRequestEnd }, quotaHook });

    const response = await llm.generateResponse(request);

    expect(response.provider).toBe('openai');
    expect(openai).toHaveBeenCalledOnce();
    expect(groq).not.toHaveBeenCalled();
    expect(onRequestEnd).toHaveBeenCalledOnce();
    // Billable usage is still recorded when telemetry fails.
    await vi.waitFor(() => expect(quotaHook.record).toHaveBeenCalledOnce());
  });

  it('rejecting async hooks raise no unhandled rejections and change nothing', async () => {
    vi.spyOn(OpenAIProvider.prototype, 'generateResponse').mockResolvedValue(ok('openai'));
    const groq = vi.spyOn(GroqProvider.prototype, 'generateResponse').mockResolvedValue(ok('groq'));
    const hooks = failingHooks('reject');
    const llm = new LLMProviders({ openai: { apiKey: 'k' }, groq: { apiKey: 'k' }, stateScope: 'client', hooks });

    const unhandled = await unhandledDuring(async () => {
      expect((await llm.generateResponse(request)).provider).toBe('openai');
    });

    expect(unhandled).toEqual([]);
    expect(groq).not.toHaveBeenCalled();
    expect(hooks.onRequestEnd).toHaveBeenCalledOnce();
  });

  it('throwing hooks leave fallback routing unchanged', async () => {
    vi.spyOn(OpenAIProvider.prototype, 'generateResponse').mockRejectedValue(new ServerError('openai', 'down'));
    const groq = vi.spyOn(GroqProvider.prototype, 'generateResponse').mockResolvedValue(ok('groq'));
    const hooks = failingHooks('throw');
    const llm = new LLMProviders({ openai: { apiKey: 'k' }, groq: { apiKey: 'k' }, stateScope: 'client', hooks });

    expect((await llm.generateResponse(request)).provider).toBe('groq');
    expect(groq).toHaveBeenCalledOnce();
    expect(hooks.onRequestError).toHaveBeenCalled();
    expect(hooks.onFallback).toHaveBeenCalledOnce();
  });

  it('throwing hooks do not change a terminal failure', async () => {
    const error = new ServerError('openai', 'down');
    vi.spyOn(OpenAIProvider.prototype, 'generateResponse').mockRejectedValue(error);
    const llm = new LLMProviders({ openai: { apiKey: 'k' }, stateScope: 'client', hooks: failingHooks('throw') });
    await expect(llm.generateResponse(request)).rejects.toBe(error);
  });

  it('throwing hooks do not break a managed stream or its accounting', async () => {
    const source = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('data:{"response":"hi"}\n\n'
          + 'data:{"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\ndata:[DONE]\n\n'));
        controller.close();
      },
    });
    const run = vi.fn().mockResolvedValue(source);
    const quotaHook = quota();
    const llm = new LLMProviders({ cloudflare: { ai: { run } as unknown as Ai }, stateScope: 'client',
      hooks: failingHooks('throw'), quotaHook });
    const result = await llm.generateManagedResponseStream({ ...request, model: '@cf/meta/llama-3.1-8b-instruct' });
    const reader = result.stream.getReader();
    let text = '';
    for (let chunk = await reader.read(); !chunk.done; chunk = await reader.read()) text += chunk.value;
    expect(text).toBe('hi');
    expect(await result.completion).toMatchObject({ outcome: 'success' });
    expect(run).toHaveBeenCalledOnce();
    expect(quotaHook.record).toHaveBeenCalledOnce();
  });

  it.each([
    ['frozen', (fn: () => void) => Object.freeze({ onRequestEnd: fn })],
    ['read-only via defineProperty', (fn: () => void) => Object.defineProperty({}, 'onRequestEnd', { value: fn, enumerable: true })],
  ] as const)('%s hooks objects still receive events without replaying the call', async (_label, build) => {
    vi.spyOn(OpenAIProvider.prototype, 'generateResponse').mockResolvedValue(ok('openai'));
    const groq = vi.spyOn(GroqProvider.prototype, 'generateResponse').mockResolvedValue(ok('groq'));
    const onRequestEnd = vi.fn();
    const llm = new LLMProviders({ openai: { apiKey: 'k' }, groq: { apiKey: 'k' }, stateScope: 'client',
      hooks: build(onRequestEnd) as ObservabilityHooks });

    expect((await llm.generateResponse(request)).provider).toBe('openai');
    expect(groq).not.toHaveBeenCalled();
    expect(onRequestEnd).toHaveBeenCalledOnce();
  });

  it('class-instance hooks keep `this` through the factory guard', async () => {
    vi.spyOn(OpenAIProvider.prototype, 'generateResponse').mockResolvedValue(ok('openai'));
    class Collector {
      events: string[] = [];
      onRequestEnd(event: { provider: string }) { this.events.push(event.provider); }
    }
    const collector = new Collector();
    const llm = new LLMProviders({ openai: { apiKey: 'k' }, stateScope: 'client', hooks: collector });
    await llm.generateResponse(request);
    expect(collector.events).toEqual(['openai']);
  });

  it('logs each failing hook once instead of failing silently', async () => {
    vi.spyOn(OpenAIProvider.prototype, 'generateResponse').mockResolvedValue(ok('openai'));
    const logger = { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
    const llm = new LLMProviders({ openai: { apiKey: 'k' }, stateScope: 'client', logger,
      hooks: { onRequestEnd: () => { throw new Error('sink down'); } } });
    await llm.generateResponse(request);
    await llm.generateResponse(request);
    const hookWarnings = logger.warn.mock.calls.filter(([message]) => String(message).includes('onRequestEnd'));
    expect(hookWarnings).toHaveLength(1);
    expect(hookWarnings[0][1]).toBe('sink down');
  });
});

describe('directly constructed adapters', () => {
  it('contain a rejecting async onSchemaDrift hook', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(new ReadableStream<Uint8Array>({
      start(controller) { controller.enqueue(new TextEncoder().encode('data:{broken}\n\n')); controller.close(); },
    }), { status: 200, headers: { 'Content-Type': 'text/event-stream' } })));
    const onSchemaDrift = vi.fn(async () => { throw new Error('drift sink down'); });
    const provider = new OpenAIProvider({ apiKey: 'k', enableCircuitBreaker: false, hooks: { onSchemaDrift } });

    const unhandled = await unhandledDuring(async () => {
      const reader = (await provider.streamResponse(request)).getReader();
      await expect(reader.read()).rejects.toBeInstanceOf(SchemaDriftError);
    });

    expect(onSchemaDrift).toHaveBeenCalledOnce();
    expect(unhandled).toEqual([]);
  });

  it('re-guard hooks replaced through updateConfig', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(new ReadableStream<Uint8Array>({
      start(controller) { controller.enqueue(new TextEncoder().encode('data:{broken}\n\n')); controller.close(); },
    }), { status: 200, headers: { 'Content-Type': 'text/event-stream' } })));
    const provider = new OpenAIProvider({ apiKey: 'k', enableCircuitBreaker: false });
    const onSchemaDrift = vi.fn(async () => { throw new Error('drift sink down'); });
    provider.updateConfig({ hooks: { onSchemaDrift } });

    const unhandled = await unhandledDuring(async () => {
      const reader = (await provider.streamResponse(request)).getReader();
      await expect(reader.read()).rejects.toBeInstanceOf(SchemaDriftError);
    });

    expect(onSchemaDrift).toHaveBeenCalledOnce();
    expect(unhandled).toEqual([]);
  });
});

describe('guardHooks', () => {
  it('exposes only the methods the caller defined, resolved at call time', () => {
    const raw: ObservabilityHooks = {};
    const guarded = guardHooks(raw);
    expect(guarded.onRetry).toBeUndefined();
    const late = vi.fn();
    raw.onFallback = late;
    guarded.onFallback?.({ fromProvider: 'a', toProvider: 'b', reason: 'x', timestamp: 0 });
    expect(late).toHaveBeenCalledOnce();
  });

  it('handles frozen hooks and keeps wrapper identity and introspection stable', () => {
    const onFallback = vi.fn();
    const guarded = guardHooks(Object.freeze({ onFallback }));
    expect(guarded.onFallback).toBe(guarded.onFallback);
    expect('onFallback' in guarded).toBe(true);
    expect('onRetry' in guarded).toBe(false);
    expect(Object.keys(guarded)).toEqual(['onFallback']);
    expect(String(guarded)).toBe('[object Object]');
    guarded.onFallback?.({ fromProvider: 'a', toProvider: 'b', reason: 'x', timestamp: 0 });
    expect(onFallback).toHaveBeenCalledOnce();
  });

  it('is idempotent, so factory-built adapters are not double wrapped', () => {
    const guarded = guardHooks({ onRequestStart: vi.fn() });
    expect(guardHooks(guarded)).toBe(guarded);
  });
});

describe('composeHooks', () => {
  it('calls class-based handlers with their implementation as `this`', () => {
    class Collector {
      events: string[] = [];
      onRequestStart(event: { provider: string }) { this.events.push(event.provider); }
    }
    const collector = new Collector();
    composeHooks(collector as ObservabilityHooks).onRequestStart?.({ provider: 'p', model: 'm', timestamp: 0 });
    expect(collector.events).toEqual(['p']);
  });

  it('captures handlers when composing, as in 1.21', () => {
    const original = vi.fn();
    const replacement = vi.fn();
    const impl: ObservabilityHooks = { onRequestStart: original };
    const composed = composeHooks(impl);
    impl.onRequestStart = replacement;
    composed.onRequestStart?.({ provider: 'p', model: 'm', timestamp: 0 });
    expect(original).toHaveBeenCalledOnce();
    expect(replacement).not.toHaveBeenCalled();
  });

  it('contains async rejections and keeps calling later handlers', async () => {
    const second = vi.fn();
    const unhandled = await unhandledDuring(async () => {
      composeHooks({ onRequestStart: () => Promise.reject(new Error('async')) } as ObservabilityHooks,
        { onRequestStart: second }).onRequestStart?.({ provider: 'p', model: 'm', timestamp: 0 });
    });
    expect(second).toHaveBeenCalledOnce();
    expect(unhandled).toEqual([]);
  });
});
