import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { LLMProviders, ServerError } from '../index';
import { OpenAIProvider } from '../providers/openai';
import { GroqProvider } from '../providers/groq';
import { CloudflareProvider } from '../providers/cloudflare';
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
function hooksThat(mode: 'throw' | 'reject'): ObservabilityHooks & Record<string, ReturnType<typeof vi.fn>> {
  return Object.fromEntries(methods.map(method => [method, vi.fn(() => {
    if (mode === 'throw') throw new Error(`${method} failed`);
    return Promise.reject(new Error(`${method} rejected`));
  })])) as never;
}
function quota() {
  return { check: vi.fn().mockResolvedValue({ allowed: true }), record: vi.fn(async () => {}) };
}

beforeEach(() => { vi.restoreAllMocks(); });
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('lifecycle hooks cannot change request outcomes (#116)', () => {
  it.each(['throw', 'reject'] as const)('a successful call is not replayed on another provider when every hook %ss', async mode => {
    const openai = vi.spyOn(OpenAIProvider.prototype, 'generateResponse').mockResolvedValue(ok('openai'));
    const groq = vi.spyOn(GroqProvider.prototype, 'generateResponse').mockResolvedValue(ok('groq'));
    const hooks = hooksThat(mode);
    const quotaHook = quota();
    const llm = new LLMProviders({ openai: { apiKey: 'k' }, groq: { apiKey: 'k' }, stateScope: 'client', hooks, quotaHook });

    const response = await llm.generateResponse(request);

    expect(response.provider).toBe('openai');
    expect(openai).toHaveBeenCalledOnce();
    expect(groq).not.toHaveBeenCalled();
    expect(hooks.onRequestEnd).toHaveBeenCalledOnce();
    // Billable usage is still recorded when telemetry fails.
    await vi.waitFor(() => expect(quotaHook.record).toHaveBeenCalledOnce());
  });

  it('throwing hooks leave fallback routing unchanged', async () => {
    vi.spyOn(OpenAIProvider.prototype, 'generateResponse').mockRejectedValue(new ServerError('openai', 'down'));
    const groq = vi.spyOn(GroqProvider.prototype, 'generateResponse').mockResolvedValue(ok('groq'));
    const hooks = hooksThat('throw');
    const llm = new LLMProviders({ openai: { apiKey: 'k' }, groq: { apiKey: 'k' }, stateScope: 'client', hooks });

    expect((await llm.generateResponse(request)).provider).toBe('groq');
    expect(groq).toHaveBeenCalledOnce();
    expect(hooks.onRequestError).toHaveBeenCalled();
    expect(hooks.onFallback).toHaveBeenCalledOnce();
  });

  it('throwing hooks do not change a terminal failure', async () => {
    const error = new ServerError('openai', 'down');
    vi.spyOn(OpenAIProvider.prototype, 'generateResponse').mockRejectedValue(error);
    const llm = new LLMProviders({ openai: { apiKey: 'k' }, stateScope: 'client', hooks: hooksThat('throw') });
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
      hooks: hooksThat('throw'), quotaHook });
    const result = await llm.generateManagedResponseStream({ ...request, model: '@cf/meta/llama-3.1-8b-instruct' });
    const reader = result.stream.getReader();
    let text = '';
    for (let chunk = await reader.read(); !chunk.done; chunk = await reader.read()) text += chunk.value;
    expect(text).toBe('hi');
    expect(await result.completion).toMatchObject({ outcome: 'success' });
    expect(run).toHaveBeenCalledOnce();
    expect(quotaHook.record).toHaveBeenCalledOnce();
  });

  it('class-based hooks keep their `this` binding', async () => {
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
});

describe('guardHooks / composeHooks', () => {
  it('guardHooks sees handlers added after construction', () => {
    const raw: ObservabilityHooks = {};
    const guarded = guardHooks(raw);
    const late = vi.fn();
    raw.onFallback = late;
    guarded.onFallback?.({ fromProvider: 'a', toProvider: 'b', reason: 'x', timestamp: 0 });
    expect(late).toHaveBeenCalledOnce();
  });

  it('composeHooks swallows async rejections and keeps calling later handlers', async () => {
    const unhandled = vi.fn();
    process.on('unhandledRejection', unhandled);
    try {
      const second = vi.fn();
      const composed = composeHooks({ onRequestStart: () => Promise.reject(new Error('async')) } as ObservabilityHooks,
        { onRequestStart: second });
      composed.onRequestStart?.({ provider: 'p', model: 'm', timestamp: 0 });
      await new Promise(resolve => setTimeout(resolve, 0));
      expect(second).toHaveBeenCalledOnce();
      expect(unhandled).not.toHaveBeenCalled();
    } finally {
      process.off('unhandledRejection', unhandled);
    }
  });
});
