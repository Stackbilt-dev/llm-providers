import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { LLMProviders, ResilienceContext, CircuitBreaker, InvalidRequestError, AuthenticationError } from '../index';
import { LLMProviderFactory } from '../factory';
import { OpenAIProvider } from '../providers/openai';
import { CloudflareProvider } from '../providers/cloudflare';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { QuotaExceededError } from '../errors';

const request = { messages: [{ role: 'user' as const, content: 'hello' }], model: 'gpt-4o-mini' };
function response() {
  return new Response(JSON.stringify({
    id: 'test', model: 'gpt-4o-mini', choices: [{ message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
  }), { status: 200 });
}
beforeEach(() => { defaultCircuitBreakerManager.resetAll(); });
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers(); });

describe('circuit breaker configuration defaults', () => {
  it('leaves the factory breaker unset by default, as in 1.21', async () => {
    vi.spyOn(OpenAIProvider.prototype, 'healthCheck').mockResolvedValue(true);
    const llm = new LLMProviderFactory({ openai: { apiKey: 'test' }, stateScope: 'client' });
    expect((await llm.getProviderHealth()).openai).toMatchObject({ healthy: true, circuitBreaker: null });
  });

  it('a per-provider enableCircuitBreaker setting wins over the factory setting', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response('{}', { status: 500 }));
    vi.stubGlobal('fetch', fetch);
    const context = new ResilienceContext();
    const llm = new LLMProviders({ openai: { apiKey: 'test', maxRetries: 0, enableCircuitBreaker: false },
      enableCircuitBreaker: true, resilienceContext: context });
    for (let i = 0; i < 8; i++) await llm.generateResponse(request).catch(() => {});
    expect(fetch).toHaveBeenCalledTimes(8);
  });
});

describe('resilience state ownership', () => {
  it('client-local failure and reset cannot alter a second client', async () => {
    const a = new ResilienceContext();
    const b = new ResilienceContext();
    const first = new LLMProviderFactory({ openai: { apiKey: 'a' }, resilienceContext: a, enableCircuitBreaker: true });
    const second = new LLMProviderFactory({ openai: { apiKey: 'b' }, resilienceContext: b, enableCircuitBreaker: true, costOptimization: true });
    a.circuitBreakers.getBreaker('openai').forceOpen();
    a.exhaustion.markExhausted('openai');
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response()));
    await expect(second.generateResponse(request)).resolves.toMatchObject({ message: 'ok' });
    expect(b.circuitBreakers.getBreaker('openai').getState().totalSuccesses).toBe(1);
    expect(b.costTracker.getTotalCost()).toBeGreaterThan(0);
    first.reset();
    expect(b.circuitBreakers.getBreaker('openai').getState().totalSuccesses).toBe(1);
    expect(b.costTracker.getTotalCost()).toBeGreaterThan(0);
    expect(second.getLatencyHistogram().openai.count).toBe(1);
  });

  it('opt-in client state does not inherit global exhaustion or failures', async () => {
    defaultCircuitBreakerManager.getBreaker('openai').forceOpen();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response()));
    const llm = LLMProviders.fromEnv({ OPENAI_API_KEY: 'test' }, { stateScope: 'client', enableCircuitBreaker: true });
    await expect(llm.generateResponse(request)).resolves.toMatchObject({ message: 'ok' });
  });

  it('intentionally shared contexts share exhaustion', async () => {
    const context = new ResilienceContext();
    const a = new LLMProviderFactory({ openai: { apiKey: 'same' }, resilienceContext: context });
    const b = new LLMProviderFactory({ openai: { apiKey: 'same' }, resilienceContext: context });
    vi.spyOn(OpenAIProvider.prototype, 'generateResponse').mockRejectedValue(new QuotaExceededError('openai'));
    await expect(a.generateResponse(request)).rejects.toBeInstanceOf(QuotaExceededError);
    expect(b.getExhaustedProviders()).toEqual(['openai']);
  });

  it('breaker disable bypasses the adapter even when its context is force-open', async () => {
    const context = new ResilienceContext();
    context.circuitBreakers.getBreaker('openai').forceOpen();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response()));
    const factory = new LLMProviderFactory({ openai: { apiKey: 'test' }, resilienceContext: context, enableCircuitBreaker: false });
    await expect(factory.generateResponse(request)).resolves.toMatchObject({ message: 'ok' });
    expect(context.circuitBreakers.getBreaker('openai').getState().totalSuccesses).toBe(0);
  });

  it('direct provider injection uses the same breaker as the client', async () => {
    const context = new ResilienceContext();
    const provider = new OpenAIProvider({ apiKey: 'test', resilienceContext: context });
    context.circuitBreakers.getBreaker('openai').forceOpen();
    vi.stubGlobal('fetch', vi.fn());
    await expect(provider.generateResponse(request)).rejects.toMatchObject({ code: 'CIRCUIT_BREAKER_OPEN' });
    expect(fetch).not.toHaveBeenCalled();
  });

  it('stream success updates the shared breaker only after drain', async () => {
    const context = new ResilienceContext();
    const stream = new ReadableStream({ start(controller) { controller.enqueue({ response: 'ok' }); controller.close(); } });
    const provider = new CloudflareProvider({ ai: { run: vi.fn().mockResolvedValue(stream) } as unknown as Ai, resilienceContext: context });
    const result = await provider.streamResponse({ messages: request.messages });
    const reader = result.getReader();
    await reader.read();
    expect(context.circuitBreakers.getBreaker('cloudflare').getState().totalSuccesses).toBe(0);
    await reader.read();
    expect(context.circuitBreakers.getBreaker('cloudflare').getState().totalSuccesses).toBe(1);
  });

  it('stream schema failure degrades health, while cancellation is neutral', async () => {
    const context = new ResilienceContext();
    const create = (body: string) => new CloudflareProvider({
      ai: { run: vi.fn().mockResolvedValue(new ReadableStream({ start(controller) {
        controller.enqueue(new TextEncoder().encode(body)); controller.close();
      } })) } as unknown as Ai, resilienceContext: context,
    });
    const failed = await create('data:{bad}\n\n').streamResponse({ messages: request.messages });
    await expect(failed.getReader().read()).rejects.toMatchObject({ code: 'SCHEMA_DRIFT' });
    const breaker = context.circuitBreakers.getBreaker('cloudflare');
    expect(breaker.getState().totalFailures).toBe(1);
    vi.spyOn(Math, 'random').mockReturnValue(0);
    const cancelled = await create('data:{"response":"ok"}\n\n').streamResponse({ messages: request.messages });
    const reader = cancelled.getReader();
    await reader.read(); await reader.cancel();
    expect(breaker.getState().totalFailures).toBe(1);
    expect(breaker.getState().totalSuccesses).toBe(0);
  });
});

describe('breaker recovery and error policy', () => {
  it('rejected traffic cannot postpone decay', async () => {
    vi.useFakeTimers();
    const breaker = new CircuitBreaker('test', { resetTimeout: 100, failureThreshold: 1, degradationCurve: [1] });
    breaker.forceOpen();
    await vi.advanceTimersByTimeAsync(90);
    await expect(breaker.execute(async () => 'ok')).rejects.toMatchObject({ code: 'CIRCUIT_BREAKER_OPEN' });
    await vi.advanceTimersByTimeAsync(11);
    expect(breaker.getState().consecutiveFailures).toBe(0);
    await expect(breaker.execute(async () => 'ok')).resolves.toBe('ok');
  });

  it.each([new InvalidRequestError('test'), new AuthenticationError('test'), new QuotaExceededError('test')])(
    'keeps caller/account errors out of service health: %s', async error => {
      const breaker = new CircuitBreaker('test');
      await expect(breaker.execute(async () => { throw error; })).rejects.toBe(error);
      expect(breaker.getState().totalFailures).toBe(0);
    }
  );

  it('settles concurrent outcomes in completion order and each permit once', () => {
    const breaker = new CircuitBreaker('test');
    const slow = breaker.beginExecution();
    const fast = breaker.beginExecution();
    fast.failure(new Error('network'));
    slow.success(); slow.failure(new Error('duplicate')); fast.success();
    expect(breaker.getState()).toMatchObject({ totalFailures: 1, totalSuccesses: 1, consecutiveFailures: 0 });
  });
});
