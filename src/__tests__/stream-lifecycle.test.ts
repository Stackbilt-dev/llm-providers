import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CloudflareProvider } from '../providers/cloudflare';
import { OpenAIProvider } from '../providers/openai';
import { GroqProvider } from '../providers/groq';
import { CerebrasProvider } from '../providers/cerebras';
import { NvidiaProvider } from '../providers/nvidia';
import { AnthropicProvider } from '../providers/anthropic';
import { AuthenticationError, LLMProviders, QuotaDeniedError, QuotaExceededError, SchemaDriftError, TimeoutError } from '../index';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { defaultExhaustionRegistry } from '../utils/exhaustion';
import { getStreamUsage } from '../utils/stream-usage';
import type { LLMRequest } from '../types';

const encoder = new TextEncoder();
const request: LLMRequest = { messages: [{ role: 'user', content: 'hello' }] };
const cfModel = '@cf/meta/llama-3.1-8b-instruct';
function bytes(text: string, size = text.length * 4) {
  const encoded = encoder.encode(text);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (let i = 0; i < encoded.length; i += size) controller.enqueue(encoded.slice(i, i + size));
      controller.close();
    },
  });
}
async function drain(stream: ReadableStream<string>) {
  const reader = stream.getReader();
  let text = '';
  try {
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) return text;
      text += chunk.value;
    }
  } finally { reader.releaseLock(); }
}
function cf(source: ReadableStream<unknown>, timeout = 30000, streamBodyTimeout?: number) {
  const run = vi.fn().mockResolvedValue(source);
  const config = { ai: { run } as unknown as Ai, timeout, streamBodyTimeout };
  return { provider: new CloudflareProvider(config), config, run };
}

beforeEach(() => {
  defaultCircuitBreakerManager.resetAll();
  defaultExhaustionRegistry.reset();
});
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); vi.restoreAllMocks(); });

describe('Workers AI byte framing', () => {
  it.each([1, 2, 7, 10000])('decodes documentation-shaped SSE with chunk size %i', async size => {
    // Synthetic fixture follows https://blog.cloudflare.com/workers-ai-streaming/.
    const source = bytes(':keepalive\r\ndata:{"response":"Hé🙂"}\r\n\r\n'
      + 'data: {"response":" world"}\r\n\r\n'
      + 'data: {"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\r\n\r\n'
      + 'data: [DONE]\r\n\r\n', size);
    const { provider } = cf(source);
    const stream = await provider.streamResponse({ ...request, model: cfModel });
    expect(await drain(stream)).toBe('Hé🙂 world');
    expect(await getStreamUsage(stream)).toMatchObject({ inputTokens: 5, outputTokens: 2, totalTokens: 7 });
  });

  it('handles multiline data, bare CR, and final frames without trailing delimiters', async () => {
    const { provider } = cf(bytes('data: {\rdata: "response":"hello"}\r\rdata:{"response":"!"}', 1));
    expect(await drain(await provider.streamResponse({ ...request, model: cfModel }))).toBe('hello!');
  });

  it('supports object and string binding forms', async () => {
    const { provider } = cf(new ReadableStream({ start(controller) {
      controller.enqueue('hello'); controller.enqueue({ response: ' world' }); controller.close();
    } }));
    expect(await drain(await provider.streamResponse({ ...request, model: cfModel }))).toBe('hello world');
  });
});

const httpProviders = [
  ['openai', () => new OpenAIProvider({ apiKey: 'test', maxRetries: 0 })],
  ['groq', () => new GroqProvider({ apiKey: 'test', maxRetries: 0 })],
  ['cerebras', () => new CerebrasProvider({ apiKey: 'test', maxRetries: 0 })],
  ['nvidia', () => new NvidiaProvider({ apiKey: 'test', maxRetries: 0 })],
  ['anthropic', () => new AnthropicProvider({ apiKey: 'test', maxRetries: 0 })],
] as const;

describe.each(httpProviders)('%s terminal usage', (name, create) => {
  const usageFrame = name === 'anthropic'
    ? 'data:{"type":"message_start","message":{"usage":{"input_tokens":5,"output_tokens":2}}}\n\n'
    : 'data:{"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7},"choices":[]}\n\n';
  const textFrame = name === 'anthropic'
    ? 'data:{"type":"content_block_delta","delta":{"type":"text_delta","text":"hello"}}\n\n'
    : 'data:{"choices":[{"delta":{"content":"hello"}}]}\n\n';

  it('preserves usage and settles on malformed JSON', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, body: bytes(usageFrame + 'data:{broken}\n\n') }));
    const stream = await create().streamResponse(request);
    await expect(drain(stream)).rejects.toBeInstanceOf(SchemaDriftError);
    expect(await getStreamUsage(stream)).toMatchObject({ inputTokens: 5, outputTokens: 2 });
  });

  it('cancels upstream and settles partial usage without draining', async () => {
    const cancel = vi.fn();
    let reads = 0;
    const source = new ReadableStream<Uint8Array>({
      pull(controller) { reads++; controller.enqueue(encoder.encode(usageFrame + textFrame)); }, cancel,
    }, { highWaterMark: 0 });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, body: source }));
    const stream = await create().streamResponse(request);
    const reader = stream.getReader();
    expect((await reader.read()).value).toBe('hello');
    await reader.cancel('stop');
    expect(cancel).toHaveBeenCalledOnce();
    expect(reads).toBe(1);
    expect(await getStreamUsage(stream)).toMatchObject({ inputTokens: 5, outputTokens: 2 });
  });

  it('treats protocol error events as failures', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true,
      body: bytes(usageFrame + 'event:error\ndata:{"error":{"message":"upstream failed"}}\n\n'),
    }));
    const stream = await create().streamResponse(request);
    await expect(drain(stream)).rejects.toThrow('upstream failed');
    expect(await getStreamUsage(stream)).toMatchObject({ inputTokens: 5 });
  });

  it('aborts a pending fetch on cancellation before headers', async () => {
    let signal: AbortSignal | undefined;
    vi.stubGlobal('fetch', vi.fn((_url, options: RequestInit) => new Promise((_resolve, reject) => {
      signal = options.signal!;
      signal.addEventListener('abort', () => reject(signal!.reason), { once: true });
    })));
    const stream = await create().streamResponse(request);
    const reader = stream.getReader();
    const pending = reader.read();
    await Promise.resolve();
    await reader.cancel('stop');
    await pending;
    expect(signal?.aborted).toBe(true);
    expect(await getStreamUsage(stream)).toBeUndefined();
  });
});

describe('managed factory lifecycle', () => {
  const usageFrame = 'data:{"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n';
  const textFrame = 'data:{"response":"hello"}\n\n';

  it('records reported usage on first-read failure before any output', async () => {
    const { config } = cf(bytes(usageFrame + 'data:{broken}\n\n'));
    const record = vi.fn(async () => {});
    const llm = new LLMProviders({ cloudflare: config, stateScope: 'client', quotaHook: {
      check: vi.fn().mockResolvedValue({ allowed: true }), record,
    } });
    await expect(llm.generateManagedResponseStream({ ...request, model: cfModel })).rejects.toBeInstanceOf(SchemaDriftError);
    expect(record).toHaveBeenCalledOnce();
    expect(record).toHaveBeenCalledWith(expect.objectContaining({ inputTokens: 5, outputTokens: 2 }));
  });

  it('does not bill an authentication rejection or poison provider service health', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: { message: 'bad key' } }), { status: 401 })));
    const record = vi.fn(async () => {});
    const llm = new LLMProviders({ openai: { apiKey: 'test' }, stateScope: 'client', quotaHook: {
      check: vi.fn().mockResolvedValue({ allowed: true }), record,
    } });
    await expect(llm.generateManagedResponseStream({ ...request, model: 'gpt-4o-mini' })).rejects.toMatchObject({ code: 'AUTHENTICATION_ERROR' });
    expect(record).not.toHaveBeenCalled();
  });

  it('quarantines vendor-exhausted streaming accounts', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      error: { message: 'account exhausted', type: 'quota_exceeded' },
    }), { status: 429, headers: { 'Content-Type': 'application/json' } })));
    const llm = new LLMProviders({ openai: { apiKey: 'test' }, stateScope: 'client' });
    await expect(llm.generateResponseStream({ ...request, model: 'gpt-4o-mini' })).rejects.toMatchObject({ code: 'QUOTA_EXCEEDED' });
    expect(llm.getExhaustedProviders()).toContain('openai');
  });

  it('records partial failures once and does not splice a fallback answer', async () => {
    const { config, run } = cf(bytes(usageFrame + textFrame + 'data:{"error":{"message":"failed"}}\n\n'));
    const record = vi.fn(async () => {});
    const end = vi.fn();
    const llm = new LLMProviders({ cloudflare: config,
      quotaHook: { check: vi.fn().mockResolvedValue({ allowed: true }), record }, hooks: { onRequestEnd: end },
    });
    const result = await llm.generateManagedResponseStream({ ...request, model: cfModel });
    const reader = result.stream.getReader();
    expect((await reader.read()).value).toBe('hello');
    await expect(reader.read()).rejects.toThrow('failed');
    expect(await result.completion).toMatchObject({ outcome: 'error', emittedChars: 5, usage: { totalTokens: 7 } });
    expect(run).toHaveBeenCalledOnce();
    expect(record).toHaveBeenCalledOnce();
    expect(end).toHaveBeenCalledOnce();
  });

  it('cancellation racing a pending read settles as cancelled exactly once', async () => {
    const cancelled = vi.fn();
    let reads = 0;
    const source = new ReadableStream<Uint8Array>({
      pull(controller) { if (reads++ === 0) controller.enqueue(encoder.encode(usageFrame + textFrame)); },
      cancel: cancelled,
    }, { highWaterMark: 0 });
    const { config } = cf(source);
    const record = vi.fn(async () => {});
    const llm = new LLMProviders({ cloudflare: config, quotaHook: {
      check: vi.fn().mockResolvedValue({ allowed: true }), record,
    } });
    const result = await llm.generateManagedResponseStream({ ...request, model: cfModel });
    const reader = result.stream.getReader();
    await reader.read();
    const pending = reader.read();
    await reader.cancel('stop');
    await pending;
    expect(await result.completion).toMatchObject({ outcome: 'cancelled', emittedChars: 5 });
    expect(cancelled).toHaveBeenCalledOnce();
    expect(record).toHaveBeenCalledOnce();
  });

  it('awaits quota persistence before successful completion', async () => {
    const { config } = cf(bytes(usageFrame + textFrame + 'data:[DONE]\n\n'));
    let settle!: () => void;
    const record = vi.fn(() => new Promise<void>(resolve => { settle = resolve; }));
    const llm = new LLMProviders({ cloudflare: config, quotaHook: {
      check: vi.fn().mockResolvedValue({ allowed: true }), record,
    } });
    const result = await llm.generateManagedResponseStream({ ...request, model: cfModel });
    const reader = result.stream.getReader();
    await reader.read();
    const last = reader.read();
    let completed = false;
    void result.completion.then(() => { completed = true; });
    await vi.waitFor(() => expect(record).toHaveBeenCalledOnce());
    expect(completed).toBe(false);
    settle();
    expect((await last).done).toBe(true);
    expect((await result.completion).outcome).toBe('success');
  });

  it('rejects only completion on settlement failure; delivered text still closes cleanly', async () => {
    const { config, run } = cf(bytes(textFrame + 'data:[DONE]\n\n'));
    const llm = new LLMProviders({ cloudflare: config, quotaHook: {
      check: vi.fn().mockResolvedValue({ allowed: true }), record: vi.fn().mockRejectedValue(new Error('record failed')),
    } });
    const result = await llm.generateManagedResponseStream({ ...request, model: cfModel });
    expect(await drain(result.stream)).toBe('hello');
    await expect(result.completion).rejects.toThrow('record failed');
    expect(run).toHaveBeenCalledOnce();
  });

  it('legacy generateResponseStream does not error a delivered stream on ledger failure', async () => {
    const { config } = cf(bytes(textFrame + 'data:[DONE]\n\n'));
    const llm = new LLMProviders({ cloudflare: config, stateScope: 'client', quotaHook: {
      check: vi.fn().mockResolvedValue({ allowed: true }), record: vi.fn().mockRejectedValue(new Error('ledger down')),
    } });
    expect(await drain(await llm.generateResponseStream({ ...request, model: cfModel }))).toBe('hello');
  });

  it('a ledger failure on a pre-output error does not mask the provider error', async () => {
    const { config } = cf(bytes(usageFrame + 'data:{broken}\n\n'));
    const llm = new LLMProviders({ cloudflare: config, stateScope: 'client', quotaHook: {
      check: vi.fn().mockResolvedValue({ allowed: true }), record: vi.fn().mockRejectedValue(new Error('ledger down')),
    } });
    await expect(llm.generateManagedResponseStream({ ...request, model: cfModel })).rejects.toBeInstanceOf(SchemaDriftError);
  });

  it('does not charge an estimate for a Workers AI rejection at stream open', async () => {
    const run = vi.fn().mockRejectedValue(new AuthenticationError('cloudflare'));
    const record = vi.fn(async () => {});
    const end = vi.fn();
    const llm = new LLMProviders({ cloudflare: { ai: { run } as unknown as Ai }, stateScope: 'client',
      quotaHook: { check: vi.fn().mockResolvedValue({ allowed: true }), record }, hooks: { onRequestEnd: end } });
    await expect(llm.generateManagedResponseStream({ ...request, model: cfModel })).rejects.toBeInstanceOf(AuthenticationError);
    expect(record).not.toHaveBeenCalled();
    expect(end).not.toHaveBeenCalled();
  });

  it('a pre-output 5xx fails over and charges only the provider that served the stream', async () => {
    const groqSse = 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
      + 'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\ndata: [DONE]\n\n';
    vi.stubGlobal('fetch', vi.fn(async (url: string) => String(url).includes('openai.com')
      ? new Response(JSON.stringify({ error: { message: 'unavailable' } }), { status: 503 })
      : new Response(bytes(groqSse), { status: 200, headers: { 'Content-Type': 'text/event-stream' } })));
    const record = vi.fn(async () => {});
    const end = vi.fn();
    const llm = new LLMProviders({ openai: { apiKey: 'test', maxRetries: 0 }, groq: { apiKey: 'test', maxRetries: 0 },
      stateScope: 'client', quotaHook: { check: vi.fn().mockResolvedValue({ allowed: true }), record },
      hooks: { onRequestEnd: end } });
    const result = await llm.generateManagedResponseStream({ ...request, model: 'gpt-4o-mini' });
    expect(await drain(result.stream)).toBe('ok');
    await result.completion;
    expect(record).toHaveBeenCalledOnce();
    expect(record).toHaveBeenCalledWith(expect.objectContaining({ provider: 'groq' }));
    expect(end).toHaveBeenCalledOnce();
  });

  it('local quota denial stays a QuotaExceededError for 1.x callers without exhausting the provider', async () => {
    const { config, run } = cf(bytes(textFrame));
    const llm = new LLMProviders({ cloudflare: config, stateScope: 'client', quotaHook: {
      check: vi.fn().mockResolvedValue({ allowed: false, reason: 'over budget' }), record: vi.fn(),
    } });
    const error = await llm.generateResponseStream({ ...request, model: cfModel }).catch(e => e);
    expect(error).toBeInstanceOf(QuotaDeniedError);
    expect(error).toBeInstanceOf(QuotaExceededError);
    expect(error).toMatchObject({ code: 'QUOTA_EXCEEDED', localPolicy: true });
    expect(llm.getExhaustedProviders()).not.toContain('cloudflare');
    expect(run).not.toHaveBeenCalled();
  });

  it('does not read past the prefetched first chunk for an idle consumer', async () => {
    let reads = 0;
    const { config } = cf(new ReadableStream<Uint8Array>({
      pull(controller) { reads++; controller.enqueue(encoder.encode(textFrame)); },
    }, { highWaterMark: 0 }));
    const llm = new LLMProviders({ cloudflare: config });
    const result = await llm.generateManagedResponseStream({ ...request, model: cfModel });
    await Promise.resolve();
    expect(reads).toBe(1);
    await result.stream.cancel();
    expect((await result.completion).outcome).toBe('cancelled');
  });

  it('opt-in streamBodyTimeout times out a stalled body and settles estimated usage', async () => {
    vi.useFakeTimers();
    const cancel = vi.fn();
    const { provider } = cf(new ReadableStream({ cancel }), 30000, 20);
    const stream = await provider.streamResponse({ ...request, model: cfModel });
    const failed = expect(drain(stream)).rejects.toBeInstanceOf(TimeoutError);
    await vi.advanceTimersByTimeAsync(21);
    await failed;
    expect(cancel).toHaveBeenCalledOnce();
    expect(await getStreamUsage(stream)).toMatchObject({ outputTokens: 0, tokenProvenance: 'estimated' });
  });

  it('the provider timeout covers opening only and does not cut off a flowing stream', async () => {
    vi.useFakeTimers();
    let n = 0;
    const source = new ReadableStream<Uint8Array>({
      async pull(controller) {
        await new Promise(resolve => setTimeout(resolve, 10));
        if (n === 8) { controller.enqueue(encoder.encode('data: [DONE]\n\n')); controller.close(); return; }
        controller.enqueue(encoder.encode('data:{"response":"t' + (n++) + '"}\n\n'));
      },
    }, { highWaterMark: 0 });
    const { provider } = cf(source, 30);
    const stream = await provider.streamResponse({ ...request, model: cfModel });
    const text = drain(stream);
    await vi.advanceTimersByTimeAsync(200);
    expect(await text).toBe('t0t1t2t3t4t5t6t7');
  });

  it('the provider timeout still fails a stream that never opens', async () => {
    vi.useFakeTimers();
    const run = vi.fn(() => new Promise(() => {}));
    const provider = new CloudflareProvider({ ai: { run } as unknown as Ai, timeout: 20 });
    const stream = await provider.streamResponse({ ...request, model: cfModel });
    const failed = expect(drain(stream)).rejects.toBeInstanceOf(TimeoutError);
    await vi.advanceTimersByTimeAsync(21);
    await failed;
  });
});
