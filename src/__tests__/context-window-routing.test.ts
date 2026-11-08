/**
 * #136 / #113: the context window is a routing constraint. Models whose window can't
 * hold the estimated input are dropped from ranking and from the fallback chain,
 * and a request no configured model can hold is refused before any call.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  estimateRequestInputTokens,
  fitsContextWindow,
  getCatalogEntry,
  rankModels,
} from '../model-catalog';
import { LLMProviders, NoRouteError } from '../index';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { defaultExhaustionRegistry } from '../utils/exhaustion';
import type { LLMRequest } from '../types';

/** A single user message of roughly `tokens` estimated input tokens. */
function requestOf(tokens: number, extra: Partial<LLMRequest> = {}): LLMRequest {
  return { messages: [{ role: 'user', content: 'x'.repeat(tokens * 4) }], maxTokens: 10, ...extra };
}

function chatResult(content: string) {
  return {
    id: 'chatcmpl-test',
    choices: [{ index: 0, message: { role: 'assistant', content }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
  };
}

function groqResponse() {
  return new Response(JSON.stringify({
    id: 'groq-1',
    object: 'chat.completion',
    created: 1,
    model: 'llama-3.3-70b-versatile',
    choices: [{ index: 0, message: { role: 'assistant', content: 'from groq' }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
  }), { status: 200, headers: { 'Content-Type': 'application/json' } });
}

function setup(cloudflare: 'ok' | 'fail') {
  const run = cloudflare === 'ok'
    ? vi.fn().mockResolvedValue(chatResult('from cloudflare'))
    : vi.fn().mockRejectedValue(new Error('binding exploded'));
  const fetchMock = vi.fn().mockImplementation(async () => groqResponse());
  vi.stubGlobal('fetch', fetchMock);
  const llm = new LLMProviders({
    cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 },
    groq: { apiKey: 'test', maxRetries: 0 },
    stateScope: 'client',
  });
  return { llm, run, fetchMock };
}

beforeEach(() => {
  defaultCircuitBreakerManager.resetAll();
  defaultExhaustionRegistry.reset();
  vi.spyOn(console, "log").mockImplementation(() => {});
  vi.spyOn(console, "warn").mockImplementation(() => {});
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('context-window estimate', () => {
  it('counts message and system-prompt characters divided by four', () => {
    expect(estimateRequestInputTokens({
      systemPrompt: 'abcd',
      messages: [{ role: 'user', content: 'x'.repeat(40) }, { role: 'assistant', content: 'y'.repeat(4) }],
    })).toBe(12);
    expect(estimateRequestInputTokens({ messages: [] })).toBe(1);
  });

  it('checks only the input against the window, not maxTokens', () => {
    const groq = getCatalogEntry('llama-3.3-70b-versatile')!;
    const window = groq.capabilities.maxContextLength;
    expect(fitsContextWindow(groq, requestOf(window))).toBe(true);
    expect(fitsContextWindow(groq, requestOf(window + 1))).toBe(false);
    expect(fitsContextWindow(groq, requestOf(window - 10, { maxTokens: 100_000 }))).toBe(true);
    expect(fitsContextWindow(groq, {})).toBe(true);
  });
});

describe('ranking prefers models that can hold the input', () => {
  it('leaves out models whose window is too small when a larger one exists', () => {
    const request = requestOf(150_000);
    const ranked = rankModels('LONG_CONTEXT', ['cloudflare', 'groq'], { request });
    expect(ranked.length).toBeGreaterThan(0);
    expect(ranked.every(entry => entry.capabilities.maxContextLength >= 150_000)).toBe(true);
    expect(ranked.some(entry => entry.provider === 'groq')).toBe(false);
  });

  it('keeps the full ranking when no model can hold the input', () => {
    const request = requestOf(1_100_000);
    const ranked = rankModels('LONG_CONTEXT', ['cloudflare', 'groq'], { request });
    expect(ranked.map(entry => entry.model)).toEqual(rankModels('LONG_CONTEXT', ['cloudflare', 'groq']).map(entry => entry.model));
  });

  it('does not change rankings for requests every model can hold', () => {
    const request = requestOf(1_000);
    expect(rankModels('BALANCED', ['cloudflare', 'groq'], { request }).map(entry => entry.model))
      .toEqual(rankModels('BALANCED', ['cloudflare', 'groq'], { request: { ...request, messages: [{ role: 'user', content: 'hi' }] } }).map(entry => entry.model));
  });
});

describe('factory fallback respects context windows', () => {
  it('falls back to a smaller-window provider when the input fits it (control)', async () => {
    const { llm, fetchMock } = setup('fail');
    const response = await llm.generateResponse(requestOf(1_000));
    expect(response.provider).toBe('groq');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('does not fall back to a provider whose model cannot hold the input', async () => {
    const { llm, run, fetchMock } = setup('fail');
    await expect(llm.generateResponse(requestOf(150_000))).rejects.toThrow('binding exploded');
    expect(run).toHaveBeenCalledTimes(1);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('serves a large request from the provider whose model can hold it', async () => {
    const { llm, run, fetchMock } = setup('ok');
    const response = await llm.generateResponse(requestOf(150_000));
    expect(response.provider).toBe('cloudflare');
    expect(run).toHaveBeenCalledTimes(1);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('refuses a request no configured model can hold, before any call', async () => {
    const { llm, run, fetchMock } = setup('ok');
    const error = await llm.generateResponse(requestOf(1_100_000)).catch(caught => caught);
    expect(error).toBeInstanceOf(NoRouteError);
    expect(error.message).toContain('1100000 tokens');
    expect(run).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('refuses it on the streaming path too', async () => {
    const { llm, run, fetchMock } = setup('ok');
    await expect(llm.generateResponseStream(requestOf(1_100_000))).rejects.toBeInstanceOf(NoRouteError);
    expect(run).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('still sends a pinned model the request; its provider decides', async () => {
    const { llm, run, fetchMock } = setup('ok');
    const response = await llm.generateResponse(requestOf(150_000, { model: 'llama-3.3-70b-versatile' }));
    expect(response.provider).toBe('groq');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(run).not.toHaveBeenCalled();
  });

  it('reports the planned provider in client routing info', () => {
    const { llm } = setup('ok');
    expect(llm.getRoutingInfo(requestOf(150_000)).provider).toBe('cloudflare');
    expect(() => llm.getRoutingInfo(requestOf(1_100_000))).toThrow(NoRouteError);
  });
});

describe('fallback rules and context windows', () => {
  function setupWithRule(rule: { condition: 'error'; fallbackProvider: 'groq'; fallbackModel?: string }) {
    const run = vi.fn().mockRejectedValue(new Error('binding exploded'));
    const fetchMock = vi.fn().mockImplementation(async () => groqResponse());
    vi.stubGlobal('fetch', fetchMock);
    const llm = new LLMProviders({
      cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 },
      groq: { apiKey: 'test', maxRetries: 0 },
      stateScope: 'client',
      fallbackRules: [rule],
    });
    return { llm, fetchMock };
  }

  it('does not let a rule without a model reach a provider whose model cannot hold the input', async () => {
    const { llm, fetchMock } = setupWithRule({ condition: 'error', fallbackProvider: 'groq' });
    await expect(llm.generateResponse(requestOf(150_000))).rejects.toThrow('binding exploded');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('honours a rule that names its model, like a pin', async () => {
    const { llm, fetchMock } = setupWithRule({ condition: 'error', fallbackProvider: 'groq', fallbackModel: 'llama-3.3-70b-versatile' });
    const response = await llm.generateResponse(requestOf(150_000));
    expect(response.provider).toBe('groq');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
