/**
 * #136 slice: reasoning control on Workers AI (#135), length-cause and reasoning-token
 * accounting, pinned-model integrity (#128) and served provider/model on streams (#133).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CloudflareProvider } from '../providers/cloudflare';
import {
  cloudflareReasoningEffortCapability,
  resolveCloudflareReasoningEffort,
} from '../providers/cloudflare-reasoning';
import { getCatalogEntry } from '../model-catalog';
import {
  InvalidRequestError, LLMProviders, NoRouteError, ReasoningEffortUnsupportedError, ResilienceContext, TimeoutError,
  normalizeLLMResponse,
} from '../index';
import { isCachedResponse } from '../utils/response-cache';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { defaultExhaustionRegistry } from '../utils/exhaustion';
import { lengthCauseFor } from '../utils/reasoning';
import type { LLMRequest } from '../types';

class TestableCloudflareProvider extends CloudflareProvider {
  exposeModelCapabilities() {
    return this.getModelCapabilities();
  }
}

const messages: LLMRequest['messages'] = [{ role: 'user', content: 'Extract the entities.' }];

function chatResult(content: string | null, extra: Record<string, unknown> = {}, finishReason = 'stop') {
  return {
    id: 'chatcmpl-test',
    choices: [{ index: 0, message: { role: 'assistant', content, ...extra }, finish_reason: finishReason }],
    usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
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

async function drain(stream: ReadableStream<string>) {
  const reader = stream.getReader();
  let text = '';
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) return text;
    text += chunk.value;
  }
}

beforeEach(() => {
  defaultCircuitBreakerManager.resetAll();
  defaultExhaustionRegistry.reset();
  vi.spyOn(console, 'log').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(console, 'error').mockImplementation(() => {});
});
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe('Workers AI reasoning effort (#135)', () => {
  let run: ReturnType<typeof vi.fn>;
  let provider: TestableCloudflareProvider;

  beforeEach(() => {
    run = vi.fn().mockResolvedValue(chatResult('{"ok":true}'));
    provider = new TestableCloudflareProvider({ ai: { run } as unknown as Ai, maxRetries: 0 });
  });

  const sent = () => run.mock.calls[0][1] as Record<string, unknown>;

  it('forwards an exact level as reasoning_effort and reports it', async () => {
    const response = await provider.generateResponse({
      model: '@cf/openai/gpt-oss-20b', messages, reasoning: { effort: 'low' },
    });
    expect(sent().reasoning_effort).toBe('low');
    expect(sent()).not.toHaveProperty('chat_template_kwargs');
    expect(response.reasoningEffort).toEqual({ requested: 'low', applied: 'low', exact: true });
  });

  it('maps to the nearest documented level and never sends an alias that means max', async () => {
    const medium = await provider.generateResponse({
      model: '@cf/zai-org/glm-5.3-flash', messages, reasoning: { effort: 'medium' },
    });
    expect(sent().reasoning_effort).toBe('high');
    expect(medium.reasoningEffort).toEqual({ requested: 'medium', applied: 'high', exact: false });

    run.mockClear();
    // GLM-5.3 Flash documents `none` as an alias for max; it must not be forwarded.
    const none = await provider.generateResponse({
      model: '@cf/zai-org/glm-5.3-flash', messages, reasoning: { effort: 'none' },
    });
    expect(sent()).not.toHaveProperty('reasoning_effort');
    expect(none.reasoningEffort).toEqual({ requested: 'none', applied: null, exact: false });
  });

  it('turns reasoning off where the model documents it', async () => {
    await provider.generateResponse({ model: '@cf/moonshotai/kimi-k2.6', messages, reasoning: { effort: 'none' } });
    expect(sent().reasoning_effort).toBe('none');
  });

  it('uses chat_template_kwargs for Nemotron', async () => {
    const off = await provider.generateResponse({
      model: '@cf/nvidia/nemotron-3-120b-a12b', messages, reasoning: { effort: 'none' },
    });
    expect(sent().chat_template_kwargs).toEqual({ enable_thinking: false });
    expect(sent()).not.toHaveProperty('reasoning_effort');
    expect(off.reasoningEffort).toMatchObject({ applied: 'enable_thinking=false', exact: true });

    run.mockClear();
    await provider.generateResponse({ model: '@cf/nvidia/nemotron-3-120b-a12b', messages, reasoning: { effort: 'low' } });
    expect(sent().chat_template_kwargs).toEqual({ enable_thinking: true, low_effort: true });

    run.mockClear();
    const high = await provider.generateResponse({
      model: '@cf/nvidia/nemotron-3-120b-a12b', messages, reasoning: { effort: 'high' },
    });
    // Normal reasoning is sent explicitly rather than relying on an undocumented default.
    expect(sent().chat_template_kwargs).toEqual({ enable_thinking: true });
    expect(sent()).not.toHaveProperty('reasoning_effort');
    expect(high.reasoningEffort).toEqual({ requested: 'high', applied: 'enable_thinking=true', exact: false });
    await expect(provider.generateResponse({
      model: '@cf/nvidia/nemotron-3-120b-a12b', messages, reasoning: { effort: 'medium', strict: true },
    })).rejects.toBeInstanceOf(ReasoningEffortUnsupportedError);
  });

  it('sends nothing and reports not-applied for a model without a mapping', async () => {
    const response = await provider.generateResponse({
      model: '@cf/meta/llama-3.3-70b-instruct-fp8-fast', messages, reasoning: { effort: 'high' },
    });
    expect(sent()).not.toHaveProperty('reasoning_effort');
    expect(response.reasoningEffort).toEqual({ requested: 'high', applied: null, exact: false });
  });

  it('adds nothing when no effort is requested', async () => {
    const response = await provider.generateResponse({ model: '@cf/openai/gpt-oss-20b', messages });
    expect(sent()).not.toHaveProperty('reasoning_effort');
    expect(response).not.toHaveProperty('reasoningEffort');
  });

  it('rejects inexact or unsupported levels in strict mode before calling the binding', async () => {
    await expect(provider.generateResponse({
      model: '@cf/openai/gpt-oss-20b', messages, reasoning: { effort: 'none', strict: true },
    })).rejects.toBeInstanceOf(InvalidRequestError);
    await expect(provider.generateResponse({
      model: '@cf/zai-org/glm-5.3-flash', messages, reasoning: { effort: 'medium', strict: true },
    })).rejects.toThrow(/nearest: 'high'/);
    await expect(provider.generateResponse({
      model: '@cf/meta/llama-3.3-70b-instruct-fp8-fast', messages, reasoning: { effort: 'low', strict: true },
    })).rejects.toBeInstanceOf(InvalidRequestError);
    expect(run).not.toHaveBeenCalled();

    await provider.generateResponse({ model: '@cf/openai/gpt-oss-20b', messages, reasoning: { effort: 'high', strict: true } });
    expect(sent().reasoning_effort).toBe('high');
  });

  it('applies effort on the streaming path too', async () => {
    run.mockResolvedValue(new ReadableStream({ start(controller) {
      controller.enqueue({ response: 'ok' }); controller.close();
    } }));
    const stream = await provider.streamResponse({
      model: '@cf/deepseek-ai/deepseek-v4-flash-0731', messages, reasoning: { effort: 'low' },
    });
    expect(await drain(stream)).toBe('ok');
    expect(sent()).toMatchObject({ stream: true, reasoning_effort: 'low' });
  });

  it('ignores prototype keys as effort values', () => {
    expect(resolveCloudflareReasoningEffort('@cf/openai/gpt-oss-20b', 'constructor' as never)?.report)
      .toEqual({ requested: 'constructor', applied: null, exact: false });
    expect(cloudflareReasoningEffortCapability('constructor')).toBeUndefined();
  });

  it('exposes the same capability through the provider and the catalog, for real models only', () => {
    const capabilities = provider.exposeModelCapabilities();
    const mapped = provider.models.filter(model => cloudflareReasoningEffortCapability(model));
    expect(mapped.sort()).toEqual([
      '@cf/deepseek-ai/deepseek-v4-flash-0731',
      '@cf/deepseek-ai/deepseek-v4-pro-0813',
      '@cf/moonshotai/kimi-k2.6',
      '@cf/nvidia/nemotron-3-120b-a12b',
      '@cf/openai/gpt-oss-120b',
      '@cf/openai/gpt-oss-20b',
      '@cf/qwen/qwen3.8-27b',
      '@cf/zai-org/glm-5.2',
      '@cf/zai-org/glm-5.3',
      '@cf/zai-org/glm-5.3-flash',
    ]);
    for (const model of mapped) {
      const capability = cloudflareReasoningEffortCapability(model)!;
      expect(capabilities[model].reasoningEffort).toEqual(capability);
      expect(getCatalogEntry(model)?.capabilities.reasoningEffort).toEqual(capability);
      expect(capability.source).toMatch(/^https:\/\/developers\.cloudflare\.com\/workers-ai\/models\//);
    }
    expect(cloudflareReasoningEffortCapability('@cf/openai/gpt-oss-20b')?.exact).toEqual(['low', 'medium', 'high']);
    expect(cloudflareReasoningEffortCapability('@cf/zai-org/glm-5.3-flash')?.exact).toEqual(['low', 'high']);
    expect(capabilities['@cf/meta/llama-3.3-70b-instruct-fp8-fast'].reasoningEffort).toBeUndefined();
  });
});

describe('reasoning accounting (#136 need 1)', () => {
  it('reports provider reasoning tokens separately', async () => {
    const run = vi.fn().mockResolvedValue({
      ...chatResult('answer'),
      usage: { prompt_tokens: 10, completion_tokens: 50, total_tokens: 60, completion_tokens_details: { reasoning_tokens: 42 } },
    });
    const provider = new CloudflareProvider({ ai: { run } as unknown as Ai, maxRetries: 0 });
    const response = await provider.generateResponse({ model: '@cf/openai/gpt-oss-20b', messages });
    expect(response.usage).toMatchObject({ outputTokens: 50, reasoningTokens: 42 });
  });

  it('ignores malformed reasoning token counts', async () => {
    const run = vi.fn().mockResolvedValue({
      ...chatResult('answer'),
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2, completion_tokens_details: { reasoning_tokens: '7' } },
    });
    const provider = new CloudflareProvider({ ai: { run } as unknown as Ai, maxRetries: 0 });
    const response = await provider.generateResponse({ model: '@cf/openai/gpt-oss-20b', messages });
    expect(response.usage).not.toHaveProperty('reasoningTokens');
  });

  it('marks a length stop with no answer as a reasoning cut-off', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('', { reasoning: 'thinking thinking thinking' }, 'length'));
    const provider = new CloudflareProvider({ ai: { run } as unknown as Ai, maxRetries: 0 });
    const response = await provider.generateResponse({ model: '@cf/openai/gpt-oss-20b', messages });
    expect(response).toMatchObject({ finishReason: 'length', message: '', lengthCause: 'reasoning' });
  });

  it('marks a length stop mid-answer as an answer cut-off, and leaves other stops alone', async () => {
    const run = vi.fn()
      .mockResolvedValueOnce(chatResult('{"partial": ', { reasoning: 'done' }, 'length'))
      .mockResolvedValueOnce(chatResult('complete'));
    const provider = new CloudflareProvider({ ai: { run } as unknown as Ai, maxRetries: 0 });
    expect((await provider.generateResponse({ model: '@cf/openai/gpt-oss-20b', messages })).lengthCause).toBe('answer');
    expect(await provider.generateResponse({ model: '@cf/openai/gpt-oss-20b', messages })).not.toHaveProperty('lengthCause');
  });

  it('classifies an unclosed think block as reasoning', () => {
    expect(lengthCauseFor('length', '<think>still going', undefined)).toBe('reasoning');
    expect(lengthCauseFor('length', '<think>a</think> answer', undefined)).toBe('answer');
    expect(lengthCauseFor('length', '', undefined)).toBeUndefined();
    expect(lengthCauseFor('stop', '', 'r')).toBeUndefined();
  });

  it('classifies leading think structure, and template-opened think blocks', () => {
    // Unclosed block right after earlier closed ones, before any answer.
    expect(lengthCauseFor('length', '<think>a</think>\n<think>more', undefined)).toBe('reasoning');
    // Tags after answer text are answer content, not reasoning.
    expect(lengthCauseFor('length', '<think>a</think> b <think>more', undefined)).toBe('answer');
    expect(lengthCauseFor('length', "Here's the parser: const OPEN = '<think>';", undefined)).toBe('answer');
    expect(lengthCauseFor('length', "Close it with '</think>' and then", undefined)).toBe('answer');
    // An answer cut off right after a literal closing tag, from a model that doesn't think.
    expect(lengthCauseFor('length', 'The closing tag is </think>', undefined)).toBe('answer');
    // Template-injected opener (thinking models only): only a closing tag appears in the output.
    const thinking = { thinkingModel: true };
    expect(lengthCauseFor('length', 'reasoning text</think>', undefined, thinking)).toBe('reasoning');
    expect(lengthCauseFor('length', 'reasoning text</think> partial answer', undefined, thinking)).toBe('answer');
    // Reasoning that quotes '<think>' before its closing tag is still one template-opened block.
    expect(lengthCauseFor('length', 'user asks about <think> tags </think> partial answer', undefined, thinking))
      .toBe('answer');
    expect(lengthCauseFor('length', '<think>x</think>', undefined)).toBe('reasoning');
    expect(lengthCauseFor('length', '<think>x</think>', undefined, thinking)).toBe('reasoning');
    // No tags at all from a model whose template opens the block: still reasoning.
    expect(lengthCauseFor('length', 'Let me think step by step', undefined, { thinkingModel: true })).toBe('reasoning');
    expect(lengthCauseFor('length', 'Let me think step by step', undefined)).toBe('answer');
    // A separate reasoning field means the message text is the answer.
    expect(lengthCauseFor('length', 'partial answer', 'thoughts', { thinkingModel: true })).toBe('answer');
  });

  it('marks an untagged truncated QwQ output as a reasoning cut-off', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('Okay, first consider the', {}, 'length'));
    const provider = new CloudflareProvider({ ai: { run } as unknown as Ai, maxRetries: 0 });
    const response = await provider.generateResponse({ model: '@cf/qwen/qwq-32b', messages });
    expect(response.lengthCause).toBe('reasoning');
  });
});

describe('pinned models and fallback (#128, #133)', () => {
  const pinned = '@cf/zai-org/glm-5.3-flash';

  function setup(cloudflareError: Error | null, fallbackRules?: []) {
    const run = cloudflareError
      ? vi.fn().mockRejectedValue(cloudflareError)
      : vi.fn().mockResolvedValue(chatResult('from cloudflare'));
    const fetchMock = vi.fn().mockImplementation(async () => groqResponse());
    vi.stubGlobal('fetch', fetchMock);
    const llm = new LLMProviders({
      cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 },
      groq: { apiKey: 'test', maxRetries: 0 },
      stateScope: 'client',
      ...(fallbackRules ? { fallbackRules } : {}),
    });
    return { llm, run, fetchMock };
  }

  it('marks a substitution on the response', async () => {
    const { llm } = setup(new Error('binding exploded'));
    const response = await llm.generateResponse({ model: pinned, messages, maxTokens: 10 });
    expect(response.provider).toBe('groq');
    expect(response.fallback).toEqual({
      requestedProvider: 'cloudflare',
      requestedModel: pinned,
      failedAttempts: [{ provider: 'cloudflare', model: pinned, errorCode: 'ERROR' }],
    });
    expect(normalizeLLMResponse(response).routing?.fallbackChain).toEqual([
      { provider: 'cloudflare', model: pinned, reason: 'ERROR' },
    ]);
  });

  it('adds no marker when the planned provider serves the request', async () => {
    const { llm } = setup(null);
    const response = await llm.generateResponse({ model: pinned, messages, maxTokens: 10 });
    expect(response.provider).toBe('cloudflare');
    expect(response).not.toHaveProperty('fallback');
  });

  it('fallbackRules: [] does not stop built-in fallback for timeouts, but allowFallback: false does', async () => {
    const loose = setup(new TimeoutError('cloudflare', 'Request timeout after 10ms'), []);
    const served = await loose.llm.generateResponse({ model: pinned, messages, maxTokens: 10 });
    expect(served.provider).toBe('groq');
    expect(served.fallback?.failedAttempts[0].errorCode).toBe('TIMEOUT');

    const strict = setup(new TimeoutError('cloudflare', 'Request timeout after 10ms'));
    await expect(strict.llm.generateResponse({ model: pinned, messages, maxTokens: 10, allowFallback: false }))
      .rejects.toBeInstanceOf(TimeoutError);
    expect(strict.fetchMock).not.toHaveBeenCalled();
  });

  it('allowFallback: false overrides an any-error fallback rule', async () => {
    const { llm, fetchMock } = setup(new Error('binding exploded'));
    await expect(llm.generateResponse({ model: pinned, messages, maxTokens: 10, allowFallback: false }))
      .rejects.toThrow('binding exploded');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('throws NoRouteError when the pinned model has no configured provider', async () => {
    const { llm, run, fetchMock } = setup(null);
    await expect(llm.generateResponse({ model: 'gpt-4o-mini', messages, allowFallback: false }))
      .rejects.toBeInstanceOf(NoRouteError);
    expect(run).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('marks a pinned provider that was skipped as quota-exhausted', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('unused'));
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async () => groqResponse()));
    const resilienceContext = new ResilienceContext();
    resilienceContext.exhaustion.markExhausted('cloudflare');
    const llm = new LLMProviders({
      cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 },
      groq: { apiKey: 'test', maxRetries: 0 },
      resilienceContext,
    });
    const response = await llm.generateResponse({ model: pinned, messages, maxTokens: 10 });
    expect(response.provider).toBe('groq');
    expect(response.fallback?.failedAttempts).toEqual([{ provider: 'cloudflare', model: pinned, errorCode: 'QUOTA_EXHAUSTED' }]);
    expect(run).not.toHaveBeenCalled();
  });

  it('reports the served provider and model on stream completion', async () => {
    const run = vi.fn().mockResolvedValue(new ReadableStream({ start(controller) {
      controller.enqueue({ response: 'streamed' }); controller.close();
    } }));
    const llm = new LLMProviders({ cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 }, stateScope: 'client' });
    const result = await llm.generateManagedResponseStream({ model: pinned, messages, maxTokens: 10 });
    expect(await drain(result.stream)).toBe('streamed');
    const completion = await result.completion;
    expect(completion).toMatchObject({ outcome: 'success', provider: 'cloudflare', model: pinned });
    expect(completion).not.toHaveProperty('fallback');
  });

  it('marks a streamed substitution, and refuses one when allowFallback is false', async () => {
    const encoder = new TextEncoder();
    const sse = () => new Response(new ReadableStream({ start(controller) {
      controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"groq"}}]}\n\ndata: [DONE]\n\n'));
      controller.close();
    } }), { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
    const fetchMock = vi.fn().mockImplementation(async () => sse());
    vi.stubGlobal('fetch', fetchMock);
    const run = vi.fn().mockRejectedValue(new Error('binding exploded'));
    const llm = new LLMProviders({
      cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 },
      groq: { apiKey: 'test', maxRetries: 0 },
      stateScope: 'client',
    });

    const result = await llm.generateManagedResponseStream({ model: pinned, messages, maxTokens: 10 });
    expect(await drain(result.stream)).toBe('groq');
    const completion = await result.completion;
    expect(completion.provider).toBe('groq');
    expect(completion.fallback).toMatchObject({ requestedProvider: 'cloudflare', requestedModel: pinned });

    fetchMock.mockClear();
    // After one failure the breaker's degradation curve sheds a random share of
    // traffic; pin the draw so the second call reaches the binding.
    vi.spyOn(Math, 'random').mockReturnValue(0);
    await expect(llm.generateManagedResponseStream({ model: pinned, messages, maxTokens: 10, allowFallback: false }))
      .rejects.toThrow('binding exploded');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('getRoutingInfo honors allowFallback: false for an unservable pin', () => {
    const { llm } = setup(null);
    expect(() => llm.getRoutingInfo({ model: 'gpt-4o-mini', messages, allowFallback: false })).toThrow(NoRouteError);
  });
});

describe('strict reasoning through the factory (#135)', () => {
  const flash = '@cf/zai-org/glm-5.3-flash';

  function setup(run: ReturnType<typeof vi.fn>, extra: Record<string, unknown> = {}) {
    const fetchMock = vi.fn().mockImplementation(async () => groqResponse());
    vi.stubGlobal('fetch', fetchMock);
    const llm = new LLMProviders({
      cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 },
      groq: { apiKey: 'test', maxRetries: 0 },
      stateScope: 'client',
      ...extra,
    });
    return { llm, fetchMock };
  }

  it('does not fall back on a strict rejection under the default any-error rule', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('unused'));
    const { llm, fetchMock } = setup(run);
    const request = { model: flash, messages, maxTokens: 10, reasoning: { effort: 'medium' as const, strict: true } };

    const rejection = llm.generateResponse(request);
    await expect(rejection).rejects.toBeInstanceOf(ReasoningEffortUnsupportedError);
    await expect(rejection).rejects.toBeInstanceOf(InvalidRequestError);
    await expect(llm.generateManagedResponseStream(request)).rejects.toBeInstanceOf(ReasoningEffortUnsupportedError);
    expect(run).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('rejects an inexact strict effort before dispatch even with a JSON Schema', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('unused'));
    const { llm, fetchMock } = setup(run);
    await expect(llm.generateResponse({
      model: flash, messages, reasoning: { effort: 'medium', strict: true },
      response_format: { type: 'json_schema', json_schema: { name: 'result', schema: { type: 'object' } } },
    })).rejects.toBeInstanceOf(ReasoningEffortUnsupportedError);
    expect(run).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('does not fall back on an unsupported schema when fallback is disabled', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('unused'));
    const { llm, fetchMock } = setup(run);
    await expect(llm.generateResponse({
      model: '@cf/meta/llama-3.1-8b-instruct-fp8', messages, allowFallback: false,
      response_format: { type: 'json_schema', json_schema: { name: 'result', schema: { type: 'object' } } },
    })).rejects.toMatchObject({ code: 'STRUCTURED_OUTPUT_UNSUPPORTED' });
    expect(run).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('falls back on the real Cloudflare streaming schema rejection', async () => {
    const run = vi.fn();
    const { llm, fetchMock } = setup(run);
    const result = await llm.generateManagedResponseStream({ model: flash, messages,
      response_format: { type: 'json_schema', json_schema: { name: 'result', schema: { type: 'object' } } } });
    await drain(result.stream);
    expect((await result.completion).provider).toBe('groq');
    expect(run).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalled();
  });

  it('does not fail over to a provider that ignores the level when the strict call errors', async () => {
    const run = vi.fn().mockRejectedValue(new Error('binding exploded'));
    const { llm, fetchMock } = setup(run, {
      fallbackRules: [{ condition: 'error', fallbackProvider: 'groq' }],
    });
    await expect(llm.generateResponse({
      model: flash, messages, maxTokens: 10, reasoning: { effort: 'low', strict: true },
    })).rejects.toThrow('binding exploded');
    expect(run).toHaveBeenCalledTimes(1);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('rejects a strict request pinned to a provider that does not enforce it, before any call', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('unused'));
    const { llm, fetchMock } = setup(run);
    const request = { model: 'llama-3.3-70b-versatile', messages, maxTokens: 10, reasoning: { effort: 'high' as const, strict: true } };
    await expect(llm.generateResponse(request)).rejects.toBeInstanceOf(ReasoningEffortUnsupportedError);
    await expect(llm.generateManagedResponseStream(request)).rejects.toBeInstanceOf(ReasoningEffortUnsupportedError);
    expect(() => llm.getRoutingInfo(request)).toThrow(ReasoningEffortUnsupportedError);
    expect(run).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('rejects a strict pin that cannot be served instead of substituting a Cloudflare default', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('unused'));
    const { llm, fetchMock } = setup(run);
    const strict = { effort: 'high' as const, strict: true };
    const unservable: LLMRequest[] = [
      // Provider not configured.
      { model: 'gpt-4o-mini', messages, maxTokens: 10, reasoning: strict },
      // Unknown model string.
      { model: 'my-custom-model', messages, maxTokens: 10, reasoning: strict },
      // Groq pin dropped because Groq can't take images.
      { model: 'llama-3.3-70b-versatile', messages, maxTokens: 10, reasoning: strict,
        images: [{ data: 'QUJD', mimeType: 'image/png' }] },
    ];
    for (const request of unservable) {
      await expect(llm.generateResponse(request)).rejects.toBeInstanceOf(ReasoningEffortUnsupportedError);
      await expect(llm.generateManagedResponseStream(request)).rejects.toBeInstanceOf(ReasoningEffortUnsupportedError);
      expect(() => llm.getRoutingInfo(request)).toThrow(ReasoningEffortUnsupportedError);
    }
    expect(run).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('rejects a strict request when no Cloudflare provider is configured', async () => {
    const fetchMock = vi.fn().mockImplementation(async () => groqResponse());
    vi.stubGlobal('fetch', fetchMock);
    const llm = new LLMProviders({ groq: { apiKey: 'test', maxRetries: 0 }, stateScope: 'client' });
    await expect(llm.generateResponse({ messages, maxTokens: 10, reasoning: { effort: 'low', strict: true } }))
      .rejects.toThrow('no Cloudflare provider is configured');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('routes an unpinned strict request to Cloudflare even when another provider is preferred', () => {
    const run = vi.fn().mockResolvedValue(chatResult('unused'));
    const { llm } = setup(run, { defaultProvider: 'groq' });
    expect(llm.getRoutingInfo({ messages, reasoning: { effort: 'low', strict: true } }).provider).toBe('cloudflare');
    expect(llm.getRoutingInfo({ messages }).provider).toBe('groq');
  });

  it('keeps strict in the tool loop', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('unused'));
    const { llm, fetchMock } = setup(run);
    await expect(llm.generateResponseWithTools(
      { model: flash, messages, maxTokens: 10, reasoning: { effort: 'medium', strict: true } },
      { execute: vi.fn() } as never,
    )).rejects.toBeInstanceOf(ReasoningEffortUnsupportedError);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('never falls back on a strict rejection, whatever the request says', async () => {
    // Isolates the getFallbackDecision guard: the request is not strict, so the chain is not trimmed.
    const run = vi.fn().mockRejectedValue(new ReasoningEffortUnsupportedError('cloudflare', 'rejected'));
    const { llm, fetchMock } = setup(run);
    await expect(llm.generateResponse({ model: flash, messages, maxTokens: 10 }))
      .rejects.toBeInstanceOf(ReasoningEffortUnsupportedError);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('reports willRetry: false when a strict request has nowhere to go', async () => {
    const run = vi.fn().mockRejectedValue(new Error('binding exploded'));
    const errors: Array<boolean | undefined> = [];
    const { llm } = setup(run, { hooks: { onRequestError: (event: { willRetry?: boolean }) => errors.push(event.willRetry) } });
    await expect(llm.generateResponse({
      model: flash, messages, maxTokens: 10, reasoning: { effort: 'low', strict: true },
    })).rejects.toThrow('binding exploded');
    expect(errors).toEqual([false]);
  });

  it('reports willRetry: false on the stream path too', async () => {
    const run = vi.fn().mockRejectedValue(new Error('binding exploded'));
    const errors: Array<boolean | undefined> = [];
    const { llm, fetchMock } = setup(run, {
      hooks: { onRequestError: (event: { willRetry?: boolean }) => errors.push(event.willRetry) },
    });
    await expect(llm.generateManagedResponseStream({
      model: flash, messages, maxTokens: 10, reasoning: { effort: 'low', strict: true },
    })).rejects.toThrow('binding exploded');
    expect(errors).toEqual([false]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('with allowFallback: false, NoRouteError for an unservable pin wins over the strict check', () => {
    const run = vi.fn().mockResolvedValue(chatResult('unused'));
    const { llm } = setup(run);
    const strict = { effort: 'high' as const, strict: true };
    // No configured provider serves the pin: the allowFallback check runs first.
    expect(() => llm.getRoutingInfo({ model: 'gpt-4o-mini', messages, reasoning: strict, allowFallback: false }))
      .toThrow(NoRouteError);
    // A pin on another configured provider is a strict failure.
    const groqPin = () => llm.getRoutingInfo({
      model: 'llama-3.3-70b-versatile', messages, reasoning: strict, allowFallback: false,
    });
    expect(groqPin).toThrow(ReasoningEffortUnsupportedError);
    expect(groqPin).not.toThrow(NoRouteError);
  });

  it('still falls back when strict is not set', async () => {
    const run = vi.fn().mockRejectedValue(new Error('binding exploded'));
    const { llm } = setup(run);
    const response = await llm.generateResponse({ model: flash, messages, maxTokens: 10, reasoning: { effort: 'medium' } });
    expect(response.provider).toBe('groq');
  });
});

describe('fallback review fixes', () => {
  const pinned = '@cf/zai-org/glm-5.3-flash';

  it('allowFallback: false with a skipped pin throws NoRouteError naming the skip', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('unused'));
    const fetchMock = vi.fn().mockImplementation(async () => groqResponse());
    vi.stubGlobal('fetch', fetchMock);
    const resilienceContext = new ResilienceContext();
    resilienceContext.exhaustion.markExhausted('cloudflare');
    const llm = new LLMProviders({
      cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 },
      groq: { apiKey: 'test', maxRetries: 0 },
      resilienceContext,
    });
    const request = { model: pinned, messages, maxTokens: 10, allowFallback: false };
    const rejection = llm.generateResponse(request);
    await expect(rejection).rejects.toBeInstanceOf(NoRouteError);
    await expect(rejection).rejects.toThrow('cloudflare: QUOTA_EXHAUSTED');
    await expect(llm.generateManagedResponseStream(request)).rejects.toBeInstanceOf(NoRouteError);
    expect(() => llm.getRoutingInfo(request)).toThrow(NoRouteError);
    expect(run).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('names skipped providers when nothing could be attempted', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('unused'));
    const resilienceContext = new ResilienceContext();
    resilienceContext.exhaustion.markExhausted('cloudflare');
    const llm = new LLMProviders({ cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 }, resilienceContext });
    await expect(llm.generateResponse({ model: pinned, messages, maxTokens: 10 }))
      .rejects.toThrow('All providers failed (cloudflare: QUOTA_EXHAUSTED)');
  });

  it('reports the model requested at plan time even if a fallback rule rewrites it', async () => {
    const run = vi.fn().mockRejectedValue(new Error('binding exploded'));
    const groqFailure = new Response(JSON.stringify({ error: { message: 'down' } }), {
      status: 500, headers: { 'Content-Type': 'application/json' },
    });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(groqFailure)
      .mockImplementation(async () => groqResponse());
    vi.stubGlobal('fetch', fetchMock);
    const llm = new LLMProviders({
      cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 },
      groq: { apiKey: 'test', maxRetries: 0 },
      cerebras: { apiKey: 'test', maxRetries: 0 },
      stateScope: 'client',
      fallbackRules: [{ condition: 'error', fallbackProvider: 'cloudflare', fallbackModel: '@cf/meta/llama-3.1-8b-instruct' }],
    });
    const response = await llm.generateResponse({ model: pinned, messages, maxTokens: 10 });
    expect(response.provider).not.toBe('cloudflare');
    expect(response.fallback?.requestedModel).toBe(pinned);
    expect(response.fallback?.failedAttempts[0]).toEqual({ provider: 'cloudflare', model: pinned, errorCode: 'ERROR' });
  });

  it('rejects cached envelopes with malformed reasoning or fallback fields', () => {
    const base = {
      message: 'cached', model: pinned, provider: 'cloudflare', responseTime: 1,
      usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2, cost: 0 },
    };
    expect(isCachedResponse(base)).toBe(true);
    expect(isCachedResponse({
      ...base,
      lengthCause: 'answer',
      reasoningEffort: { requested: 'high', applied: 'high', exact: true },
      fallback: { requestedProvider: 'cloudflare', failedAttempts: [{ provider: 'cloudflare', errorCode: 'ERROR' }] },
    })).toBe(true);
    expect(isCachedResponse({ ...base, fallback: { requestedProvider: 'cloudflare', failedAttempts: 'x' } })).toBe(false);
    expect(isCachedResponse({ ...base, fallback: { requestedProvider: 'cloudflare', failedAttempts: [null] } })).toBe(false);
    expect(isCachedResponse({ ...base, fallback: null })).toBe(false);
    expect(isCachedResponse({ ...base, reasoningEffort: { requested: 'extreme', applied: null, exact: false } })).toBe(false);
    expect(isCachedResponse({ ...base, reasoningEffort: { requested: 'high', applied: 1, exact: true } })).toBe(false);
    expect(isCachedResponse({ ...base, lengthCause: 'other' })).toBe(false);
  });

  it('treats a corrupt cached fallback report as a miss instead of failing normalization', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('fresh'));
    const corrupt = {
      message: 'cached', model: pinned, provider: 'cloudflare', responseTime: 1,
      usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2, cost: 0 },
      fallback: { requestedProvider: 'cloudflare', failedAttempts: 'x' },
    };
    const cache = { get: vi.fn().mockResolvedValue(JSON.stringify(corrupt)), put: vi.fn().mockResolvedValue(undefined) };
    const llm = new LLMProviders({
      cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 }, stateScope: 'client', responseCache: cache,
    });
    const response = await llm.generateResponse({ model: pinned, messages, maxTokens: 10 });
    expect(response.message).toBe('fresh');
    expect(() => normalizeLLMResponse(response)).not.toThrow();
  });

  it('does not cache a response its own read path would reject', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('fresh'));
    const cache = { get: vi.fn().mockResolvedValue(null), put: vi.fn().mockResolvedValue(undefined) };
    const events: string[] = [];
    const llm = new LLMProviders({
      cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 }, stateScope: 'client', responseCache: cache,
      hooks: { onCache: event => events.push(`${event.status}:${event.cache.factory?.error ?? ''}`) },
    });
    // An untyped caller can pass an effort outside the type; the report echoes it.
    const odd = await llm.generateResponse({
      model: pinned, messages, maxTokens: 10, reasoning: { effort: 'max' as never },
    });
    expect(odd.reasoningEffort).toMatchObject({ requested: 'max', applied: null });
    expect(cache.put).not.toHaveBeenCalled();
    expect(events).toContain('bypass:Response envelope is not cacheable');

    await llm.generateResponse({ model: pinned, messages, maxTokens: 10, reasoning: { effort: 'low' } });
    expect(cache.put).toHaveBeenCalledTimes(1);
  });

  it('replays the original fallback report on a cache hit', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('unused'));
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async () => groqResponse()));
    const fallback = {
      requestedProvider: 'cloudflare', requestedModel: pinned,
      failedAttempts: [{ provider: 'cloudflare', model: pinned, errorCode: 'ERROR' }],
    };
    const cached = {
      message: 'from groq', model: 'llama-3.3-70b-versatile', provider: 'groq', responseTime: 1,
      usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2, cost: 0 }, fallback,
    };
    const cache = { get: vi.fn().mockResolvedValue(JSON.stringify(cached)), put: vi.fn().mockResolvedValue(undefined) };
    const llm = new LLMProviders({
      cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 },
      groq: { apiKey: 'test', maxRetries: 0 },
      stateScope: 'client',
      responseCache: cache,
    });
    const response = await llm.generateResponse({ model: pinned, messages, maxTokens: 10 });
    expect(response.cache?.factory?.status).toBe('hit');
    expect(response.fallback).toEqual(fallback);
    expect(run).not.toHaveBeenCalled();
  });
});
