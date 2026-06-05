import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  canonicalToLLMRequest,
  normalizeLLMRequest,
  normalizeLLMResponse,
  type CanonicalLLMRequest,
} from '../canonical';
import { OpenAIProvider } from '../providers/openai';
import { AnthropicProvider } from '../providers/anthropic';
import { GroqProvider } from '../providers/groq';
import { CerebrasProvider } from '../providers/cerebras';
import { NvidiaProvider } from '../providers/nvidia';
import { CloudflareProvider } from '../providers/cloudflare';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import type { LLMRequest } from '../types';

const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

const canonicalFixture: CanonicalLLMRequest = {
  system: 'Return terse answers.',
  messages: [{ role: 'user', content: 'Return JSON for status ok.' }],
  sampling: {
    temperature: 0,
    maxTokens: 64,
    seed: 7,
  },
  tools: [{
    type: 'function',
    function: {
      name: 'record_status',
      description: 'Record status.',
      parameters: {
        type: 'object',
        properties: {
          status: { type: 'string' },
        },
        required: ['status'],
      },
    },
  }],
  toolMode: { toolName: 'record_status' },
  output: { kind: 'json_object' },
  workload: 'TOOL_CALLING',
  requirements: {
    toolCalling: true,
    structuredOutput: true,
  },
  metadata: {
    requestId: 'req-canonical',
    tenantId: 'tenant-a',
    custom: { trace: 'contract-fixture' },
  },
};

function openAICompatibleResponse(model: string, providerContent = '{"ok":true}') {
  return {
    ok: true,
    json: async () => ({
      id: `chatcmpl-${model}`,
      object: 'chat.completion',
      created: 1700000000,
      model,
      choices: [{
        index: 0,
        message: { role: 'assistant', content: providerContent },
        finish_reason: 'stop',
      }],
      usage: { prompt_tokens: 10, completion_tokens: 4, total_tokens: 14 },
    }),
    headers: new Headers({ 'content-type': 'application/json' }),
  };
}

describe('canonical provider contract', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
  });

  it('normalizes legacy response_format into canonical output', () => {
    const request: LLMRequest = {
      messages: [{ role: 'user', content: 'JSON please' }],
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'Status',
          schema: {
            type: 'object',
            properties: { ok: { type: 'boolean' } },
            required: ['ok'],
          },
          strict: true,
        },
      },
    };

    const canonical = normalizeLLMRequest(request);

    expect(canonical.output).toEqual({
      kind: 'json_schema',
      schemaName: 'Status',
      schema: request.response_format?.type === 'json_schema'
        ? request.response_format.json_schema.schema
        : undefined,
      strict: true,
    });
    expect(Object.keys(canonical)).not.toContain('response_format');
  });

  it('normalizes legacy toolChoice into canonical toolMode', () => {
    const canonical = normalizeLLMRequest({
      messages: [{ role: 'user', content: 'Call the tool' }],
      tools: canonicalFixture.tools,
      toolChoice: { type: 'function', function: { name: 'record_status' } },
    });

    expect(canonical.toolMode).toEqual({ toolName: 'record_status' });
    expect(Object.keys(canonical)).not.toContain('toolChoice');
  });

  it('moves legacy provider-specific knobs into providerOptions', () => {
    const canonical = normalizeLLMRequest({
      messages: [{ role: 'user', content: 'Hello' }],
      lora: 'adapter-1',
      topP: 0.9,
      frequencyPenalty: 0.2,
      reasoning: { effort: 'high', format: 'parsed', clearThinking: false },
      prediction: 'known prefix',
    });

    expect(canonical.providerOptions).toEqual({
      cloudflare: {
        lora: 'adapter-1',
        topP: 0.9,
        frequencyPenalty: 0.2,
      },
      cerebras: {
        reasoning: { effort: 'high', format: 'parsed', clearThinking: false },
        prediction: 'known prefix',
      },
    });
    expect(Object.keys(canonical)).not.toContain('lora');
    expect(Object.keys(canonical)).not.toContain('prediction');
  });

  it('normalizes legacy images into canonical media parts', () => {
    const canonical = normalizeLLMRequest({
      messages: [{ role: 'user', content: 'Describe this' }],
      images: [{ data: 'QUJD', mimeType: 'image/png' }],
    });

    expect(canonical.media).toEqual([{ type: 'image', data: 'QUJD', mimeType: 'image/png' }]);
    expect(canonical.requirements?.vision).toBe(true);
    expect(Object.keys(canonical)).not.toContain('images');
  });

  it('converts canonical requests back to compatibility LLMRequest input', () => {
    const legacy = canonicalToLLMRequest({
      ...canonicalFixture,
      providerOptions: {
        cloudflare: { lora: 'adapter-1', topP: 0.8, frequencyPenalty: 0.1 },
        cerebras: { reasoning: { effort: 'medium' }, prediction: 'expected' },
      },
    });

    expect(legacy.response_format).toEqual({ type: 'json_object' });
    expect(legacy.toolChoice).toEqual({ type: 'function', function: { name: 'record_status' } });
    expect(legacy.lora).toBe('adapter-1');
    expect(legacy.topP).toBe(0.8);
    expect(legacy.frequencyPenalty).toBe(0.1);
    expect(legacy.reasoning).toEqual({ effort: 'medium' });
    expect(legacy.prediction).toBe('expected');
    expect(legacy.metadata).toMatchObject({
      trace: 'contract-fixture',
      useCase: 'TOOL_CALLING',
    });
  });

  it('adds stable canonical response routing metadata', () => {
    const canonical = normalizeLLMResponse({
      id: 'res-1',
      message: 'ok',
      model: 'llama-3.3-70b-versatile',
      provider: 'groq',
      usage: { inputTokens: 10, outputTokens: 2, totalTokens: 12, cost: 0.001 },
      responseTime: 42,
      finishReason: 'stop',
      metadata: { providerRaw: true },
    }, {
      routing: {
        selectedProvider: 'groq',
        selectedModel: 'llama-3.3-70b-versatile',
        selectionReason: 'TOOL_CALLING',
        fallbackChain: [{ provider: 'cerebras', model: 'openai/gpt-oss-120b', reason: 'rate_limit' }],
        degradations: [{
          capability: 'structuredOutput',
          action: 'emulated',
          reason: 'provider uses prompt-level JSON instruction',
        }],
      },
    });

    expect(canonical.routing).toMatchObject({
      selectedProvider: 'groq',
      selectedModel: 'llama-3.3-70b-versatile',
      selectionReason: 'TOOL_CALLING',
    });
    expect(canonical.routing?.fallbackChain).toHaveLength(1);
    expect(canonical.routing?.degradations?.[0].action).toBe('emulated');
    expect(canonical.metadata).toEqual({ providerRaw: true });
  });

  it('prepares one canonical fixture for OpenAI-compatible, Anthropic-compatible, Groq/Cerebras, NVIDIA, and Cloudflare adapters', async () => {
    const openai = new OpenAIProvider({ apiKey: 'test-key' });
    mockFetch.mockResolvedValueOnce(openAICompatibleResponse('gpt-4o-mini'));
    await expect(openai.generateResponse(canonicalToLLMRequest({
      ...canonicalFixture,
      model: 'gpt-4o-mini',
    }))).resolves.toMatchObject({ provider: 'openai', model: 'gpt-4o-mini' });
    expect(JSON.parse(mockFetch.mock.calls[0][1].body as string).response_format).toEqual({ type: 'json_object' });

    const anthropic = new AnthropicProvider({ apiKey: 'test-key' });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        id: 'msg-test',
        type: 'message',
        role: 'assistant',
        content: [{ type: 'text', text: '"ok":true}' }],
        model: 'claude-3-5-haiku-20241022',
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 3 },
      }),
      headers: new Headers({ 'content-type': 'application/json' }),
    });
    await expect(anthropic.generateResponse(canonicalToLLMRequest({
      ...canonicalFixture,
      model: 'claude-3-5-haiku-20241022',
    }))).resolves.toMatchObject({ provider: 'anthropic', model: 'claude-3-5-haiku-20241022' });
    const anthropicBody = JSON.parse(mockFetch.mock.calls[1][1].body as string);
    expect(anthropicBody.system).toContain('Return terse answers.');
    expect(anthropicBody.messages.at(-1)).toEqual({ role: 'assistant', content: '{' });

    const groq = new GroqProvider({ apiKey: 'test-key' });
    mockFetch.mockResolvedValueOnce(openAICompatibleResponse('llama-3.3-70b-versatile'));
    await expect(groq.generateResponse(canonicalToLLMRequest({
      ...canonicalFixture,
      model: 'llama-3.3-70b-versatile',
    }))).resolves.toMatchObject({ provider: 'groq', model: 'llama-3.3-70b-versatile' });

    const cerebras = new CerebrasProvider({ apiKey: 'test-key' });
    mockFetch.mockResolvedValueOnce(openAICompatibleResponse('openai/gpt-oss-120b'));
    await expect(cerebras.generateResponse(canonicalToLLMRequest({
      ...canonicalFixture,
      model: 'openai/gpt-oss-120b',
    }))).resolves.toMatchObject({ provider: 'cerebras', model: 'openai/gpt-oss-120b' });

    const nvidia = new NvidiaProvider({ apiKey: 'test-key' });
    mockFetch.mockResolvedValueOnce(openAICompatibleResponse('meta/llama-3.3-70b-instruct'));
    await expect(nvidia.generateResponse(canonicalToLLMRequest({
      ...canonicalFixture,
      model: 'meta/llama-3.3-70b-instruct',
    }))).resolves.toMatchObject({ provider: 'nvidia', model: 'meta/llama-3.3-70b-instruct' });

    const mockAiRun = vi.fn().mockResolvedValueOnce({
      choices: [{ message: { role: 'assistant', content: '{"ok":true}' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 10, completion_tokens: 4, total_tokens: 14 },
    });
    const cloudflare = new CloudflareProvider({ ai: { run: mockAiRun } as unknown as Ai });
    await expect(cloudflare.generateResponse(canonicalToLLMRequest({
      ...canonicalFixture,
      model: '@cf/openai/gpt-oss-120b',
    }))).resolves.toMatchObject({ provider: 'cloudflare', model: '@cf/openai/gpt-oss-120b' });
    expect(mockAiRun.mock.calls[0][1].tools).toEqual(canonicalFixture.tools);
  });
});
