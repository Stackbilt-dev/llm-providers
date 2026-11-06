/**
 * Anthropic tool loops (#124): tool_result blocks carry tool_use_id, a failed
 * tool's error text reaches Claude, and the assistant turn (thinking blocks with
 * signatures, redacted_thinking, text, tool_use) is sent back unmodified.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { AnthropicProvider } from '../providers/anthropic';
import { GroqProvider } from '../providers/groq';
import { CerebrasProvider } from '../providers/cerebras';
import { NvidiaProvider } from '../providers/nvidia';
import { OpenAIProvider } from '../providers/openai';
import { LLMProviderFactory } from '../factory';
import { ConfigurationError } from '../errors';
import { isCachedResponse } from '../utils/response-cache';
import { normalizeLegacyRequest, normalizeLLMResponse } from '../canonical';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { LLMProviders } from '../index';
import type { LLMRequest, LLMResponse } from '../types';

const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

const JSON_HEADERS = new Headers({ 'content-type': 'application/json' });

function anthropicResponse(content: unknown[], stopReason = 'end_turn') {
  return {
    ok: true,
    headers: JSON_HEADERS,
    json: async () => ({
      id: 'msg_1',
      type: 'message',
      role: 'assistant',
      content,
      model: 'claude-opus-5-5',
      stop_reason: stopReason,
      stop_sequence: null,
      usage: { input_tokens: 10, output_tokens: 5 },
    }),
  };
}

/** A Claude 5.x tool-use turn: reasoning, a redacted block, a progress note, the call. */
const TOOL_TURN = [
  { type: 'thinking', thinking: '', signature: 'sig-reasoning-1' },
  { type: 'redacted_thinking', data: 'encrypted-blob' },
  { type: 'text', text: 'Let me look that up.' },
  { type: 'thinking', thinking: 'Checking the record.', signature: 'sig-update-2' },
  { type: 'tool_use', id: 'toolu_01', name: 'lookup', input: { id: 42, tags: ['a', 'b'] } },
];

function serverError() {
  return {
    ok: false,
    status: 500,
    headers: JSON_HEADERS,
    json: async () => ({ error: { type: 'api_error', message: 'Internal server error' } }),
    text: async () => '{"error":{"type":"api_error","message":"Internal server error"}}',
  };
}

function openAiResponse(message: Record<string, unknown>, finishReason = 'stop') {
  return {
    ok: true,
    headers: JSON_HEADERS,
    json: async () => ({
      id: 'chatcmpl_1',
      object: 'chat.completion',
      created: 1700000000,
      model: 'gpt-4o-mini',
      choices: [{ index: 0, message: { role: 'assistant', ...message }, finish_reason: finishReason }],
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    }),
  };
}

function sseBody(events: unknown[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const text = events.map(event => `data: ${JSON.stringify(event)}\n\n`).join('');
  return new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(text));
      controller.close();
    },
  });
}

function requestBody(callIndex: number): { messages: Array<{ role: string; content: unknown }> } {
  const init = mockFetch.mock.calls[callIndex][1] as RequestInit;
  return JSON.parse(init.body as string);
}

const lookupTool = {
  type: 'function' as const,
  function: {
    name: 'lookup',
    description: 'Look up a record',
    parameters: { type: 'object', properties: { id: { type: 'number' } }, required: ['id'] },
  },
};

describe('Anthropic tool loops (#124)', () => {
  let provider: AnthropicProvider;

  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
    provider = new AnthropicProvider({ apiKey: 'test-key', maxRetries: 0 });
  });

  describe('tool_result blocks', () => {
    it('send tool_use_id, not id', async () => {
      mockFetch.mockResolvedValueOnce(anthropicResponse([{ type: 'text', text: 'done' }]));

      await provider.generateResponse({
        model: 'claude-opus-5-5',
        messages: [
          { role: 'user', content: 'Look up 42' },
          { role: 'assistant', content: '', toolCalls: [{ id: 'toolu_01', type: 'function', function: { name: 'lookup', arguments: '{"id":42}' } }] },
          { role: 'user', content: '', toolResults: [{ id: 'toolu_01', output: '{"value":42}' }] },
        ],
      });

      const result = requestBody(0).messages[2];
      expect(result.role).toBe('user');
      expect(result.content).toEqual([
        { type: 'tool_result', tool_use_id: 'toolu_01', content: '{"value":42}', is_error: false },
      ]);
    });

    it('put a failed tool’s error text in content with is_error: true', async () => {
      mockFetch.mockResolvedValueOnce(anthropicResponse([{ type: 'text', text: 'sorry' }]));

      await provider.generateResponse({
        model: 'claude-opus-5-5',
        messages: [
          { role: 'user', content: 'Look up 42' },
          { role: 'assistant', content: '', toolCalls: [
            { id: 'toolu_01', type: 'function', function: { name: 'lookup', arguments: '{}' } },
            { id: 'toolu_02', type: 'function', function: { name: 'lookup', arguments: '{}' } },
          ] },
          { role: 'user', content: '', toolResults: [
            { id: 'toolu_01', output: '', error: 'ConnectionError: service unavailable' },
            { id: 'toolu_02', output: 'partial rows', error: 'Timed out after 5s' },
          ] },
        ],
      });

      expect(requestBody(0).messages[2].content).toEqual([
        { type: 'tool_result', tool_use_id: 'toolu_01', content: 'ConnectionError: service unavailable', is_error: true },
        { type: 'tool_result', tool_use_id: 'toolu_02', content: 'partial rows\n\nTimed out after 5s', is_error: true },
      ]);
    });
  });

  describe('LLMResponse.providerContent', () => {
    it('keeps the whole tool-use turn in order, including signatures and redacted blocks', async () => {
      mockFetch.mockResolvedValueOnce(anthropicResponse(TOOL_TURN, 'tool_use'));

      const response = await provider.generateResponse({
        model: 'claude-opus-5-5',
        messages: [{ role: 'user', content: 'Look up 42' }],
        tools: [lookupTool],
      });

      expect(response.providerContent).toEqual({ provider: 'anthropic', blocks: TOOL_TURN });
      expect(response.reasoning).toBe('Checking the record.');
      expect(response.toolCalls?.[0]).toMatchObject({ id: 'toolu_01', function: { name: 'lookup' } });
      // A copy, not the parsed body: later edits to the response can't leak into it.
      expect(response.providerContent?.blocks).not.toBe(TOOL_TURN);
    });

    it('is not set when a tool_use block fails validation, so the turn is rebuilt', async () => {
      mockFetch.mockResolvedValueOnce(anthropicResponse([
        ...TOOL_TURN,
        { type: 'tool_use', id: 'toolu_02', name: '', input: {} },
      ], 'tool_use'));

      const response = await provider.generateResponse({
        model: 'claude-opus-5-5',
        messages: [{ role: 'user', content: 'Look up 42' }],
        tools: [lookupTool],
      });

      expect(response.toolCalls?.map(call => call.id)).toEqual(['toolu_01']);
      expect(response.providerContent).toBeUndefined();
    });

    it('is not set on a JSON-prefill response, whose text starts after the prefilled brace', async () => {
      mockFetch.mockResolvedValueOnce(anthropicResponse([
        { type: 'text', text: '"id": 42}' },
        { type: 'tool_use', id: 'toolu_01', name: 'lookup', input: { id: 42 } },
      ], 'tool_use'));

      const response = await provider.generateResponse({
        model: 'claude-haiku-4-5-20251001',
        messages: [{ role: 'user', content: 'Look up 42' }],
        tools: [lookupTool],
        response_format: { type: 'json_object' },
      });

      expect(response.message).toBe('{"id": 42}');
      expect(response.toolCalls).toHaveLength(1);
      expect(response.providerContent).toBeUndefined();
    });

    it('is not set when the response calls no tools', async () => {
      mockFetch.mockResolvedValueOnce(anthropicResponse([
        { type: 'thinking', thinking: '', signature: 'sig' },
        { type: 'text', text: 'hello' },
      ]));

      const response = await provider.generateResponse({
        model: 'claude-opus-5-5',
        messages: [{ role: 'user', content: 'hi' }],
      });

      expect(response.providerContent).toBeUndefined();
      expect(response.message).toBe('hello');
    });
  });

  describe('replaying an assistant turn', () => {
    const followUp = (providerContent: LLMRequest['messages'][number]['providerContent']): LLMRequest => ({
      model: 'claude-opus-5-5',
      messages: [
        { role: 'user', content: 'Look up 42' },
        {
          role: 'assistant',
          content: 'Let me look that up.',
          toolCalls: [{ id: 'toolu_01', type: 'function', function: { name: 'lookup', arguments: '{"id":42,"tags":["a","b"]}' } }],
          providerContent,
        },
        { role: 'user', content: '', toolResults: [{ id: 'toolu_01', output: 'found' }] },
      ],
      tools: [lookupTool],
    });

    it('sends Anthropic content back unchanged', async () => {
      mockFetch.mockResolvedValueOnce(anthropicResponse([{ type: 'text', text: 'done' }]));

      await provider.generateResponse(followUp({ provider: 'anthropic', blocks: TOOL_TURN }));

      const assistant = requestBody(0).messages[1];
      expect(assistant.role).toBe('assistant');
      expect(assistant.content).toEqual(TOOL_TURN);
    });

    it('rebuilds the turn, keeping its text, when the content came from another provider', async () => {
      mockFetch.mockResolvedValueOnce(anthropicResponse([{ type: 'text', text: 'done' }]));

      await provider.generateResponse(followUp({ provider: 'openai', blocks: [{ type: 'whatever' }] }));

      expect(requestBody(0).messages[1].content).toEqual([
        { type: 'text', text: 'Let me look that up.' },
        { type: 'tool_use', id: 'toolu_01', name: 'lookup', input: { id: 42, tags: ['a', 'b'] } },
      ]);
    });

    it('rebuilds the turn when its tool_use ids differ from toolCalls', async () => {
      mockFetch.mockResolvedValueOnce(anthropicResponse([{ type: 'text', text: 'done' }]));
      const request = followUp({ provider: 'anthropic', blocks: TOOL_TURN });
      request.messages[1] = {
        ...request.messages[1],
        toolCalls: [{ id: 'toolu_99', type: 'function', function: { name: 'lookup', arguments: '{"id":1}' } }],
      };
      request.messages[2] = { role: 'user', content: '', toolResults: [{ id: 'toolu_99', output: 'found' }] };

      await provider.generateResponse(request);

      expect(requestBody(0).messages[1].content).toEqual([
        { type: 'text', text: 'Let me look that up.' },
        { type: 'tool_use', id: 'toolu_99', name: 'lookup', input: { id: 1 } },
      ]);
    });

    it('drops whitespace-only text when rebuilding a turn', async () => {
      mockFetch.mockResolvedValueOnce(anthropicResponse([{ type: 'text', text: 'done' }]));
      const request = followUp(undefined);
      request.messages[1] = { ...request.messages[1], content: '\n\n ' };

      await provider.generateResponse(request);

      expect(requestBody(0).messages[1].content).toEqual([
        { type: 'tool_use', id: 'toolu_01', name: 'lookup', input: { id: 42, tags: ['a', 'b'] } },
      ]);
    });

    it('replays the turn unchanged on the streaming path too', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'text/event-stream' }),
        body: sseBody([
          { type: 'message_start', message: { id: 'msg_2', usage: { input_tokens: 5, output_tokens: 0 } } },
          { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'done' } },
          { type: 'message_stop' },
        ]),
      });

      const stream = await provider.streamResponse(followUp({ provider: 'anthropic', blocks: TOOL_TURN }));
      const reader = stream.getReader();
      while (!(await reader.read()).done) { /* drain */ }

      expect(requestBody(0).messages[1].content).toEqual(TOOL_TURN);
    });

    it.each([
      ['an empty array', []],
      ['a block without a type', [{ text: 'x' }]],
      ['a non-array', 'not blocks'],
      ['a user-turn block', [{ type: 'tool_result', tool_use_id: 'toolu_01', content: 'x' }]],
      ['an image block', [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'AA==' } }]],
      ['a tool_use block without an id', [{ type: 'tool_use', name: 'lookup', input: {} }]],
      ['a tool_use block without input', [{ type: 'tool_use', id: 'toolu_01', name: 'lookup' }]],
    ])('rejects malformed Anthropic content (%s) before the call', async (_label, blocks) => {
      await expect(provider.generateResponse(followUp({ provider: 'anthropic', blocks: blocks as never })))
        .rejects.toBeInstanceOf(ConfigurationError);
      expect(mockFetch).not.toHaveBeenCalled();
    });
  });

  describe('generateResponseWithTools', () => {
    it('sends the first turn back verbatim and answers it with tool_use_id', async () => {
      mockFetch
        .mockResolvedValueOnce(anthropicResponse(TOOL_TURN, 'tool_use'))
        .mockResolvedValueOnce(anthropicResponse([
          { type: 'thinking', thinking: '', signature: 'sig-final' },
          { type: 'text', text: 'Record 42 is ok.' },
        ]));

      const factory = new LLMProviderFactory({
        anthropic: { apiKey: 'test-key', maxRetries: 0 },
        defaultProvider: 'anthropic',
        costOptimization: false,
      });
      const execute = vi.fn()
        .mockResolvedValueOnce({ status: 'ok' });

      const response = await factory.generateResponseWithTools({
        model: 'claude-opus-5-5',
        messages: [{ role: 'user', content: 'Look up 42' }],
        tools: [lookupTool],
      }, { execute });

      expect(response.message).toBe('Record 42 is ok.');
      expect(execute).toHaveBeenCalledWith('lookup', { id: 42, tags: ['a', 'b'] });
      expect(mockFetch).toHaveBeenCalledTimes(2);

      const second = requestBody(1).messages;
      expect(second).toHaveLength(3);
      expect(second[0]).toEqual({ role: 'user', content: 'Look up 42' });
      expect(second[1]).toEqual({ role: 'assistant', content: TOOL_TURN });
      expect(second[2]).toEqual({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 'toolu_01', content: '{"status":"ok"}', is_error: false }],
      });
    });

    it('sends a failed tool’s error message to Claude', async () => {
      mockFetch
        .mockResolvedValueOnce(anthropicResponse(TOOL_TURN, 'tool_use'))
        .mockResolvedValueOnce(anthropicResponse([{ type: 'text', text: 'The lookup failed.' }]));

      const factory = new LLMProviderFactory({
        anthropic: { apiKey: 'test-key', maxRetries: 0 },
        defaultProvider: 'anthropic',
        costOptimization: false,
      });

      await factory.generateResponseWithTools({
        model: 'claude-opus-5-5',
        messages: [{ role: 'user', content: 'Look up 42' }],
        tools: [lookupTool],
      }, { execute: vi.fn().mockRejectedValueOnce(new Error('Database is read-only')) });

      expect(requestBody(1).messages[2].content).toEqual([
        { type: 'tool_result', tool_use_id: 'toolu_01', content: 'Database is read-only', is_error: true },
      ]);
    });
  });

  describe('tool failures without a message', () => {
    it.each([
      ['a non-Error throw', 'disk full', 'disk full'],
      ['an Error with no message', new Error(), "Tool 'lookup' failed without an error message"],
      ['a thrown object', { code: 7 }, "Tool 'lookup' failed without an error message"],
      ['an error-like object', { message: 'Rate limited, retry in 30s' }, 'Rate limited, retry in 30s'],
      ['a whitespace-only message', new Error('   '), "Tool 'lookup' failed without an error message"],
    ])('still sends an error result for %s', async (_label, thrown, expected) => {
      mockFetch
        .mockResolvedValueOnce(anthropicResponse(TOOL_TURN, 'tool_use'))
        .mockResolvedValueOnce(anthropicResponse([{ type: 'text', text: 'The lookup failed.' }]));

      const factory = new LLMProviderFactory({
        anthropic: { apiKey: 'test-key', maxRetries: 0 },
        defaultProvider: 'anthropic',
        costOptimization: false,
      });

      await factory.generateResponseWithTools({
        model: 'claude-opus-5-5',
        messages: [{ role: 'user', content: 'Look up 42' }],
        tools: [lookupTool],
      }, { execute: vi.fn().mockRejectedValueOnce(thrown) });

      expect(requestBody(1).messages[2].content).toEqual([
        { type: 'tool_result', tool_use_id: 'toolu_01', content: expected, is_error: true },
      ]);
    });
  });

  describe('longer loops and the response cache', () => {
    const SECOND_TURN = [
      { type: 'thinking', thinking: '', signature: 'sig-round-2' },
      { type: 'tool_use', id: 'toolu_02', name: 'lookup', input: { id: 43 } },
    ];

    it('sends every earlier turn back unchanged across three rounds', async () => {
      mockFetch
        .mockResolvedValueOnce(anthropicResponse(TOOL_TURN, 'tool_use'))
        .mockResolvedValueOnce(anthropicResponse(SECOND_TURN, 'tool_use'))
        .mockResolvedValueOnce(anthropicResponse([{ type: 'text', text: 'Both are ok.' }]));

      const factory = new LLMProviderFactory({
        anthropic: { apiKey: 'test-key', maxRetries: 0 },
        defaultProvider: 'anthropic',
        costOptimization: false,
      });
      const response = await factory.generateResponseWithTools({
        model: 'claude-opus-5-5',
        messages: [{ role: 'user', content: 'Look up 42 and 43' }],
        tools: [lookupTool],
      }, { execute: vi.fn().mockResolvedValueOnce('first').mockResolvedValueOnce('second') });

      expect(response.message).toBe('Both are ok.');
      const third = requestBody(2).messages;
      expect(third).toHaveLength(5);
      expect(third[1]).toEqual({ role: 'assistant', content: TOOL_TURN });
      expect(third[2].content).toEqual([{ type: 'tool_result', tool_use_id: 'toolu_01', content: 'first', is_error: false }]);
      expect(third[3]).toEqual({ role: 'assistant', content: SECOND_TURN });
      expect(third[4].content).toEqual([{ type: 'tool_result', tool_use_id: 'toolu_02', content: 'second', is_error: false }]);
    });

    const cachedToolTurn = (providerContent: unknown) => JSON.stringify({
      message: 'Let me look that up.',
      model: 'claude-opus-5-5',
      provider: 'anthropic',
      responseTime: 1,
      usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2, cost: 0 },
      finishReason: 'tool_calls',
      toolCalls: [{ id: 'toolu_01', type: 'function', function: { name: 'lookup', arguments: '{"id":42}' } }],
      providerContent,
    });

    it('serves providerContent from a cache hit', async () => {
      const cache = {
        get: vi.fn().mockResolvedValue(cachedToolTurn({ provider: 'anthropic', blocks: TOOL_TURN })),
        put: vi.fn().mockResolvedValue(undefined),
      };
      const llm = new LLMProviders({
        anthropic: { apiKey: 'test-key', maxRetries: 0 }, stateScope: 'client', responseCache: cache,
      });

      const response = await llm.generateResponse({
        model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'Look up 42' }], tools: [lookupTool],
      });

      expect(mockFetch).not.toHaveBeenCalled();
      expect(response.providerContent).toEqual({ provider: 'anthropic', blocks: TOOL_TURN });
    });

    it('treats a cached entry with malformed providerContent as a miss', async () => {
      mockFetch.mockResolvedValueOnce(anthropicResponse([{ type: 'text', text: 'fresh' }]));
      const cache = {
        get: vi.fn().mockResolvedValue(cachedToolTurn({ provider: 'anthropic', blocks: [] })),
        put: vi.fn().mockResolvedValue(undefined),
      };
      const llm = new LLMProviders({
        anthropic: { apiKey: 'test-key', maxRetries: 0 }, stateScope: 'client', responseCache: cache,
      });

      const response = await llm.generateResponse({
        model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'Look up 42' }], tools: [lookupTool],
      });

      expect(mockFetch).toHaveBeenCalledTimes(1);
      expect(response.message).toBe('fresh');
    });
  });

  describe('fallback partway through a tool loop', () => {
    const bothProviders = (defaultProvider: 'anthropic' | 'openai') => new LLMProviderFactory({
      anthropic: { apiKey: 'test-key', maxRetries: 0 },
      openai: { apiKey: 'test-openai-key', maxRetries: 0 },
      defaultProvider,
      costOptimization: false,
    });

    it('does not send Anthropic content to another provider', async () => {
      mockFetch
        .mockResolvedValueOnce(anthropicResponse(TOOL_TURN, 'tool_use'))
        .mockResolvedValueOnce(serverError())
        .mockResolvedValueOnce(openAiResponse({ content: 'Record 42 is ok.' }));

      const response = await bothProviders('anthropic').generateResponseWithTools({
        model: 'claude-opus-5-5',
        messages: [{ role: 'user', content: 'Look up 42' }],
        tools: [lookupTool],
      }, { execute: vi.fn().mockResolvedValueOnce('found') });

      expect(response.provider).toBe('openai');
      const [url, init] = mockFetch.mock.calls[2] as [string, RequestInit];
      expect(url).toContain('openai');
      const body = init.body as string;
      expect(body).not.toContain('sig-reasoning-1');
      expect(body).not.toContain('redacted_thinking');
      expect(body).not.toContain('providerContent');
      const assistant = (JSON.parse(body).messages as Array<Record<string, unknown>>)
        .find(message => message.role === 'assistant');
      expect(assistant).toMatchObject({ tool_calls: [{ id: 'toolu_01', function: { name: 'lookup' } }] });
    });

    it('rebuilds another provider’s turn for Anthropic, without whitespace-only text', async () => {
      mockFetch
        .mockResolvedValueOnce(openAiResponse({
          content: '\n\n',
          tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'lookup', arguments: '{"id":42}' } }],
        }, 'tool_calls'))
        .mockResolvedValueOnce(serverError())
        .mockResolvedValueOnce(anthropicResponse([{ type: 'text', text: 'Record 42 is ok.' }]));

      const response = await bothProviders('openai').generateResponseWithTools({
        messages: [{ role: 'user', content: 'Look up 42' }],
        tools: [lookupTool],
      }, { execute: vi.fn().mockResolvedValueOnce('found') });

      expect(response.provider).toBe('anthropic');
      const [url] = mockFetch.mock.calls[2] as [string, RequestInit];
      expect(url).toContain('anthropic');
      const messages = requestBody(2).messages;
      expect(messages[1]).toEqual({
        role: 'assistant',
        content: [{ type: 'tool_use', id: 'call_1', name: 'lookup', input: { id: 42 } }],
      });
      expect(messages[2]).toEqual({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 'call_1', content: 'found', is_error: false }],
      });
    });
  });

  describe('tool errors on OpenAI-compatible providers', () => {
    const failedLookup: LLMRequest = {
      messages: [
        { role: 'user', content: 'Look up 42' },
        { role: 'assistant', content: '', toolCalls: [{ id: 'call_1', type: 'function', function: { name: 'lookup', arguments: '{"id":42}' } }] },
        { role: 'user', content: '', toolResults: [{ id: 'call_1', output: '', error: 'Database is read-only' }] },
      ],
      tools: [lookupTool],
    };

    it.each([
      ['groq', () => new GroqProvider({ apiKey: 'test-key', maxRetries: 0 })],
      ['cerebras', () => new CerebrasProvider({ apiKey: 'test-key', maxRetries: 0 })],
      ['nvidia', () => new NvidiaProvider({ apiKey: 'test-key', maxRetries: 0 })],
      ['openai', () => new OpenAIProvider({ apiKey: 'test-key', maxRetries: 0 })],
    ])('%s sends the error text in the tool message', async (_name, make) => {
      mockFetch.mockResolvedValueOnce(openAiResponse({ content: 'The lookup failed.' }));

      await make().generateResponse(failedLookup);

      const body = JSON.parse((mockFetch.mock.calls[0][1] as RequestInit).body as string) as {
        messages: Array<Record<string, unknown>>;
      };
      const toolMessage = body.messages.find(message => message.role === 'tool');
      expect(toolMessage).toEqual({
        role: 'tool',
        tool_call_id: 'call_1',
        content: JSON.stringify({ output: '', error: 'Database is read-only' }),
      });
    });

    it('a successful result is still sent as its output', async () => {
      mockFetch.mockResolvedValueOnce(openAiResponse({ content: 'ok' }));
      const request: LLMRequest = {
        ...failedLookup,
        messages: [...failedLookup.messages.slice(0, 2), { role: 'user', content: '', toolResults: [{ id: 'call_1', output: 'found' }] }],
      };

      await new GroqProvider({ apiKey: 'test-key', maxRetries: 0 }).generateResponse(request);

      const body = JSON.parse((mockFetch.mock.calls[0][1] as RequestInit).body as string) as {
        messages: Array<Record<string, unknown>>;
      };
      expect(body.messages.find(message => message.role === 'tool')).toMatchObject({ content: 'found' });
    });
  });

  describe('boundaries', () => {
    const base: LLMResponse = {
      message: '',
      usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2, cost: 0 },
      model: 'claude-opus-5-5',
      provider: 'anthropic',
      responseTime: 1,
    };

    it('the response cache accepts valid providerContent and rejects malformed values', () => {
      expect(isCachedResponse({ ...base, providerContent: { provider: 'anthropic', blocks: TOOL_TURN } })).toBe(true);
      for (const providerContent of [
        null,
        { provider: 'anthropic' },
        { provider: 7, blocks: [] },
        { provider: 'anthropic', blocks: [{ text: 'no type' }] },
        { provider: 'anthropic', blocks: 'nope' },
        { provider: 'anthropic', blocks: [] },
        { provider: '', blocks: TOOL_TURN },
        { provider: 'anthropic', blocks: [{ type: 'tool_result', tool_use_id: 'x' }] },
      ]) {
        expect(isCachedResponse({ ...base, providerContent })).toBe(false);
      }
    });

    it('canonical normalization keeps providerContent on messages and responses', () => {
      const providerContent = { provider: 'anthropic', blocks: TOOL_TURN };
      const canonical = normalizeLegacyRequest({
        messages: [
          { role: 'user', content: 'Look up 42' },
          { role: 'assistant', content: '', providerContent },
        ],
      });
      expect(canonical.messages[1].providerContent).toEqual(providerContent);
      expect(normalizeLLMResponse({ ...base, providerContent }).providerContent).toEqual(providerContent);
    });

    it('canonical normalization rejects blocks that are not JSON with a library error', () => {
      const circular: Record<string, unknown> = { type: 'text', text: 'x' };
      circular.self = circular;
      expect(() => normalizeLegacyRequest({
        messages: [
          { role: 'user', content: 'Look up 42' },
          { role: 'assistant', content: '', providerContent: { provider: 'anthropic', blocks: [circular] } },
        ],
      })).toThrow(ConfigurationError);
    });

    it('canonical copies are deep: editing one leaves the original blocks unchanged', () => {
      const blocks = JSON.parse(JSON.stringify(TOOL_TURN)) as Array<Record<string, unknown>>;
      const canonical = normalizeLegacyRequest({
        messages: [
          { role: 'user', content: 'Look up 42' },
          { role: 'assistant', content: '', providerContent: { provider: 'anthropic', blocks } },
        ],
      });
      const copied = canonical.messages[1].providerContent!.blocks;
      (copied[0] as { signature: string }).signature = 'edited';
      copied.push({ type: 'text', text: 'extra' });

      expect(blocks).toEqual(TOOL_TURN);
    });
  });
});
