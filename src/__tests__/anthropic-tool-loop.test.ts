/**
 * Anthropic tool loops (#124): tool_result blocks carry tool_use_id, a failed
 * tool's error text reaches Claude, and the assistant turn (thinking blocks with
 * signatures, redacted_thinking, text, tool_use) is sent back unmodified.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { AnthropicProvider } from '../providers/anthropic';
import { LLMProviderFactory } from '../factory';
import { ConfigurationError } from '../errors';
import { isCachedResponse } from '../utils/response-cache';
import { normalizeLegacyRequest, normalizeLLMResponse } from '../canonical';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
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

    it.each([
      ['an empty array', []],
      ['a block without a type', [{ text: 'x' }]],
      ['a non-array', 'not blocks'],
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
  });
});
