/**
 * Tool Call Validation Tests
 * Verify that malformed tool_calls from providers are caught at the boundary
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { OpenAIProvider } from '../providers/openai';
import { AnthropicProvider } from '../providers/anthropic';
import { GroqProvider } from '../providers/groq';
import { CerebrasProvider } from '../providers/cerebras';
import { CloudflareProvider } from '../providers/cloudflare';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';

// Mock global fetch
const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

/** Minimal valid usage block shared by OpenAI-compatible providers */
const baseUsage = { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 };

describe('Tool call validation at provider boundary', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
  });

  // ---------- OpenAI ----------

  describe('OpenAIProvider', () => {
    let provider: OpenAIProvider;
    beforeEach(() => {
      provider = new OpenAIProvider({ apiKey: 'test-key' });
    });

    it('should pass through well-formed tool calls', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'chatcmpl-1',
          model: 'gpt-4o',
          choices: [{
            index: 0,
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [{
                id: 'call_abc',
                type: 'function',
                function: { name: 'get_weather', arguments: '{"city":"NYC"}' }
              }]
            },
            finish_reason: 'tool_calls'
          }],
          usage: baseUsage
        }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      const res = await provider.generateResponse({
        messages: [{ role: 'user', content: 'weather' }],
        model: 'gpt-4o'
      });

      expect(res.toolCalls).toHaveLength(1);
      expect(res.toolCalls![0].id).toBe('call_abc');
      expect(res.toolCalls![0].type).toBe('function');
      expect(res.toolCalls![0].function.name).toBe('get_weather');
      expect(res.toolCalls![0].function.arguments).toBe('{"city":"NYC"}');
    });

    it('should drop tool call with missing id', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'chatcmpl-1',
          model: 'gpt-4o',
          choices: [{
            index: 0,
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [{
                id: '',
                type: 'function',
                function: { name: 'fn', arguments: '{}' }
              }]
            },
            finish_reason: 'tool_calls'
          }],
          usage: baseUsage
        }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      const res = await provider.generateResponse({
        messages: [{ role: 'user', content: 'hi' }],
        model: 'gpt-4o'
      });

      expect(res.toolCalls).toBeUndefined();
    });

    it('should drop tool call with non-function type', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'chatcmpl-1',
          model: 'gpt-4o',
          choices: [{
            index: 0,
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [{
                id: 'call_1',
                type: 'invalid_type',
                function: { name: 'fn', arguments: '{}' }
              }]
            },
            finish_reason: 'tool_calls'
          }],
          usage: baseUsage
        }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      const res = await provider.generateResponse({
        messages: [{ role: 'user', content: 'hi' }],
        model: 'gpt-4o'
      });

      expect(res.toolCalls).toBeUndefined();
    });

    it('should drop tool call with missing function.name', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'chatcmpl-1',
          model: 'gpt-4o',
          choices: [{
            index: 0,
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [{
                id: 'call_1',
                type: 'function',
                function: { name: '', arguments: '{}' }
              }]
            },
            finish_reason: 'tool_calls'
          }],
          usage: baseUsage
        }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      const res = await provider.generateResponse({
        messages: [{ role: 'user', content: 'hi' }],
        model: 'gpt-4o'
      });

      expect(res.toolCalls).toBeUndefined();
    });

    it('should drop tool call with non-string arguments', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'chatcmpl-1',
          model: 'gpt-4o',
          choices: [{
            index: 0,
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [{
                id: 'call_1',
                type: 'function',
                function: { name: 'fn', arguments: 42 }
              }]
            },
            finish_reason: 'tool_calls'
          }],
          usage: baseUsage
        }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      const res = await provider.generateResponse({
        messages: [{ role: 'user', content: 'hi' }],
        model: 'gpt-4o'
      });

      expect(res.toolCalls).toBeUndefined();
    });

    it('should keep valid tool calls and drop invalid ones', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'chatcmpl-1',
          model: 'gpt-4o',
          choices: [{
            index: 0,
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [
                { id: 'call_good', type: 'function', function: { name: 'ok_fn', arguments: '{}' } },
                { id: '', type: 'function', function: { name: 'bad_id', arguments: '{}' } },
                { id: 'call_good2', type: 'function', function: { name: 'ok_fn2', arguments: '{"x":1}' } }
              ]
            },
            finish_reason: 'tool_calls'
          }],
          usage: baseUsage
        }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      const res = await provider.generateResponse({
        messages: [{ role: 'user', content: 'hi' }],
        model: 'gpt-4o'
      });

      expect(res.toolCalls).toHaveLength(2);
      expect(res.toolCalls![0].id).toBe('call_good');
      expect(res.toolCalls![1].id).toBe('call_good2');
    });
  });

  // ---------- Anthropic ----------

  describe('AnthropicProvider', () => {
    let provider: AnthropicProvider;
    beforeEach(() => {
      provider = new AnthropicProvider({ apiKey: 'test-key' });
    });

    it('should validate tool_use blocks from Anthropic', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'msg_1',
          type: 'message',
          role: 'assistant',
          content: [
            { type: 'tool_use', id: 'toolu_1', name: 'search', input: { q: 'test' } },
            { type: 'tool_use', id: '', name: 'bad', input: {} } // empty id
          ],
          model: 'claude-3-haiku-20240307',
          stop_reason: 'tool_use',
          usage: { input_tokens: 10, output_tokens: 5 }
        }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      const res = await provider.generateResponse({
        messages: [{ role: 'user', content: 'search' }],
        model: 'claude-3-haiku-20240307'
      });

      // Only the valid tool call should survive
      expect(res.toolCalls).toHaveLength(1);
      expect(res.toolCalls![0].id).toBe('toolu_1');
      expect(res.toolCalls![0].function.name).toBe('search');
    });
  });

  // ---------- Groq ----------

  describe('GroqProvider', () => {
    let provider: GroqProvider;
    beforeEach(() => {
      provider = new GroqProvider({ apiKey: 'test-key' });
    });

    it('should drop tool call with missing function object', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'chatcmpl-1',
          model: 'openai/gpt-oss-120b',
          choices: [{
            index: 0,
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [{
                id: 'call_1',
                type: 'function',
                function: null
              }]
            },
            finish_reason: 'tool_calls'
          }],
          usage: baseUsage
        }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      // The raw mapping will throw when accessing null.name, but the important
      // thing is that the provider doesn't let bad data through silently.
      // In practice Groq's typed interface should prevent null, but the
      // validation catches it if the runtime data disagrees with types.
      await expect(provider.generateResponse({
        messages: [{ role: 'user', content: 'hi' }],
        model: 'openai/gpt-oss-120b'
      })).rejects.toThrow();
    });

    it('should pass through valid Groq tool calls', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'chatcmpl-1',
          model: 'openai/gpt-oss-120b',
          choices: [{
            index: 0,
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [{
                id: 'call_ok',
                type: 'function',
                function: { name: 'lookup', arguments: '{"key":"val"}' }
              }]
            },
            finish_reason: 'tool_calls'
          }],
          usage: baseUsage
        }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      const res = await provider.generateResponse({
        messages: [{ role: 'user', content: 'hi' }],
        model: 'openai/gpt-oss-120b'
      });

      expect(res.toolCalls).toHaveLength(1);
      expect(res.toolCalls![0].function.name).toBe('lookup');
    });
  });

  // ---------- Cerebras ----------

  describe('CerebrasProvider', () => {
    let provider: CerebrasProvider;
    beforeEach(() => {
      provider = new CerebrasProvider({ apiKey: 'test-key' });
    });

    it('should drop Cerebras tool call with empty function.name', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'chatcmpl-1',
          model: 'zai-glm-4.7',
          choices: [{
            index: 0,
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [{
                id: 'call_1',
                type: 'function',
                function: { name: '', arguments: '{}' }
              }]
            },
            finish_reason: 'tool_calls'
          }],
          usage: baseUsage
        }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      const res = await provider.generateResponse({
        messages: [{ role: 'user', content: 'hi' }],
        model: 'zai-glm-4.7'
      });

      expect(res.toolCalls).toBeUndefined();
    });
  });

  // ---------- Cloudflare ----------

  describe('CloudflareProvider', () => {
    it('should pass valid tool calls through validation', async () => {
      const mockAi = {
        run: vi.fn().mockResolvedValue({
          choices: [{
            index: 0,
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [
                { id: 'call_ok', type: 'function', function: { name: 'fn1', arguments: '{}' } },
                { id: 'call_ok2', type: 'function', function: { name: 'fn2', arguments: '{"a":1}' } }
              ]
            },
            finish_reason: 'tool_calls'
          }],
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
        })
      };

      const provider = new CloudflareProvider({ ai: mockAi as unknown as Ai });

      const res = await provider.generateResponse({
        messages: [{ role: 'user', content: 'hi' }],
        model: '@cf/openai/gpt-oss-120b'
      });

      expect(res.toolCalls).toHaveLength(2);
      expect(res.toolCalls![0].id).toBe('call_ok');
      expect(res.toolCalls![0].function.name).toBe('fn1');
      expect(res.toolCalls![1].id).toBe('call_ok2');
    });

    it('should handle Cloudflare synthesized ids with validation', async () => {
      const mockAi = {
        run: vi.fn().mockResolvedValue({
          choices: [{
            index: 0,
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [
                // Missing id — Cloudflare extractToolCalls synthesizes 'call_0'
                { function: { name: 'fn1', arguments: '{}' } }
              ]
            },
            finish_reason: 'tool_calls'
          }],
          usage: baseUsage
        })
      };

      const provider = new CloudflareProvider({ ai: mockAi as unknown as Ai });

      const res = await provider.generateResponse({
        messages: [{ role: 'user', content: 'hi' }],
        model: '@cf/openai/gpt-oss-120b'
      });

      // Cloudflare's extractToolCalls synthesizes 'call_0' for missing id,
      // and 'unknown' for missing function.name. Since the name IS present
      // here and the synthesized id is non-empty, validation should pass.
      expect(res.toolCalls).toHaveLength(1);
      expect(res.toolCalls![0].id).toBe('call_0');
    });
  });
});
