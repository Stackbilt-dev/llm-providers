/**
 * NVIDIA NIM Provider Tests
 * Tests for the NVIDIA NIM provider with mocked HTTP responses
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { NvidiaProvider } from '../providers/nvidia';
import { AuthenticationError, ConfigurationError } from '../errors';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import type { LLMRequest } from '../types';

// Mock global fetch
const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

const MOCK_RESPONSE = {
  id: 'chatcmpl-test123',
  object: 'chat.completion',
  created: 1779442965,
  model: 'meta/llama-3.3-70b-instruct',
  choices: [{
    index: 0,
    message: {
      role: 'assistant',
      content: 'Hello! How can I help you?',
      refusal: null,
      annotations: null,
      audio: null,
      function_call: null,
      tool_calls: [],
      reasoning: null,
      reasoning_content: null
    },
    logprobs: null,
    finish_reason: 'stop',
    stop_reason: null,
    token_ids: null
  }],
  service_tier: null,
  system_fingerprint: null,
  usage: {
    prompt_tokens: 36,
    total_tokens: 50,
    completion_tokens: 14,
    prompt_tokens_details: null
  },
  prompt_logprobs: null,
  prompt_token_ids: null,
  kv_transfer_params: null
};

const MOCK_TOOL_RESPONSE = {
  id: 'chatcmpl-tool-test456',
  object: 'chat.completion',
  created: 1779442966,
  model: 'meta/llama-3.3-70b-instruct',
  choices: [{
    index: 0,
    message: {
      role: 'assistant',
      content: null,
      refusal: null,
      tool_calls: [{
        id: 'chatcmpl-tool-abc123',
        type: 'function',
        function: { name: 'get_weather', arguments: '{"city":"Paris"}' }
      }]
    },
    finish_reason: 'tool_calls'
  }],
  usage: {
    prompt_tokens: 50,
    total_tokens: 70,
    completion_tokens: 20,
    prompt_tokens_details: null
  },
  system_fingerprint: null
};

describe('NvidiaProvider', () => {
  let provider: NvidiaProvider;

  const testRequest: LLMRequest = {
    messages: [{ role: 'user', content: 'Hello, world!' }],
    model: 'meta/llama-3.3-70b-instruct',
    maxTokens: 100,
    temperature: 0.7
  };

  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
    provider = new NvidiaProvider({
      apiKey: 'nvapi-test-key'
    });
  });

  describe('Constructor', () => {
    it('should initialize with valid config', () => {
      expect(provider.name).toBe('nvidia');
      expect(provider.models).toContain('meta/llama-3.3-70b-instruct');
      expect(provider.models).toContain('nvidia/llama-3.1-nemotron-70b-instruct');
      expect(provider.models).toContain('deepseek-ai/deepseek-v4-flash');
      expect(provider.supportsStreaming).toBe(true);
      expect(provider.supportsTools).toBe(true);
      expect(provider.supportsBatching).toBe(false);
    });

    it('should throw AuthenticationError without API key', () => {
      expect(() => new NvidiaProvider({})).toThrow(AuthenticationError);
      expect(() => new NvidiaProvider({})).toThrow('NVIDIA API key is required');
    });

    it('should use default base URL', () => {
      const p = new NvidiaProvider({ apiKey: 'nvapi-test' });
      expect(p.validateConfig()).toBe(true);
    });

    it('should accept custom base URL', () => {
      const p = new NvidiaProvider({
        apiKey: 'nvapi-test',
        baseUrl: 'https://custom.nvidia.com/v1'
      });
      expect(p.validateConfig()).toBe(true);
    });
  });

  describe('validateConfig', () => {
    it('should return true with valid config', () => {
      expect(provider.validateConfig()).toBe(true);
    });
  });

  describe('getModels', () => {
    it('should return available models', () => {
      const models = provider.getModels();
      expect(models).toContain('meta/llama-3.3-70b-instruct');
      expect(models).toContain('meta/llama-4-maverick-17b-128e-instruct');
      expect(models).toContain('nvidia/llama-3.1-nemotron-70b-instruct');
      expect(models).toContain('nvidia/llama-3.3-nemotron-super-49b-v1');
      expect(models).toContain('deepseek-ai/deepseek-v4-flash');
      expect(models).toContain('meta/llama-3.1-70b-instruct');
      expect(models).toContain('nvidia/llama-3.1-nemotron-ultra-253b-v1');
      expect(models).toContain('mistralai/mistral-large-2-instruct');
      expect(models).toContain('deepseek-ai/deepseek-v4-pro');
    });

    it('should return a copy of the models array', () => {
      const models = provider.getModels();
      models.push('fake-model');
      expect(provider.getModels()).not.toContain('fake-model');
    });
  });

  describe('generateResponse', () => {
    it('should generate a response successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => MOCK_RESPONSE
      });

      const response = await provider.generateResponse(testRequest);

      expect(response.provider).toBe('nvidia');
      expect(response.content).toBe('Hello! How can I help you?');
      expect(response.usage.inputTokens).toBe(36);
      expect(response.usage.outputTokens).toBe(14);
      expect(response.usage.totalTokens).toBe(50);
      expect(response.usage.cost).toBe(0); // zero-placeholder pricing
      expect(response.finishReason).toBe('stop');
      expect(response.toolCalls).toBeUndefined();
    });

    it('should use Authorization Bearer header', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => MOCK_RESPONSE
      });

      await provider.generateResponse(testRequest);

      const [url, options] = mockFetch.mock.calls[0];
      expect(url).toContain('https://integrate.api.nvidia.com/v1/chat/completions');
      expect(options.headers['Authorization']).toBe('Bearer nvapi-test-key');
    });

    it('should send model in request body', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => MOCK_RESPONSE
      });

      await provider.generateResponse(testRequest);

      const [, options] = mockFetch.mock.calls[0];
      const body = JSON.parse(options.body);
      expect(body.model).toBe('meta/llama-3.3-70b-instruct');
    });

    it('should handle tool_calls as empty array (NVIDIA-specific response shape)', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          ...MOCK_RESPONSE,
          choices: [{
            ...MOCK_RESPONSE.choices[0],
            message: { ...MOCK_RESPONSE.choices[0].message, tool_calls: [] }
          }]
        })
      });

      const response = await provider.generateResponse(testRequest);
      expect(response.toolCalls).toBeUndefined();
    });

    it('should handle null prompt_tokens_details gracefully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          ...MOCK_RESPONSE,
          usage: { ...MOCK_RESPONSE.usage, prompt_tokens_details: null }
        })
      });

      const response = await provider.generateResponse(testRequest);
      expect(response.usage.cachedInputTokens).toBeUndefined();
    });

    it('should extract tool calls from response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => MOCK_TOOL_RESPONSE
      });

      const toolRequest: LLMRequest = {
        ...testRequest,
        tools: [{
          type: 'function',
          function: {
            name: 'get_weather',
            description: 'Get current weather',
            parameters: {
              type: 'object',
              properties: { city: { type: 'string' } },
              required: ['city']
            }
          }
        }]
      };

      const response = await provider.generateResponse(toolRequest);
      expect(response.toolCalls).toHaveLength(1);
      expect(response.toolCalls?.[0].function.name).toBe('get_weather');
      expect(response.toolCalls?.[0].function.arguments).toBe('{"city":"Paris"}');
    });

    it('should throw ConfigurationError for tools on non-tool-capable model', () => {
      const badRequest: LLMRequest = {
        messages: [{ role: 'user', content: 'hello' }],
        model: 'meta/llama2-70b',
        tools: [{
          type: 'function',
          function: {
            name: 'test',
            description: 'test',
            parameters: { type: 'object', properties: {} }
          }
        }]
      };

      expect(() => provider.generateResponse(badRequest)).rejects.toThrow(ConfigurationError);
    });

    it('should add json instruction to system prompt in json_object mode', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => MOCK_RESPONSE
      });

      await provider.generateResponse({
        ...testRequest,
        systemPrompt: 'You are helpful.',
        response_format: { type: 'json_object' }
      });

      const [, options] = mockFetch.mock.calls[0];
      const body = JSON.parse(options.body);
      const systemMsg = body.messages.find((m: { role: string }) => m.role === 'system');
      expect(systemMsg.content).toContain('You are helpful.');
      expect(systemMsg.content).toContain('valid JSON');
    });

    it('should forward systemPrompt as first message', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => MOCK_RESPONSE
      });

      await provider.generateResponse({
        ...testRequest,
        systemPrompt: 'You are a test assistant.'
      });

      const [, options] = mockFetch.mock.calls[0];
      const body = JSON.parse(options.body);
      expect(body.messages[0].role).toBe('system');
      expect(body.messages[0].content).toBe('You are a test assistant.');
    });
  });

  describe('healthCheck', () => {
    it('should return true when /models endpoint is reachable', async () => {
      mockFetch.mockResolvedValueOnce({ ok: true });
      const healthy = await provider.healthCheck();
      expect(healthy).toBe(true);
    });

    it('should return false when /models endpoint fails', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));
      const healthy = await provider.healthCheck();
      expect(healthy).toBe(false);
    });
  });

  describe('getProviderBalance', () => {
    it('should return not_supported status', async () => {
      const balance = await provider.getProviderBalance!();
      expect(balance.status).toBe('unavailable');
      expect(balance.source).toBe('not_supported');
    });
  });

  describe('estimateCost', () => {
    it('should return 0 for all models (zero-placeholder pricing)', () => {
      const cost = provider.estimateCost({
        messages: [{ role: 'user', content: 'hello' }],
        model: 'meta/llama-3.3-70b-instruct',
        maxTokens: 1000
      });
      expect(cost).toBe(0);
    });
  });

  describe('streamResponse', () => {
    it('should stream SSE chunks from NVIDIA NIM', async () => {
      const sseChunks = [
        'data: {"choices":[{"delta":{"content":"Hello"}}],"model":"meta/llama-3.3-70b-instruct"}\n\n',
        'data: {"choices":[{"delta":{"content":" world"}}],"model":"meta/llama-3.3-70b-instruct"}\n\n',
        'data: [DONE]\n\n',
      ];

      const encoder = new TextEncoder();
      let chunkIndex = 0;

      const mockReader = {
        read: vi.fn().mockImplementation(async () => {
          if (chunkIndex < sseChunks.length) {
            return { done: false, value: encoder.encode(sseChunks[chunkIndex++]) };
          }
          return { done: true, value: undefined };
        }),
        cancel: vi.fn().mockResolvedValue(undefined)
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: { getReader: () => mockReader }
      });

      const stream = await provider.streamResponse(testRequest);
      const reader = stream.getReader();

      const chunks: string[] = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
      }

      expect(chunks).toEqual(['Hello', ' world']);
    });
  });
});
