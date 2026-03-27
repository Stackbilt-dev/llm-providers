/**
 * Cerebras Provider Tests
 * Tests for the Cerebras provider with mocked HTTP responses
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { CerebrasProvider } from '../providers/cerebras';
import { AuthenticationError, ConfigurationError } from '../errors';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import type { LLMRequest } from '../types';

// Mock global fetch
const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

describe('CerebrasProvider', () => {
  let provider: CerebrasProvider;

  const testRequest: LLMRequest = {
    messages: [{ role: 'user', content: 'Hello, world!' }],
    model: 'llama-3.1-8b',
    maxTokens: 100,
    temperature: 0.7
  };

  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
    provider = new CerebrasProvider({
      apiKey: 'test-cerebras-key'
    });
  });

  describe('Constructor', () => {
    it('should initialize with valid config', () => {
      expect(provider.name).toBe('cerebras');
      expect(provider.models).toContain('llama-3.1-8b');
      expect(provider.models).toContain('llama-3.3-70b');
      expect(provider.models).toContain('zai-glm-4.7');
      expect(provider.models).toContain('qwen-3-235b-a22b-instruct-2507');
      expect(provider.supportsStreaming).toBe(true);
      expect(provider.supportsTools).toBe(true);
      expect(provider.supportsBatching).toBe(false);
    });

    it('should throw AuthenticationError without API key', () => {
      expect(() => new CerebrasProvider({})).toThrow(AuthenticationError);
      expect(() => new CerebrasProvider({})).toThrow('Cerebras API key is required');
    });

    it('should use default base URL', () => {
      const p = new CerebrasProvider({ apiKey: 'key' });
      expect(p.validateConfig()).toBe(true);
    });

    it('should accept custom base URL', () => {
      const p = new CerebrasProvider({
        apiKey: 'key',
        baseUrl: 'https://custom.cerebras.ai/v1'
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
      expect(models).toEqual(['llama-3.1-8b', 'llama-3.3-70b', 'zai-glm-4.7', 'qwen-3-235b-a22b-instruct-2507']);
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
        json: async () => ({
          id: 'chatcmpl-123',
          object: 'chat.completion',
          created: 1700000000,
          model: 'llama-3.1-8b',
          choices: [{
            index: 0,
            message: { role: 'assistant', content: 'Hello!' },
            finish_reason: 'stop'
          }],
          usage: {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15
          }
        }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      const response = await provider.generateResponse(testRequest);

      expect(response.message).toBe('Hello!');
      expect(response.provider).toBe('cerebras');
      expect(response.model).toBe('llama-3.1-8b');
      expect(response.usage.inputTokens).toBe(10);
      expect(response.usage.outputTokens).toBe(5);
      expect(response.usage.totalTokens).toBe(15);
      expect(response.finishReason).toBe('stop');
    });

    it('should send correct request format', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'chatcmpl-123',
          object: 'chat.completion',
          created: 1700000000,
          model: 'llama-3.1-8b',
          choices: [{
            index: 0,
            message: { role: 'assistant', content: 'Hi!' },
            finish_reason: 'stop'
          }],
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
        }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      await provider.generateResponse(testRequest);

      expect(mockFetch).toHaveBeenCalledOnce();
      const [url, options] = mockFetch.mock.calls[0];

      expect(url).toBe('https://api.cerebras.ai/v1/chat/completions');
      expect(options.method).toBe('POST');
      expect(options.headers['Authorization']).toBe('Bearer test-cerebras-key');
      expect(options.headers['Content-Type']).toBe('application/json');

      const body = JSON.parse(options.body);
      expect(body.model).toBe('llama-3.1-8b');
      expect(body.messages).toEqual([{ role: 'user', content: 'Hello, world!' }]);
      expect(body.temperature).toBe(0.7);
      expect(body.max_tokens).toBe(100);
    });

    it('should include system prompt when provided', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'chatcmpl-123',
          object: 'chat.completion',
          created: 1700000000,
          model: 'llama-3.1-8b',
          choices: [{
            index: 0,
            message: { role: 'assistant', content: 'I am helpful.' },
            finish_reason: 'stop'
          }],
          usage: { prompt_tokens: 20, completion_tokens: 5, total_tokens: 25 }
        }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      await provider.generateResponse({
        ...testRequest,
        systemPrompt: 'You are a helpful assistant.'
      });

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.messages[0]).toEqual({
        role: 'system',
        content: 'You are a helpful assistant.'
      });
      expect(body.messages[1]).toEqual({
        role: 'user',
        content: 'Hello, world!'
      });
    });

    it('should throw on HTTP error', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({ error: { message: 'Invalid API key' } })
      });

      await expect(provider.generateResponse(testRequest)).rejects.toThrow();
    });

    it('should reject empty messages', async () => {
      await expect(provider.generateResponse({
        messages: [],
        model: 'llama-3.1-8b'
      })).rejects.toThrow('Request must contain at least one message');
    });

    it('should reject unsupported model', async () => {
      await expect(provider.generateResponse({
        messages: [{ role: 'user', content: 'Hi' }],
        model: 'gpt-4'
      })).rejects.toThrow("Model 'gpt-4' not supported");
    });

    it('should reject tools for non-tool-capable models', async () => {
      await expect(provider.generateResponse({
        messages: [{ role: 'user', content: 'What is the weather?' }],
        model: 'llama-3.1-8b',
        tools: [{
          type: 'function',
          function: {
            name: 'get_weather',
            description: 'Get current weather',
            parameters: { type: 'object', properties: { location: { type: 'string' } } }
          }
        }],
        toolChoice: 'auto'
      })).rejects.toBeInstanceOf(ConfigurationError);

      expect(mockFetch).not.toHaveBeenCalled();
    });
  });

  describe('estimateCost', () => {
    it('should estimate cost for llama-3.1-8b', () => {
      const cost = provider.estimateCost(testRequest);
      expect(cost).toBeGreaterThan(0);
    });

    it('should estimate cost for llama-3.3-70b', () => {
      const cost = provider.estimateCost({
        ...testRequest,
        model: 'llama-3.3-70b'
      });
      expect(cost).toBeGreaterThan(0);
    });

    it('should return 0 for unknown model', () => {
      const cost = provider.estimateCost({
        ...testRequest,
        model: 'unknown-model'
      });
      expect(cost).toBe(0);
    });

    it('should estimate higher cost for 70b vs 8b', () => {
      const cost8b = provider.estimateCost({ ...testRequest, model: 'llama-3.1-8b' });
      const cost70b = provider.estimateCost({ ...testRequest, model: 'llama-3.3-70b' });
      expect(cost70b).toBeGreaterThan(cost8b);
    });
  });

  describe('healthCheck', () => {
    it('should return true when API is healthy', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: [] }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      const healthy = await provider.healthCheck();
      expect(healthy).toBe(true);

      const [url, options] = mockFetch.mock.calls[0];
      expect(url).toBe('https://api.cerebras.ai/v1/models');
      expect(options.method).toBe('GET');
    });

    it('should return false when API is unhealthy', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        headers: new Headers()
      });

      const healthy = await provider.healthCheck();
      expect(healthy).toBe(false);
    });

    it('should return false on network error', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      const healthy = await provider.healthCheck();
      expect(healthy).toBe(false);
    });
  });

  describe('metrics', () => {
    it('should track metrics on successful request', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'chatcmpl-123',
          object: 'chat.completion',
          created: 1700000000,
          model: 'llama-3.1-8b',
          choices: [{
            index: 0,
            message: { role: 'assistant', content: 'Hello!' },
            finish_reason: 'stop'
          }],
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
        }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      await provider.generateResponse(testRequest);

      const metrics = provider.getMetrics();
      expect(metrics.requestCount).toBe(1);
      expect(metrics.successCount).toBe(1);
      expect(metrics.errorCount).toBe(0);
      expect(metrics.averageLatency).toBeGreaterThanOrEqual(0);
    });

    it('should track metrics on failed request', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({ error: { message: 'Server error' } })
      });

      await expect(provider.generateResponse(testRequest)).rejects.toThrow();

      const metrics = provider.getMetrics();
      expect(metrics.requestCount).toBe(1);
      expect(metrics.errorCount).toBe(1);
    });

    it('should reset metrics', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'chatcmpl-123',
          object: 'chat.completion',
          created: 1700000000,
          model: 'llama-3.1-8b',
          choices: [{
            index: 0,
            message: { role: 'assistant', content: 'Hello!' },
            finish_reason: 'stop'
          }],
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
        }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      await provider.generateResponse(testRequest);
      provider.resetMetrics();

      const metrics = provider.getMetrics();
      expect(metrics.requestCount).toBe(0);
      expect(metrics.successCount).toBe(0);
      expect(metrics.totalCost).toBe(0);
    });
  });
});
