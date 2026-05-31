/**
 * Groq Provider Tests
 * Tests for the Groq provider with mocked HTTP responses
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { GroqProvider } from '../providers/groq';
import { AuthenticationError, ConfigurationError } from '../errors';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import type { LLMRequest } from '../types';

// Mock global fetch
const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

describe('GroqProvider', () => {
  let provider: GroqProvider;

  const testRequest: LLMRequest = {
    messages: [{ role: 'user', content: 'Hello, world!' }],
    model: 'llama-3.3-70b-versatile',
    maxTokens: 100,
    temperature: 0.7
  };

  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
    provider = new GroqProvider({
      apiKey: 'test-groq-key'
    });
  });

  describe('Constructor', () => {
    it('should initialize with valid config', () => {
      expect(provider.name).toBe('groq');
      expect(provider.models).toContain('llama-3.3-70b-versatile');
      expect(provider.models).toContain('llama-3.1-8b-instant');
      expect(provider.models).toContain('openai/gpt-oss-120b');
      expect(provider.supportsStreaming).toBe(true);
      expect(provider.supportsTools).toBe(true);
      expect(provider.supportsBatching).toBe(false);
      expect(provider.supportsVision).toBe(false);
    });

    it('should throw AuthenticationError without API key', () => {
      expect(() => new GroqProvider({})).toThrow(AuthenticationError);
      expect(() => new GroqProvider({})).toThrow('Groq API key is required');
    });

    it('should use default base URL', () => {
      const p = new GroqProvider({ apiKey: 'key' });
      expect(p.validateConfig()).toBe(true);
    });

    it('should accept custom base URL', () => {
      const p = new GroqProvider({
        apiKey: 'key',
        baseUrl: 'https://custom.groq.com/v1'
      });
      expect(p.validateConfig()).toBe(true);
    });
  });

  describe('vision handling', () => {
    it('rejects image inputs instead of silently dropping them', async () => {
      await expect(provider.generateResponse({
        ...testRequest,
        images: [{ data: 'QUJD', mimeType: 'image/png' }]
      })).rejects.toThrow(ConfigurationError);
      expect(mockFetch).not.toHaveBeenCalled();
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
      expect(models).toEqual([
        'llama-3.3-70b-versatile',
        'llama-3.1-8b-instant',
        'openai/gpt-oss-120b',
        'groq/compound',
        'groq/compound-mini',
      ]);
    });

    it('should return a copy of the models array', () => {
      const models = provider.getModels();
      models.push('fake-model');
      expect(provider.getModels()).not.toContain('fake-model');
    });
  });

  describe('getProviderBalance', () => {
    it('should report Groq billing API as unavailable', async () => {
      const balance = await provider.getProviderBalance();

      expect(balance).toMatchObject({
        provider: 'groq',
        status: 'unavailable',
        source: 'not_supported'
      });
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
          model: 'llama-3.3-70b-versatile',
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
      expect(response.provider).toBe('groq');
      expect(response.model).toBe('llama-3.3-70b-versatile');
      expect(response.usage.inputTokens).toBe(10);
      expect(response.usage.outputTokens).toBe(5);
      expect(response.usage.totalTokens).toBe(15);
      expect(response.finishReason).toBe('stop');
    });

    it('should forward Cloudflare AI Gateway metadata headers only for gateway base URLs', async () => {
      provider = new GroqProvider({
        apiKey: 'test-groq-key',
        baseUrl: 'https://gateway.ai.cloudflare.com/v1/account/gateway/groq'
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'chatcmpl-123',
          object: 'chat.completion',
          created: 1700000000,
          model: 'llama-3.3-70b-versatile',
          choices: [{
            index: 0,
            message: { role: 'assistant', content: 'Hello!' },
            finish_reason: 'stop'
          }],
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
        }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      await provider.generateResponse({
        ...testRequest,
        requestId: 'req-1',
        tenantId: 'tenant-1',
        gatewayMetadata: {
          requestId: 'gw-req-1',
          cacheKey: 'cache-key',
          cacheTtl: 60,
          customMetadata: { feature: 'demo' }
        }
      });

      const headers = mockFetch.mock.calls[0][1].headers;
      expect(headers['cf-aig-cache-key']).toBe('cache-key');
      expect(headers['cf-aig-cache-ttl']).toBe('60');
      expect(JSON.parse(headers['cf-aig-metadata'])).toMatchObject({
        requestId: 'gw-req-1',
        llmRequestId: 'req-1',
        tenantId: 'tenant-1',
        feature: 'demo'
      });
    });

    it('should send correct request format', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'chatcmpl-123',
          object: 'chat.completion',
          created: 1700000000,
          model: 'llama-3.3-70b-versatile',
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

      expect(url).toBe('https://api.groq.com/openai/v1/chat/completions');
      expect(options.method).toBe('POST');
      expect(options.headers['Authorization']).toBe('Bearer test-groq-key');
      expect(options.headers['Content-Type']).toBe('application/json');

      const body = JSON.parse(options.body);
      expect(body.model).toBe('llama-3.3-70b-versatile');
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
          model: 'llama-3.3-70b-versatile',
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
        model: 'llama-3.3-70b-versatile'
      })).rejects.toThrow('Request must contain at least one message');
    });

    it('should reject unsupported model', async () => {
      await expect(provider.generateResponse({
        messages: [{ role: 'user', content: 'Hi' }],
        model: 'gpt-4'
      })).rejects.toThrow("Model 'gpt-4' not supported");
    });
  });

  describe('estimateCost', () => {
    it('should estimate cost for llama-3.3-70b-versatile', () => {
      const cost = provider.estimateCost(testRequest);
      expect(cost).toBeGreaterThan(0);
    });

    it('should estimate cost for llama-3.1-8b-instant', () => {
      const cost = provider.estimateCost({
        ...testRequest,
        model: 'llama-3.1-8b-instant'
      });
      expect(cost).toBeGreaterThan(0);
    });

    it('should estimate cost for openai/gpt-oss-120b', () => {
      const cost = provider.estimateCost({
        ...testRequest,
        model: 'openai/gpt-oss-120b'
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
      const cost70b = provider.estimateCost({ ...testRequest, model: 'llama-3.3-70b-versatile' });
      const cost8b = provider.estimateCost({ ...testRequest, model: 'llama-3.1-8b-instant' });
      expect(cost70b).toBeGreaterThan(cost8b);
    });
  });

  describe('tool calling', () => {
    const toolRequest: LLMRequest = {
      messages: [{ role: 'user', content: 'What is the weather?' }],
      model: 'openai/gpt-oss-120b',
      maxTokens: 100,
      tools: [{
        type: 'function',
        function: {
          name: 'get_weather',
          description: 'Get current weather',
          parameters: { type: 'object', properties: { location: { type: 'string' } } }
        }
      }],
      toolChoice: 'auto'
    };

    const toolCallResponse = {
      id: 'chatcmpl-tool-1',
      object: 'chat.completion',
      created: 1700000000,
      model: 'openai/gpt-oss-120b',
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content: null,
          tool_calls: [{
            id: 'call_abc123',
            type: 'function',
            function: { name: 'get_weather', arguments: '{"location":"London"}' }
          }]
        },
        finish_reason: 'tool_calls'
      }],
      usage: { prompt_tokens: 20, completion_tokens: 15, total_tokens: 35 }
    };

    it('should pass tools and parse tool_calls in response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => toolCallResponse,
        headers: new Headers({ 'content-type': 'application/json' })
      });

      const response = await provider.generateResponse(toolRequest);

      // Verify response has tool calls
      expect(response.toolCalls).toHaveLength(1);
      expect(response.toolCalls![0].id).toBe('call_abc123');
      expect(response.toolCalls![0].function.name).toBe('get_weather');
      expect(response.toolCalls![0].function.arguments).toBe('{"location":"London"}');
      expect(response.finishReason).toBe('tool_calls');

      // Verify request body included tools
      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.tools).toHaveLength(1);
      expect(body.tools[0].function.name).toBe('get_weather');
      expect(body.tool_choice).toBe('auto');
    });

    it('should handle multi-turn tool conversations', async () => {
      const multiTurnRequest: LLMRequest = {
        messages: [
          { role: 'user', content: 'What is the weather?' },
          {
            role: 'assistant',
            content: '',
            toolCalls: [{
              id: 'call_abc123',
              type: 'function',
              function: { name: 'get_weather', arguments: '{"location":"London"}' }
            }]
          },
          {
            role: 'user',
            content: '',
            toolResults: [{
              id: 'call_abc123',
              output: '{"temp": 15, "condition": "cloudy"}'
            }]
          }
        ],
        model: 'openai/gpt-oss-120b',
        maxTokens: 100,
        tools: toolRequest.tools
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'chatcmpl-tool-2',
          object: 'chat.completion',
          created: 1700000000,
          model: 'openai/gpt-oss-120b',
          choices: [{
            index: 0,
            message: { role: 'assistant', content: 'It is 15°C and cloudy in London.' },
            finish_reason: 'stop'
          }],
          usage: { prompt_tokens: 40, completion_tokens: 15, total_tokens: 55 }
        }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      const response = await provider.generateResponse(multiTurnRequest);

      expect(response.message).toBe('It is 15°C and cloudy in London.');

      // Verify the serialized messages include tool_calls and tool role
      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      const assistantMsg = body.messages.find((m: Record<string, unknown>) => m.role === 'assistant');
      expect(assistantMsg.tool_calls).toHaveLength(1);
      expect(assistantMsg.tool_calls[0].id).toBe('call_abc123');

      const toolMsg = body.messages.find((m: Record<string, unknown>) => m.role === 'tool');
      expect(toolMsg.tool_call_id).toBe('call_abc123');
      expect(toolMsg.content).toBe('{"temp": 15, "condition": "cloudy"}');
    });

    it('should reject tools for non-tool-capable models', async () => {
      await expect(provider.generateResponse({
        ...toolRequest,
        model: 'llama-3.1-8b-instant'
      })).rejects.toBeInstanceOf(ConfigurationError);

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it('should support tool calling on llama-3.3-70b-versatile', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          ...toolCallResponse,
          model: 'llama-3.3-70b-versatile'
        }),
        headers: new Headers({ 'content-type': 'application/json' })
      });

      const response = await provider.generateResponse({
        ...toolRequest,
        model: 'llama-3.3-70b-versatile'
      });

      expect(response.toolCalls).toHaveLength(1);

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.tools).toHaveLength(1);
    });
  });

  describe('built-in tools (S4)', () => {
    const okResponse = (model: string) => ({
      ok: true,
      json: async () => ({
        id: 'chatcmpl-bi-1',
        object: 'chat.completion',
        created: 1700000000,
        model,
        choices: [{ index: 0, message: { role: 'assistant', content: 'done' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
      }),
      headers: new Headers({ 'content-type': 'application/json' })
    });

    it('forks web_search onto compound_custom.tools.enabled_tools for groq/compound', async () => {
      mockFetch.mockResolvedValueOnce(okResponse('groq/compound'));

      await provider.generateResponse({
        messages: [{ role: 'user', content: 'Find sources on X.' }],
        model: 'groq/compound',
        builtInTools: [{ type: 'web_search' }],
        maxTokens: 100,
      });

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.compound_custom).toEqual({ tools: { enabled_tools: ['web_search'] } });
      // Compound does NOT use the OpenAI-style tools array for built-ins.
      expect(body.tools).toBeUndefined();
    });

    it('passes all five normalized identifiers verbatim to compound', async () => {
      mockFetch.mockResolvedValueOnce(okResponse('groq/compound-mini'));

      await provider.generateResponse({
        messages: [{ role: 'user', content: 'Research.' }],
        model: 'groq/compound-mini',
        builtInTools: [
          { type: 'web_search' },
          { type: 'visit_website' },
          { type: 'browser_automation' },
          { type: 'code_interpreter' },
          { type: 'wolfram_alpha' },
        ],
        maxTokens: 100,
      });

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.compound_custom.tools.enabled_tools).toEqual([
        'web_search', 'visit_website', 'browser_automation', 'code_interpreter', 'wolfram_alpha',
      ]);
    });

    it('translates web_search → browser_search on the gpt-oss tools array', async () => {
      mockFetch.mockResolvedValueOnce(okResponse('openai/gpt-oss-120b'));

      await provider.generateResponse({
        messages: [{ role: 'user', content: 'Find sources on X.' }],
        model: 'openai/gpt-oss-120b',
        builtInTools: [{ type: 'web_search' }],
        maxTokens: 100,
      });

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.tools).toEqual([{ type: 'browser_search' }]);
      expect(body.compound_custom).toBeUndefined();
    });

    it('merges built-in tools alongside function tools on gpt-oss', async () => {
      mockFetch.mockResolvedValueOnce(okResponse('openai/gpt-oss-120b'));

      await provider.generateResponse({
        messages: [{ role: 'user', content: 'Weather then search.' }],
        model: 'openai/gpt-oss-120b',
        tools: [{
          type: 'function',
          function: { name: 'get_weather', description: 'Weather', parameters: { type: 'object' } }
        }],
        builtInTools: [{ type: 'web_search' }],
        maxTokens: 100,
      });

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      const types = body.tools.map((t: { type: string }) => t.type);
      expect(types).toContain('function');
      expect(types).toContain('browser_search');
    });

    it('rejects built-in tools on a function-only model, naming capable models', async () => {
      // Single invocation, asserted twice — a second generateResponse call would
      // accumulate a circuit-breaker failure and mask the ConfigurationError.
      const promise = provider.generateResponse({
        messages: [{ role: 'user', content: 'hi' }],
        model: 'llama-3.3-70b-versatile',
        builtInTools: [{ type: 'web_search' }],
        maxTokens: 100,
      });

      await expect(promise).rejects.toBeInstanceOf(ConfigurationError);
      await expect(promise).rejects.toThrow(/groq\/compound/);

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it('rejects an unsupported tool on gpt-oss, naming its supported subset', async () => {
      await expect(provider.generateResponse({
        messages: [{ role: 'user', content: 'hi' }],
        model: 'openai/gpt-oss-120b',
        builtInTools: [{ type: 'wolfram_alpha' }],
        maxTokens: 100,
      })).rejects.toThrow(/web_search/);

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it('accepts a pinned groq/compound model through validateRequest', async () => {
      mockFetch.mockResolvedValueOnce(okResponse('groq/compound'));

      const response = await provider.generateResponse({
        messages: [{ role: 'user', content: 'Research.' }],
        model: 'groq/compound',
        builtInTools: [{ type: 'web_search' }],
        maxTokens: 100,
      });

      expect(response.provider).toBe('groq');
      expect(mockFetch).toHaveBeenCalled();
    });
  });

  describe('built-in tool results (S5)', () => {
    // Real wire shape locked from the S0 spike: executed_tools[].search_results
    // is an object { results: [...] }, results carry {title,url,content,score}.
    const searchResponse = (model: string, message: Record<string, unknown>) => ({
      ok: true,
      json: async () => ({
        id: 'chatcmpl-bi-res',
        object: 'chat.completion',
        created: 1700000000,
        model,
        choices: [{ index: 0, message: { role: 'assistant', content: 'answer', ...message }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 30, completion_tokens: 40, total_tokens: 70 }
      }),
      headers: new Headers({ 'content-type': 'application/json' })
    });

    it('flattens compound executed_tools into metadata.builtInToolResults (all four citation fields)', async () => {
      mockFetch.mockResolvedValueOnce(searchResponse('groq/compound', {
        reasoning: 'I should search the web.',
        executed_tools: [{
          index: 0,
          type: 'search',
          arguments: '{"query":"authoritative sources on X"}',
          search_results: {
            results: [
              { title: 'Source A', url: 'https://a.example/x', content: 'snippet A', score: 0.91 },
              { title: 'Source B', url: 'https://b.example/x', content: 'snippet B', score: 0.84 },
            ]
          }
        }]
      }));

      const res = await provider.generateResponse({
        messages: [{ role: 'user', content: 'Find sources.' }],
        model: 'groq/compound',
        builtInTools: [{ type: 'web_search' }],
        maxTokens: 100,
      });

      const results = res.metadata?.builtInToolResults as Array<Record<string, unknown>>;
      expect(results).toHaveLength(1);
      expect(results[0].type).toBe('search');
      expect(results[0].arguments).toBe('{"query":"authoritative sources on X"}');
      expect(results[0].name).toBeUndefined(); // compound omits name
      const citations = results[0].results as Array<Record<string, unknown>>;
      expect(citations).toHaveLength(2);
      // Direct assertion on the four citation fields (the binding S5 note).
      expect(citations[0]).toEqual({ title: 'Source A', url: 'https://a.example/x', content: 'snippet A', score: 0.91 });
      // reasoning surfaces too
      expect(res.metadata?.reasoning).toBe('I should search the web.');
    });

    it('keeps only search executions and preserves gpt-oss name/arguments', async () => {
      mockFetch.mockResolvedValueOnce(searchResponse('openai/gpt-oss-120b', {
        executed_tools: [
          {
            index: 0,
            type: 'browser_search',
            name: 'browser.search',
            arguments: '{"query":"X"}',
            search_results: { results: [{ title: 'T', url: 'https://t.example', content: 'c', score: 0.5 }] }
          },
          // Non-search execution (no search_results) — dropped by design.
          { index: 1, type: 'browser.open', name: 'browser.open', arguments: '{"id":1}' },
        ]
      }));

      const res = await provider.generateResponse({
        messages: [{ role: 'user', content: 'Find.' }],
        model: 'openai/gpt-oss-120b',
        builtInTools: [{ type: 'web_search' }],
        maxTokens: 100,
      });

      const results = res.metadata?.builtInToolResults as Array<Record<string, unknown>>;
      expect(results).toHaveLength(1);
      expect(results[0].type).toBe('browser_search');
      expect(results[0].name).toBe('browser.search');
    });

    it('omits builtInToolResults when no execution carries results', async () => {
      mockFetch.mockResolvedValueOnce(searchResponse('groq/compound', {
        executed_tools: [
          { index: 0, type: 'code_interpreter', arguments: '{}', output: '42' },
          { index: 1, type: 'search', search_results: { results: [] } },
        ]
      }));

      const res = await provider.generateResponse({
        messages: [{ role: 'user', content: 'Compute.' }],
        model: 'groq/compound',
        builtInTools: [{ type: 'code_interpreter' }],
        maxTokens: 100,
      });

      expect(res.metadata?.builtInToolResults).toBeUndefined();
    });

    it('omits builtInToolResults entirely for a plain response (no executed_tools)', async () => {
      mockFetch.mockResolvedValueOnce(searchResponse('llama-3.3-70b-versatile', {}));

      const res = await provider.generateResponse({
        messages: [{ role: 'user', content: 'hi' }],
        model: 'llama-3.3-70b-versatile',
        maxTokens: 100,
      });

      expect(res.metadata?.builtInToolResults).toBeUndefined();
      expect(res.metadata?.reasoning).toBeUndefined();
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
      expect(url).toBe('https://api.groq.com/openai/v1/models');
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
          model: 'llama-3.3-70b-versatile',
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
          model: 'llama-3.3-70b-versatile',
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
