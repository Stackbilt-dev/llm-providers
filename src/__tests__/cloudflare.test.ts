/**
 * Cloudflare Provider Tests
 * Tests for the Cloudflare Workers AI provider and response normalization
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { CloudflareProvider } from '../providers/cloudflare';
import { ConfigurationError } from '../errors';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import type { LLMRequest } from '../types';

class TestableCloudflareProvider extends CloudflareProvider {
  exposeModelCapabilities() {
    return this.getModelCapabilities();
  }
}

describe('CloudflareProvider', () => {
  let provider: TestableCloudflareProvider;
  let mockAiRun: ReturnType<typeof vi.fn>;

  const testRequest: LLMRequest = {
    messages: [{ role: 'user', content: 'Hello, world!' }],
    model: '@cf/meta/llama-3.1-8b-instruct',
    maxTokens: 100,
    temperature: 0.7
  };

  const toolRequest: LLMRequest = {
    messages: [{ role: 'user', content: 'What is the weather in Austin?' }],
    model: '@cf/openai/gpt-oss-120b',
    maxTokens: 100,
    temperature: 0.2,
    tools: [
      {
        type: 'function',
        function: {
          name: 'get_weather',
          description: 'Get weather for a city',
          parameters: {
            type: 'object',
            properties: {
              city: { type: 'string' }
            },
            required: ['city']
          }
        }
      }
    ],
    toolChoice: 'auto'
  };

  beforeEach(() => {
    mockAiRun = vi.fn();
    defaultCircuitBreakerManager.resetAll();
    provider = new TestableCloudflareProvider({
      ai: { run: mockAiRun } as unknown as Ai
    });

    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Constructor', () => {
    it('should initialize with valid config and expose GPT-OSS tool support', () => {
      expect(provider.name).toBe('cloudflare');
      expect(provider.models).toContain('@cf/openai/gpt-oss-120b');
      expect(provider.supportsStreaming).toBe(true);
      expect(provider.supportsTools).toBe(true);
      expect(provider.supportsBatching).toBe(true);

      const capabilities = provider.exposeModelCapabilities();
      expect(capabilities['@cf/openai/gpt-oss-120b'].supportsTools).toBe(true);
      expect(capabilities['@cf/openai/gpt-oss-120b'].toolCalling).toBe(true);
    });

    it('should throw without an AI binding', () => {
      expect(() => new CloudflareProvider({})).toThrow(ConfigurationError);
      expect(() => new CloudflareProvider({})).toThrow('Cloudflare AI binding is required');
    });
  });

  describe('generateResponse', () => {
    it('should normalize the standard Workers AI response format', async () => {
      mockAiRun.mockResolvedValueOnce({
        response: 'Hello from Workers AI'
      });

      const response = await provider.generateResponse(testRequest);

      expect(mockAiRun).toHaveBeenCalledWith(
        '@cf/meta/llama-3.1-8b-instruct',
        expect.objectContaining({
          messages: [{ role: 'user', content: 'Hello, world!' }],
          max_tokens: 100,
          temperature: 0.7
        })
      );
      expect(response.message).toBe('Hello from Workers AI');
      expect(response.content).toBe('Hello from Workers AI');
      expect(response.provider).toBe('cloudflare');
      expect(response.model).toBe('@cf/meta/llama-3.1-8b-instruct');
      expect(response.finishReason).toBe('stop');
      expect(response.usage.inputTokens).toBeGreaterThan(0);
      expect(response.usage.outputTokens).toBeGreaterThan(0);
    });

    it('should pass OpenAI-format tools and parse chat completions tool calls', async () => {
      mockAiRun.mockResolvedValueOnce({
        id: 'chatcmpl-cf-123',
        model: '@cf/openai/gpt-oss-120b',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [
                {
                  id: 'call_weather',
                  type: 'function',
                  function: {
                    name: 'get_weather',
                    arguments: '{"city":"Austin"}'
                  }
                }
              ]
            },
            finish_reason: 'tool_calls'
          }
        ],
        usage: {
          prompt_tokens: 25,
          completion_tokens: 7,
          total_tokens: 32
        }
      });

      const response = await provider.generateResponse(toolRequest);
      const [model, body] = mockAiRun.mock.calls[0];

      expect(model).toBe('@cf/openai/gpt-oss-120b');
      expect(body.tools).toEqual(toolRequest.tools);
      expect(body.tool_choice).toBe('auto');
      expect(response.message).toBe('');
      expect(response.finishReason).toBe('tool_calls');
      expect(response.toolCalls).toEqual([
        {
          id: 'call_weather',
          type: 'function',
          function: {
            name: 'get_weather',
            arguments: '{"city":"Austin"}'
          }
        }
      ]);
      expect(response.usage.inputTokens).toBe(25);
      expect(response.usage.outputTokens).toBe(7);
      expect(response.usage.totalTokens).toBe(32);
    });

    it('should normalize Responses API output items into text and tool calls', async () => {
      mockAiRun.mockResolvedValueOnce({
        id: 'resp_123',
        model: '@cf/openai/gpt-oss-120b',
        output: [
          {
            type: 'message',
            role: 'assistant',
            content: [{ type: 'output_text', text: 'Calling weather tool.' }]
          },
          {
            type: 'function_call',
            id: 'fc_123',
            call_id: 'call_weather',
            name: 'get_weather',
            arguments: { city: 'Austin' }
          }
        ],
        usage: {
          input_tokens: 18,
          output_tokens: 9,
          total_tokens: 27
        }
      });

      const response = await provider.generateResponse(toolRequest);

      expect(response.message).toBe('Calling weather tool.');
      expect(response.finishReason).toBe('tool_calls');
      expect(response.toolCalls).toEqual([
        {
          id: 'call_weather',
          type: 'function',
          function: {
            name: 'get_weather',
            arguments: '{"city":"Austin"}'
          }
        }
      ]);
      expect(response.usage.inputTokens).toBe(18);
      expect(response.usage.outputTokens).toBe(9);
      expect(response.usage.totalTokens).toBe(27);
    });

    it('should reject tools on models without tool calling support', async () => {
      await expect(provider.generateResponse({
        ...toolRequest,
        model: '@cf/meta/llama-3.1-8b-instruct'
      })).rejects.toThrow("Model '@cf/meta/llama-3.1-8b-instruct' does not support tool calling");

      expect(mockAiRun).not.toHaveBeenCalled();
    });

    it('should preserve assistant tool calls and emit tool result messages in OpenAI format', async () => {
      mockAiRun.mockResolvedValueOnce({
        response: 'Done'
      });

      await provider.generateResponse({
        messages: [
          {
            role: 'assistant',
            content: '',
            toolCalls: [
              {
                id: 'call_weather',
                type: 'function',
                function: {
                  name: 'get_weather',
                  arguments: '{"city":"Austin"}'
                }
              }
            ],
            toolResults: [
              {
                id: 'call_weather',
                output: '{"temperature":72}'
              }
            ]
          }
        ],
        model: '@cf/openai/gpt-oss-120b'
      });

      const [, body] = mockAiRun.mock.calls[0];
      expect(body.messages).toEqual([
        {
          role: 'assistant',
          content: null,
          tool_calls: [
            {
              id: 'call_weather',
              type: 'function',
              function: {
                name: 'get_weather',
                arguments: '{"city":"Austin"}'
              }
            }
          ]
        },
        {
          role: 'tool',
          content: '{"temperature":72}',
          tool_call_id: 'call_weather'
        }
      ]);
    });
  });
});
