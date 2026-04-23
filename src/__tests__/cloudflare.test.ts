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

    it('should recommend modern active models instead of legacy tiny defaults', () => {
      expect(provider.getRecommendedModel({
        messages: [{ role: 'user', content: 'Hello' }],
        maxTokens: 200
      })).toBe('@cf/google/gemma-4-26b-a4b-it');

      expect(provider.getRecommendedModel({
        messages: [{ role: 'user', content: 'Use the weather tool' }],
        maxTokens: 400,
        tools: [{
          type: 'function',
          function: {
            name: 'get_weather',
            description: 'Get weather',
            parameters: { type: 'object' }
          }
        }]
      })).toBe('@cf/openai/gpt-oss-120b');
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

  describe('vision', () => {
    it('advertises vision capability on the provider', () => {
      expect(provider.supportsVision).toBe(true);
    });

    it('marks Gemma 4, Llama 4 Scout, and Llama 3.2 Vision as vision-capable', () => {
      const capabilities = provider.exposeModelCapabilities();
      expect(capabilities['@cf/google/gemma-4-26b-a4b-it'].supportsVision).toBe(true);
      expect(capabilities['@cf/meta/llama-4-scout-17b-16e-instruct'].supportsVision).toBe(true);
      expect(capabilities['@cf/meta/llama-3.2-11b-vision-instruct'].supportsVision).toBe(true);
    });

    it('attaches images to the last user message as OpenAI image_url parts', async () => {
      mockAiRun.mockResolvedValueOnce({
        choices: [{ message: { role: 'assistant', content: 'A ripe tomato.' }, finish_reason: 'stop' }]
      });

      await provider.generateResponse({
        model: '@cf/google/gemma-4-26b-a4b-it',
        messages: [{ role: 'user', content: 'What is in this image?' }],
        images: [{ data: 'QUJD', mimeType: 'image/png' }],
        maxTokens: 256
      });

      const [modelArg, body] = mockAiRun.mock.calls[0];
      expect(modelArg).toBe('@cf/google/gemma-4-26b-a4b-it');
      expect(body.messages).toHaveLength(1);
      const userMsg = body.messages[0];
      expect(userMsg.role).toBe('user');
      expect(Array.isArray(userMsg.content)).toBe(true);
      expect(userMsg.content[0]).toEqual({ type: 'text', text: 'What is in this image?' });
      expect(userMsg.content[1]).toEqual({
        type: 'image_url',
        image_url: { url: 'data:image/png;base64,QUJD' }
      });
    });

    it('appends multiple images as separate image_url parts', async () => {
      mockAiRun.mockResolvedValueOnce({
        choices: [{ message: { content: 'Two tomatoes.' }, finish_reason: 'stop' }]
      });

      await provider.generateResponse({
        model: '@cf/meta/llama-4-scout-17b-16e-instruct',
        messages: [{ role: 'user', content: 'compare' }],
        images: [
          { data: 'QQ==', mimeType: 'image/jpeg' },
          { data: 'Qg==', mimeType: 'image/jpeg' }
        ]
      });

      const [, body] = mockAiRun.mock.calls[0];
      const content = body.messages[body.messages.length - 1].content;
      expect(content.filter((p: { type: string }) => p.type === 'image_url')).toHaveLength(2);
    });

    it('accepts pre-formed data: URLs via image.url', async () => {
      mockAiRun.mockResolvedValueOnce({
        choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }]
      });

      await provider.generateResponse({
        model: '@cf/meta/llama-3.2-11b-vision-instruct',
        messages: [{ role: 'user', content: 'x' }],
        images: [{ url: 'data:image/webp;base64,ZEFUQQ==' }]
      });

      const [, body] = mockAiRun.mock.calls[0];
      const imagePart = body.messages[0].content[1];
      expect(imagePart.image_url.url).toBe('data:image/webp;base64,ZEFUQQ==');
    });

    it('rejects HTTP image URLs (requires base64 bytes)', async () => {
      await expect(
        provider.generateResponse({
          model: '@cf/google/gemma-4-26b-a4b-it',
          messages: [{ role: 'user', content: 'x' }],
          images: [{ url: 'https://example.com/img.jpg' }]
        })
      ).rejects.toThrow(/HTTP image URLs are not supported/);
    });

    it('rejects images on non-vision models with a helpful error', async () => {
      await expect(
        provider.generateResponse({
          model: '@cf/meta/llama-3.1-8b-instruct',
          messages: [{ role: 'user', content: 'x' }],
          images: [{ data: 'QUJD', mimeType: 'image/png' }]
        })
      ).rejects.toThrow(/does not support image input/);
    });
  });
});
