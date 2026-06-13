/**
 * Cloudflare Provider Tests
 * Tests for the Cloudflare Workers AI provider and response normalization
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { CloudflareProvider } from '../providers/cloudflare';
import { ConfigurationError } from '../errors';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { getStreamUsage } from '../utils/stream-usage';
import type { LLMRequest } from '../types';

class TestableCloudflareProvider extends CloudflareProvider {
  exposeModelCapabilities() {
    return this.getModelCapabilities();
  }
}

async function readStream(stream: ReadableStream<string>): Promise<string> {
  const reader = stream.getReader();
  let output = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    output += value;
  }

  return output;
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

    it('should recommend a BALANCED catalog model over tiny legacy defaults', () => {
      // Newer Workers AI catalog entries should outrank tiny legacy defaults
      // for normal balanced requests.
      expect(provider.getRecommendedModel({
        messages: [{ role: 'user', content: 'Hello' }],
        maxTokens: 200
      })).toBe('@cf/meta/llama-3.3-70b-instruct-fp8-fast');

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
      })).toBe('@cf/meta/llama-3.3-70b-instruct-fp8-fast');
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

    it('stringifies numeric response field — CF returns bare numbers for short numeric answers', async () => {
      // Regression: CF Workers AI returns {"response": 4} (not "4") when the model
      // produces a short numeric answer. Previously threw SCHEMA_DRIFT; now coerced.
      mockAiRun.mockResolvedValueOnce({ response: 4 });

      const response = await provider.generateResponse({
        ...testRequest,
        messages: [{ role: 'user', content: 'What is 2+2? Answer with only the number.' }]
      });

      expect(response.message).toBe('4');
      expect(response.content).toBe('4');
      expect(response.provider).toBe('cloudflare');
    });

    it('passes x-session-affinity in Workers AI run options for provider-prefix caching', async () => {
      mockAiRun.mockResolvedValueOnce({ response: 'Cached prefix.' });

      await provider.generateResponse({
        ...testRequest,
        cache: { strategy: 'provider-prefix', sessionId: 'agent-session-123' }
      });

      expect(mockAiRun).toHaveBeenCalledWith(
        '@cf/meta/llama-3.1-8b-instruct',
        expect.objectContaining({ messages: [{ role: 'user', content: 'Hello, world!' }] }),
        { extraHeaders: { 'x-session-affinity': 'agent-session-123' } }
      );
    });

    it('does not pass x-session-affinity for response-only cache strategy', async () => {
      mockAiRun.mockResolvedValueOnce({ response: 'Gateway cache only.' });

      await provider.generateResponse({
        ...testRequest,
        cache: { strategy: 'response', sessionId: 'agent-session-123' }
      });

      expect(mockAiRun.mock.calls[0]).toHaveLength(2);
    });

    it('passes configured AI Gateway binding options with request cache metadata overrides', async () => {
      provider = new TestableCloudflareProvider({
        ai: { run: mockAiRun } as unknown as Ai,
        gateway: {
          id: 'default',
          skipCache: true,
          cacheKey: 'default-key',
          metadata: { service: 'llm-providers' }
        }
      });
      mockAiRun.mockResolvedValueOnce({ response: 'Gateway routed.' });

      await provider.generateResponse({
        ...testRequest,
        requestId: 'req-123',
        tenantId: 'tenant-a',
        gatewayMetadata: {
          cacheKey: 'request-key',
          cacheTtl: 120,
          customMetadata: { routeClass: 'planning' }
        }
      });

      expect(mockAiRun.mock.calls[0][2]).toEqual({
        gateway: {
          id: 'default',
          skipCache: true,
          cacheKey: 'request-key',
          cacheTtl: 120,
          metadata: {
            service: 'llm-providers',
            routeClass: 'planning',
            llmRequestId: 'req-123',
            tenantId: 'tenant-a'
          }
        }
      });
    });

    it('normalizes Workers AI cached input tokens when returned in usage details', async () => {
      mockAiRun.mockResolvedValueOnce({
        response: 'Warm response.',
        usage: {
          prompt_tokens: 64,
          completion_tokens: 6,
          total_tokens: 70,
          prompt_tokens_details: { cached_tokens: 48 }
        }
      });

      const response = await provider.generateResponse(testRequest);

      expect(response.usage.cachedInputTokens).toBe(48);
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

    it('normalizes Cloudflare reasoning_content when content is null', async () => {
      mockAiRun.mockResolvedValueOnce({
        id: 'chatcmpl-cf-kimi',
        model: '@cf/moonshotai/kimi-k2.6',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: null,
              reasoning_content: 'Interim Kimi reasoning text'
            },
            finish_reason: 'length'
          }
        ],
        usage: {
          prompt_tokens: 17,
          completion_tokens: 32,
          total_tokens: 49
        }
      });

      const response = await provider.generateResponse({
        ...testRequest,
        model: '@cf/moonshotai/kimi-k2.6'
      });

      expect(response.message).toBe('Interim Kimi reasoning text');
      expect(response.content).toBe('Interim Kimi reasoning text');
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

  describe('streamResponse', () => {
    it('attaches Workers AI usage from object stream chunks', async () => {
      mockAiRun.mockResolvedValueOnce(new ReadableStream({
        start(controller) {
          controller.enqueue({
            response: 'Hello ',
            usage: { prompt_tokens: 8, completion_tokens: 2, total_tokens: 10 }
          });
          controller.enqueue({ response: 'stream' });
          controller.close();
        }
      }));

      const stream = await provider.streamResponse(testRequest);
      const usagePromise = getStreamUsage(stream);

      expect(await readStream(stream)).toBe('Hello stream');
      await expect(usagePromise).resolves.toMatchObject({
        inputTokens: 8,
        outputTokens: 2,
        totalTokens: 10
      });
    });

    it('passes x-session-affinity in Workers AI run options for streaming calls', async () => {
      mockAiRun.mockResolvedValueOnce(new ReadableStream({
        start(controller) {
          controller.enqueue('streamed');
          controller.close();
        }
      }));

      const stream = await provider.streamResponse({
        ...testRequest,
        cache: { strategy: 'both', sessionId: 'stream-session-123' }
      });

      expect(await readStream(stream)).toBe('streamed');
      expect(mockAiRun.mock.calls[0][2]).toEqual({
        extraHeaders: { 'x-session-affinity': 'stream-session-123' }
      });
    });

    it('attaches non-zero estimated usage when stream chunks omit provider usage', async () => {
      mockAiRun.mockResolvedValueOnce(new ReadableStream({
        start(controller) {
          controller.enqueue('Cloudflare ');
          controller.enqueue('stream');
          controller.close();
        }
      }));

      const stream = await provider.streamResponse(testRequest);
      const usagePromise = getStreamUsage(stream);

      expect(await readStream(stream)).toBe('Cloudflare stream');
      const usage = await usagePromise;
      expect(usage?.inputTokens).toBeGreaterThan(0);
      expect(usage?.outputTokens).toBeGreaterThan(0);
      expect(usage?.totalTokens).toBe((usage?.inputTokens ?? 0) + (usage?.outputTokens ?? 0));
    });

    it('normalizes chat-completion text when local REST shim returns JSON for a stream request', async () => {
      mockAiRun.mockResolvedValueOnce({
        id: 'chatcmpl-cf-kimi',
        model: '@cf/moonshotai/kimi-k2.6',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: 'cf-stream-ok'
            },
            finish_reason: 'stop'
          }
        ],
        usage: {
          prompt_tokens: 18,
          completion_tokens: 4,
          total_tokens: 22
        }
      });

      const stream = await provider.streamResponse({
        ...testRequest,
        model: '@cf/moonshotai/kimi-k2.6',
        stream: true
      });
      const usagePromise = getStreamUsage(stream);

      expect(await readStream(stream)).toBe('cf-stream-ok');
      await expect(usagePromise).resolves.toMatchObject({
        inputTokens: 18,
        outputTokens: 4,
        totalTokens: 22
      });
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

    it('uses raw binding format for llama-3.2-11b-vision-instruct (fixes silent empty response)', async () => {
      mockAiRun.mockResolvedValueOnce({ response: 'A delicious pasta dish.' });

      const result = await provider.generateResponse({
        model: '@cf/meta/llama-3.2-11b-vision-instruct',
        messages: [{ role: 'user', content: 'Describe this food image.' }],
        images: [{ data: 'QUJD', mimeType: 'image/jpeg' }],
        maxTokens: 512
      });

      const [modelArg, body] = mockAiRun.mock.calls[0];
      expect(modelArg).toBe('@cf/meta/llama-3.2-11b-vision-instruct');
      expect(Array.isArray(body.image)).toBe(true);
      expect(body.image).toHaveLength(3); // QUJD = 3 bytes: [65, 66, 67]
      expect(body.prompt).toBe('Describe this food image.');
      expect(body.max_tokens).toBe(512);
      expect(body.messages).toBeUndefined();
      expect(result.content).toBe('A delicious pasta dish.');
      expect(result.message).toBe('A delicious pasta dish.');
    });

    it('passes Workers AI run options through the llama-3.2 raw binding path', async () => {
      mockAiRun.mockResolvedValueOnce({ response: 'A cached image answer.' });

      await provider.generateResponse({
        model: '@cf/meta/llama-3.2-11b-vision-instruct',
        messages: [{ role: 'user', content: 'Describe this food image.' }],
        images: [{ data: 'QUJD', mimeType: 'image/jpeg' }],
        cache: { strategy: 'provider-prefix', sessionId: 'vision-session-123' }
      });

      expect(mockAiRun.mock.calls[0][2]).toEqual({
        extraHeaders: { 'x-session-affinity': 'vision-session-123' }
      });
    });

    it('prepends system prompt to raw binding prompt for llama-3.2', async () => {
      mockAiRun.mockResolvedValueOnce({ response: 'Pasta.' });

      await provider.generateResponse({
        model: '@cf/meta/llama-3.2-11b-vision-instruct',
        messages: [{ role: 'user', content: 'What is this?' }],
        images: [{ data: 'QUJD', mimeType: 'image/jpeg' }],
        systemPrompt: 'You are a food critic.',
      });

      const [, body] = mockAiRun.mock.calls[0];
      expect(body.prompt).toBe('You are a food critic.\n\nWhat is this?');
    });

    it('rejects multiple images on llama-3.2 raw binding with a clear error', async () => {
      await expect(
        provider.generateResponse({
          model: '@cf/meta/llama-3.2-11b-vision-instruct',
          messages: [{ role: 'user', content: 'compare' }],
          images: [
            { data: 'QUJD', mimeType: 'image/jpeg' },
            { data: 'REVG', mimeType: 'image/jpeg' }
          ]
        })
      ).rejects.toThrow(/supports exactly one image/);
    });

    it('extracts text from array-content user message for llama-3.2 raw binding', async () => {
      mockAiRun.mockResolvedValueOnce({ response: 'Spaghetti.' });

      await provider.generateResponse({
        model: '@cf/meta/llama-3.2-11b-vision-instruct',
        messages: [{
          role: 'user',
          content: [
            { type: 'text', text: 'What food is this?' },
            { type: 'text', text: 'Be brief.' }
          ] as unknown as string
        }],
        images: [{ data: 'QUJD', mimeType: 'image/jpeg' }]
      });

      const [, body] = mockAiRun.mock.calls[0];
      expect(body.prompt).toBe('What food is this? Be brief.');
    });

    it('defaults max_tokens to 512 when not specified for llama-3.2 raw binding', async () => {
      mockAiRun.mockResolvedValueOnce({ response: 'ok' });

      await provider.generateResponse({
        model: '@cf/meta/llama-3.2-11b-vision-instruct',
        messages: [{ role: 'user', content: 'x' }],
        images: [{ data: 'QUJD', mimeType: 'image/jpeg' }]
      });

      const [, body] = mockAiRun.mock.calls[0];
      expect(body.max_tokens).toBe(512);
    });

    it('accepts pre-formed data: URL for llama-3.2 raw binding', async () => {
      mockAiRun.mockResolvedValueOnce({ response: 'ok' });

      await provider.generateResponse({
        model: '@cf/meta/llama-3.2-11b-vision-instruct',
        messages: [{ role: 'user', content: 'x' }],
        images: [{ url: 'data:image/webp;base64,ZEFUQQ==' }]
      });

      const [, body] = mockAiRun.mock.calls[0];
      expect(Array.isArray(body.image)).toBe(true);
      expect(body.messages).toBeUndefined();
    });

    it('other vision models (gemma-4, llama-4-scout) still use chat/image_url format', async () => {
      mockAiRun.mockResolvedValueOnce({
        choices: [{ message: { content: 'A tomato.' }, finish_reason: 'stop' }]
      });

      await provider.generateResponse({
        model: '@cf/google/gemma-4-26b-a4b-it',
        messages: [{ role: 'user', content: 'What is in this image?' }],
        images: [{ data: 'QUJD', mimeType: 'image/png' }],
        maxTokens: 256
      });

      const [, body] = mockAiRun.mock.calls[0];
      expect(body.messages).toBeDefined();
      expect(body.image).toBeUndefined();
      const imagePart = body.messages[body.messages.length - 1].content[1];
      expect(imagePart.image_url.url).toBe('data:image/png;base64,QUJD');
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

  describe('LoRA (fine-tune) support', () => {
    it('forwards lora field to ai.run() when present', async () => {
      mockAiRun.mockResolvedValueOnce({ response: 'Fine-tuned output.' });

      await provider.generateResponse({
        model: '@cf/qwen/qwen1.5-7b-chat-awq',
        messages: [{ role: 'user', content: 'Hello' }],
        lora: '6d028a43-759e-417f-83fb-fa9b681d81f4',
        maxTokens: 50
      });

      const [, body] = mockAiRun.mock.calls[0];
      expect(body.lora).toBe('6d028a43-759e-417f-83fb-fa9b681d81f4');
    });

    it('does not include lora key when lora is absent', async () => {
      mockAiRun.mockResolvedValueOnce({ response: 'Normal output.' });

      await provider.generateResponse({
        model: '@cf/qwen/qwen1.5-7b-chat-awq',
        messages: [{ role: 'user', content: 'Hello' }],
        maxTokens: 50
      });

      const [, body] = mockAiRun.mock.calls[0];
      expect(Object.prototype.hasOwnProperty.call(body, 'lora')).toBe(false);
    });

    it('accepts lora as a model name string (not just UUID)', async () => {
      mockAiRun.mockResolvedValueOnce({ response: 'Named adapter output.' });

      await provider.generateResponse({
        model: '@cf/qwen/qwen1.5-7b-chat-awq',
        messages: [{ role: 'user', content: 'Hi' }],
        lora: 'my-custom-adapter',
        maxTokens: 20
      });

      const [, body] = mockAiRun.mock.calls[0];
      expect(body.lora).toBe('my-custom-adapter');
    });
  });

  describe('AiError wrapping', () => {
    it('wraps CF Bad-input AiError as InvalidRequestError', async () => {
      const aiErr = new Error('Bad input: Error: oneOf at \'/\' not met, 0 matches: Type mismatch of \'/messages/0/content\', \'array\' not in \'string\'');
      aiErr.name = 'AiError';
      mockAiRun.mockRejectedValueOnce(aiErr);

      const { InvalidRequestError } = await import('../errors');
      await expect(
        provider.generateResponse({
          model: '@cf/openai/gpt-oss-120b',
          messages: [{ role: 'user', content: 'test' }]
        })
      ).rejects.toThrow(InvalidRequestError);
    });

    it('does not wrap non-bad-input errors as InvalidRequestError', async () => {
      const serverErr = new Error('Workers AI: internal server error');
      serverErr.name = 'AiError';
      mockAiRun.mockRejectedValueOnce(serverErr);

      const { InvalidRequestError } = await import('../errors');
      await expect(
        provider.generateResponse({
          model: '@cf/openai/gpt-oss-120b',
          messages: [{ role: 'user', content: 'test' }]
        })
      ).rejects.not.toThrow(InvalidRequestError);
    });
  });

  describe('response envelope schema validation', () => {
    it('accepts a valid choices-based response', async () => {
      mockAiRun.mockResolvedValueOnce({
        id: 'res-1',
        choices: [{ message: { content: 'Hello!' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 }
      });

      const result = await provider.generateResponse({
        model: '@cf/openai/gpt-oss-120b',
        messages: [{ role: 'user', content: 'Hi' }]
      });

      expect(result.message).toBe('Hello!');
    });

    it('accepts a valid simple response shape', async () => {
      mockAiRun.mockResolvedValueOnce({ response: 'Simple.' });

      const result = await provider.generateResponse({
        model: '@cf/meta/llama-3.1-8b-instruct',
        messages: [{ role: 'user', content: 'Hi' }]
      });

      expect(result.message).toBe('Simple.');
    });

    it('throws SchemaDriftError when choices is not an array', async () => {
      // Simulate API drift where choices becomes an object instead of array
      mockAiRun.mockResolvedValueOnce({ choices: 'bad', id: 'x' });

      const { SchemaDriftError } = await import('../errors');
      await expect(
        provider.generateResponse({
          model: '@cf/openai/gpt-oss-120b',
          messages: [{ role: 'user', content: 'Hi' }]
        })
      ).rejects.toThrow(SchemaDriftError);
    });

    it('throws SchemaDriftError when choices[0].message is not an object', async () => {
      mockAiRun.mockResolvedValueOnce({
        choices: [{ message: 'not-an-object', finish_reason: 'stop' }]
      });

      const { SchemaDriftError } = await import('../errors');
      await expect(
        provider.generateResponse({
          model: '@cf/openai/gpt-oss-120b',
          messages: [{ role: 'user', content: 'Hi' }]
        })
      ).rejects.toThrow(SchemaDriftError);
    });
  });
});
