/**
 * OpenAI Provider
 * Implementation for OpenAI GPT models with streaming and tools support
 */

import type { LLMRequest, LLMResponse, OpenAIConfig, ModelCapabilities, ToolCall, Tool } from '../types';
import { BaseProvider } from './base';
import {
  LLMErrorFactory,
  AuthenticationError,
  ModelNotFoundError,
  RateLimitError,
  SchemaDriftError
} from '../errors';
import { getProviderDefaultModel } from '../model-catalog';
import { validateSchema, type SchemaField } from '../utils/schema-validator';

// Minimum envelope `formatResponse` reads. `tool_calls` uses a discriminated
// union (single `function` variant today) so an additive new tool type upstream
// is forward-compat rather than drift. Empty `choices` is surfaced as drift at
// the `choices[0]` path rather than a bare throw, so it routes through the
// fallback/hook machinery like every other envelope failure.
const OPENAI_RESPONSE_SCHEMA: SchemaField[] = [
  { path: 'id', type: 'string' },
  { path: 'model', type: 'string' },
  {
    path: 'choices',
    type: 'array',
    items: {
      shape: [
        { path: 'message', type: 'object' },
        { path: 'message.content', type: 'string-or-null' },
        { path: 'finish_reason', type: 'string' },
        {
          path: 'message.tool_calls',
          type: 'array',
          optional: true,
          items: {
            discriminator: 'type',
            variants: {
              function: [
                { path: 'id', type: 'string' },
                { path: 'function.name', type: 'string' },
                { path: 'function.arguments', type: 'string' },
              ],
            },
          },
        },
      ],
    },
  },
  { path: 'usage', type: 'object' },
  { path: 'usage.prompt_tokens', type: 'number' },
  { path: 'usage.completion_tokens', type: 'number' },
  { path: 'usage.total_tokens', type: 'number' },
];

interface OpenAIToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

interface OpenAIContentPart {
  type: 'text' | 'image_url';
  text?: string;
  image_url?: { url: string };
}

interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null | OpenAIContentPart[];
  name?: string;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
}

interface OpenAITool {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

type OpenAIToolChoice = 'auto' | 'none' | { type: 'function'; function: { name: string } };

interface OpenAIRequest {
  model: string;
  messages: OpenAIMessage[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  tools?: OpenAITool[];
  tool_choice?: OpenAIToolChoice;
  response_format?: { type: 'json_object' | 'text' };
  seed?: number;
}

interface OpenAIResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string | null;
      tool_calls?: OpenAIToolCall[];
    };
    finish_reason: 'stop' | 'length' | 'tool_calls' | 'content_filter';
    logprobs?: Record<string, unknown> | null;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  system_fingerprint?: string;
}

export class OpenAIProvider extends BaseProvider {
  name = 'openai';
  models = [
    'gpt-4o-mini',
    'gpt-4-turbo',
    'gpt-4',
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-16k'
  ];
  supportsStreaming = true;
  supportsTools = true;
  supportsBatching = false;
  supportsVision = true;

  private apiKey: string;
  private baseUrl: string;
  private organization?: string;
  private project?: string;

  constructor(config: OpenAIConfig) {
    super(config);

    if (!config.apiKey) {
      throw new AuthenticationError('openai', 'OpenAI API key is required');
    }

    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl || 'https://api.openai.com/v1';
    this.organization = config.organization;
    this.project = config.project;
  }

  async generateResponse(request: LLMRequest): Promise<LLMResponse> {
    this.validateRequest(request);

    const startTime = Date.now();

    try {
      const response = await this.executeWithResiliency(async () => {
        const openaiRequest = this.formatRequest(request);
        const httpResponse = await this.makeOpenAIRequest('/chat/completions', openaiRequest, 'POST', request);

        if (!httpResponse.ok) {
          throw await LLMErrorFactory.fromFetchResponse('openai', httpResponse);
        }

        const data = await httpResponse.json() as unknown;
        validateSchema('openai', data, OPENAI_RESPONSE_SCHEMA);
        return this.formatResponse(data as OpenAIResponse, Date.now() - startTime);
      });

      this.updateMetrics(response.responseTime, true, response.usage.cost);
      this.logRequest(request, response);

      return response;
    } catch (error) {
      const responseTime = Date.now() - startTime;
      this.updateMetrics(responseTime, false);
      this.logRequest(request, undefined, error as Error);
      throw error;
    }
  }

  validateConfig(): boolean {
    return !!(this.apiKey && this.baseUrl);
  }

  getModels(): string[] {
    return [...this.models];
  }

  estimateCost(request: LLMRequest): number {
    const model = request.model || this.getDefaultModel(request);
    const capabilities = this.getModelCapabilities()[model];

    if (!capabilities) return 0;

    // Estimate input tokens (rough approximation)
    const inputTokens = request.messages.reduce((sum, msg) =>
      sum + Math.ceil(msg.content.length / 4), 0
    );
    const outputTokens = request.maxTokens || 1000;

    return this.calculateCost(inputTokens, outputTokens, model);
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.makeOpenAIRequest('/models', null, 'GET');
      return response.ok;
    } catch {
      return false;
    }
  }

  protected getModelCapabilities(): Record<string, ModelCapabilities> {
    return {
      'gpt-4o-mini': {
        maxContextLength: 128000,
        supportsStreaming: true,
        supportsTools: true,
        supportsVision: true,
        supportsBatching: false,
        inputTokenCost: 0.00015, // $0.15 per 1M tokens
        outputTokenCost: 0.0006, // $0.60 per 1M tokens
        description: 'GPT-4 Omni Mini - Cost-effective version'
      },
      'gpt-4-turbo': {
        maxContextLength: 128000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.01, // $10 per 1M tokens
        outputTokenCost: 0.03, // $30 per 1M tokens
        description: 'GPT-4 Turbo - High performance model'
      },
      'gpt-4': {
        maxContextLength: 8192,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.03, // $30 per 1M tokens
        outputTokenCost: 0.06, // $60 per 1M tokens
        description: 'GPT-4 - Original high-capability model'
      },
      'gpt-3.5-turbo': {
        maxContextLength: 16385,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.0005, // $0.50 per 1M tokens
        outputTokenCost: 0.0015, // $1.50 per 1M tokens
        description: 'GPT-3.5 Turbo - Fast and cost-effective'
      },
      'gpt-3.5-turbo-16k': {
        maxContextLength: 16385,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.001, // $1 per 1M tokens
        outputTokenCost: 0.002, // $2 per 1M tokens
        description: 'GPT-3.5 Turbo 16k - Extended context'
      }
    };
  }

  private async makeOpenAIRequest(
    endpoint: string,
    body: OpenAIRequest | null,
    method: string = 'POST',
    request?: LLMRequest
  ): Promise<Response> {
    const headers: Record<string, string> = {
      'Authorization': `Bearer ${this.apiKey}`,
      'Content-Type': 'application/json',
      ...this.getAIGatewayHeaders(request)
    };

    if (this.organization) {
      headers['OpenAI-Organization'] = this.organization;
    }

    if (this.project) {
      headers['OpenAI-Project'] = this.project;
    }

    const options: RequestInit = {
      method,
      headers
    };

    if (body && method !== 'GET') {
      options.body = JSON.stringify(body);
    }

    return this.makeRequest(`${this.baseUrl}${endpoint}`, options);
  }

  private formatRequest(request: LLMRequest): OpenAIRequest {
    const messages: OpenAIMessage[] = [];

    // Add system prompt if provided
    if (request.systemPrompt) {
      messages.push({
        role: 'system',
        content: request.systemPrompt
      });
    }

    // Convert messages
    for (const message of request.messages) {
      if (message.role === 'system' && request.systemPrompt) {
        continue; // Skip if system prompt already added
      }

      const openaiMessage: OpenAIMessage = {
        role: message.role as OpenAIMessage['role'],
        content: this.formatMessageContent(message.content, message.role === 'user' ? request.images : undefined)
      };

      // Add tool calls if present
      if (message.toolCalls && message.toolCalls.length > 0) {
        openaiMessage.tool_calls = message.toolCalls.map(tc => ({
          id: tc.id,
          type: tc.type,
          function: tc.function
        }));
        openaiMessage.content = null; // Content must be null when tool_calls present
      }

      if (message.toolResults && message.toolResults.length > 0) {
        for (const toolResult of message.toolResults) {
          messages.push({
            role: 'tool',
            content: toolResult.error
              ? JSON.stringify({ output: toolResult.output, error: toolResult.error })
              : toolResult.output,
            tool_call_id: toolResult.id
          });
        }
        continue;
      }

      messages.push(openaiMessage);
    }

    const openaiRequest: OpenAIRequest = {
      model: request.model || this.getDefaultModel(request),
      messages,
      temperature: request.temperature,
      max_tokens: request.maxTokens,
      stream: request.stream,
      seed: request.seed
    };

    // Add tools if provided
    if (request.tools && request.tools.length > 0) {
      openaiRequest.tools = request.tools.map(tool => ({
        type: tool.type,
        function: tool.function
      }));

      if (request.toolChoice) {
        openaiRequest.tool_choice = request.toolChoice;
      }
    }

    // Pass through response_format if provided
    if (request.response_format) {
      openaiRequest.response_format = request.response_format;
    }

    return openaiRequest;
  }

  private formatMessageContent(
    text: string,
    images?: LLMRequest['images']
  ): OpenAIMessage['content'] {
    if (!images || images.length === 0) {
      return text;
    }

    return [
      { type: 'text', text },
      ...images.map(image => ({
        type: 'image_url' as const,
        image_url: {
          url: image.url || `data:${image.mimeType};base64,${image.data}`
        }
      }))
    ];
  }

  private formatResponse(
    data: OpenAIResponse,
    responseTime: number
  ): LLMResponse {
    const choice = data.choices[0];
    if (!choice) {
      throw new SchemaDriftError('openai', 'choices[0]', 'object', 'undefined');
    }

    const content = choice.message.content || '';
    const usage = {
      inputTokens: data.usage.prompt_tokens,
      outputTokens: data.usage.completion_tokens,
      totalTokens: data.usage.total_tokens,
      cost: this.calculateCost(
        data.usage.prompt_tokens,
        data.usage.completion_tokens,
        data.model
      )
    };

    const response: LLMResponse = {
      id: data.id,
      message: content,
      content,
      usage,
      model: data.model,
      provider: this.name,
      responseTime,
      finishReason: choice.finish_reason,
      metadata: {
        systemFingerprint: data.system_fingerprint,
        created: data.created
      }
    };

    // Add tool calls if present (validated at provider boundary)
    if (choice.message.tool_calls && choice.message.tool_calls.length > 0) {
      const raw: ToolCall[] = choice.message.tool_calls.map(tc => ({
        id: tc.id,
        type: tc.type,
        function: tc.function
      }));
      response.toolCalls = this.validateToolCalls(raw);
    }

    return response;
  }

  /**
   * Stream response support
   */
  async streamResponse(request: LLMRequest): Promise<ReadableStream<string>> {
    this.validateRequest(request);

    const openaiRequest = { ...this.formatRequest(request), stream: true };

    return new ReadableStream({
      start: async (controller) => {
        try {
          const response = await this.makeOpenAIRequest('/chat/completions', openaiRequest, 'POST', request);

          if (!response.ok) {
            throw await LLMErrorFactory.fromFetchResponse('openai', response);
          }

          const reader = response.body?.getReader();
          if (!reader) {
            throw new Error('No response body');
          }

          const decoder = new TextDecoder();
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6);

                if (data === '[DONE]') {
                  controller.close();
                  return;
                }

                try {
                  const parsed = JSON.parse(data);
                  const sseContent = parsed.choices?.[0]?.delta?.content;

                  if (sseContent) {
                    controller.enqueue(sseContent);
                  }
                } catch {
                  // Malformed SSE chunk — skip silently
                }
              }
            }
          }

          controller.close();
        } catch (error) {
          controller.error(error);
        }
      }
    });
  }

  /**
   * Batch processing support
   */
  async processBatch(requests: LLMRequest[]): Promise<LLMResponse[]> {
    // OpenAI doesn't have native batch API, so we process sequentially
    // In production, you might want to implement concurrent processing with rate limiting
    const responses: LLMResponse[] = [];

    for (const request of requests) {
      try {
        const response = await this.generateResponse(request);
        responses.push(response);
      } catch (error) {
        // Handle individual request failures
        responses.push({
          message: '',
          usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0, cost: 0 },
          model: request.model || this.getDefaultModel(request),
          provider: this.name,
          responseTime: 0,
          metadata: { error: (error as Error).message }
        });
      }
    }

    return responses;
  }

  private getDefaultModel(request: LLMRequest): string {
    return getProviderDefaultModel('openai', request);
  }
}
