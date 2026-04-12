/**
 * Anthropic Provider
 * Implementation for Claude models with streaming and tools support
 */

import type {
  LLMRequest,
  LLMResponse,
  AnthropicConfig,
  ModelCapabilities,
  ProviderBalance,
  Tool,
  ToolCall
} from '../types';
import { BaseProvider } from './base';
import {
  LLMErrorFactory,
  AuthenticationError,
  ModelNotFoundError,
  RateLimitError
} from '../errors';
import { validateSchema, type SchemaField } from '../utils/schema-validator';

/**
 * Minimum envelope fields the Anthropic response parser reads. Changing this
 * list is a contract change — keep it in sync with formatResponse below.
 *
 * Content blocks are validated as a discriminated union on `type`. Unknown
 * block types are forward-compatible (skipped) so Anthropic adding a new
 * block variant doesn't instantly break the fallback routing on every deploy.
 */
const ANTHROPIC_RESPONSE_SCHEMA: SchemaField[] = [
  { path: 'id', type: 'string' },
  {
    path: 'content',
    type: 'array',
    items: {
      discriminator: 'type',
      variants: {
        text: [
          { path: 'text', type: 'string' },
        ],
        tool_use: [
          { path: 'id', type: 'string' },
          { path: 'name', type: 'string' },
          { path: 'input', type: 'object' },
        ],
      },
    },
  },
  { path: 'model', type: 'string' },
  { path: 'stop_reason', type: 'string' },
  { path: 'stop_sequence', type: 'string', optional: true },
  { path: 'usage', type: 'object' },
  { path: 'usage.input_tokens', type: 'number' },
  { path: 'usage.output_tokens', type: 'number' },
];

interface AnthropicContentBlock {
  type: 'text' | 'tool_use' | 'tool_result' | 'image';
  text?: string;
  id?: string;
  name?: string;
  input?: Record<string, unknown>;
  content?: string;
  is_error?: boolean;
  source?: {
    type: 'base64';
    media_type: string;
    data: string;
  };
}

interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: string | AnthropicContentBlock[];
}

interface AnthropicTool {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

type AnthropicToolChoice = 'auto' | 'none' | { type: 'function'; function: { name: string } };

interface AnthropicRequest {
  model: string;
  messages: AnthropicMessage[];
  system?: string;
  max_tokens: number;
  temperature?: number;
  stream?: boolean;
  tools?: AnthropicTool[];
  tool_choice?: AnthropicToolChoice;
}

interface AnthropicResponseContentBlock {
  type: 'text' | 'tool_use';
  text?: string;
  id?: string;
  name?: string;
  input?: Record<string, unknown>;
}

interface AnthropicResponse {
  id: string;
  type: string;
  role: string;
  content: AnthropicResponseContentBlock[];
  model: string;
  stop_reason: 'end_turn' | 'max_tokens' | 'stop_sequence' | 'tool_use';
  stop_sequence?: string;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
}

export class AnthropicProvider extends BaseProvider {
  name = 'anthropic';
  models = [
    'claude-opus-4-6-20250618',
    'claude-sonnet-4-6-20250618',
    'claude-opus-4-20250618',
    'claude-sonnet-4-20250514',
    'claude-haiku-4-5-20251001',
    'claude-3-7-sonnet-20250219',
    'claude-3-5-sonnet-20241022',
    'claude-3-5-haiku-20241022',
    'claude-3-opus-20240229',
    'claude-3-sonnet-20240229',
    'claude-3-haiku-20240307'
  ];
  supportsStreaming = true;
  supportsTools = true;
  supportsBatching = false;
  supportsVision = true;

  private apiKey: string;
  private baseUrl: string;
  private version: string;

  constructor(config: AnthropicConfig) {
    super(config);

    if (!config.apiKey) {
      throw new AuthenticationError('anthropic', 'Anthropic API key is required');
    }

    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl || 'https://api.anthropic.com';
    this.version = config.version || '2023-06-01';
  }

  async generateResponse(request: LLMRequest): Promise<LLMResponse> {
    this.validateRequest(request);

    const startTime = Date.now();
    const jsonMode = request.response_format?.type === 'json_object';

    try {
      const response = await this.executeWithResiliency(async () => {
        const anthropicRequest = this.formatRequest(request);
        const httpResponse = await this.makeAnthropicRequest('/v1/messages', anthropicRequest, 'POST', request);

        if (!httpResponse.ok) {
          throw await LLMErrorFactory.fromFetchResponse('anthropic', httpResponse);
        }

        const data = await httpResponse.json() as unknown;
        validateSchema('anthropic', data, ANTHROPIC_RESPONSE_SCHEMA);
        const formatted = this.formatResponse(data as AnthropicResponse, Date.now() - startTime);

        // Restore the prefilled '{' consumed by the assistant turn,
        // but only if the response doesn't already start with one
        if (jsonMode && !formatted.message.startsWith('{')) {
          const restored = '{' + formatted.message;
          formatted.message = restored;
          formatted.content = restored;
        }

        return formatted;
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
    const model = request.model || 'claude-haiku-4-5-20251001';
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
      // Lightweight reachability check — HEAD to base URL avoids burning tokens
      const response = await this.makeAnthropicRequest('/v1/messages', null, 'OPTIONS');
      // Anthropic returns 405 for OPTIONS but that confirms the API is reachable
      return response.status !== 0;
    } catch {
      return false;
    }
  }

  async getProviderBalance(): Promise<ProviderBalance> {
    try {
      const response = await this.makeAnthropicRequest('/v1/organizations/cost_report', null, 'GET');
      if (!response.ok) {
        return {
          provider: this.name,
          status: 'unavailable',
          source: 'provider_api',
          message: `Anthropic usage API returned HTTP ${response.status}`
        };
      }

      const raw = await response.json();
      return {
        provider: this.name,
        status: 'available',
        source: 'provider_api',
        raw
      };
    } catch (error) {
      return {
        provider: this.name,
        status: 'error',
        source: 'provider_api',
        message: (error as Error).message
      };
    }
  }

  protected getModelCapabilities(): Record<string, ModelCapabilities> {
    return {
      'claude-opus-4-6-20250618': {
        maxContextLength: 200000,
        supportsStreaming: true,
        supportsTools: true,
        supportsVision: true,
        supportsBatching: false,
        inputTokenCost: 0.015, // $15 per 1M tokens
        outputTokenCost: 0.075, // $75 per 1M tokens
        description: 'Claude Opus 4.6 - Latest, highest intelligence and capability'
      },
      'claude-sonnet-4-6-20250618': {
        maxContextLength: 200000,
        supportsStreaming: true,
        supportsTools: true,
        supportsVision: true,
        supportsBatching: false,
        inputTokenCost: 0.003, // $3 per 1M tokens
        outputTokenCost: 0.015, // $15 per 1M tokens
        description: 'Claude Sonnet 4.6 - Latest balanced performance model'
      },
      'claude-opus-4-20250618': {
        maxContextLength: 200000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.015, // $15 per 1M tokens
        outputTokenCost: 0.075, // $75 per 1M tokens
        description: 'Claude Opus 4 - Highest level of intelligence and capability'
      },
      'claude-sonnet-4-20250514': {
        maxContextLength: 200000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.003, // $3 per 1M tokens
        outputTokenCost: 0.015, // $15 per 1M tokens
        description: 'Claude Sonnet 4 - High intelligence and balanced performance'
      },
      'claude-haiku-4-5-20251001': {
        maxContextLength: 200000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.001, // $1 per 1M tokens
        outputTokenCost: 0.005, // $5 per 1M tokens
        description: 'Claude Haiku 4.5 - Fast and cost-effective'
      },
      'claude-3-7-sonnet-20250219': {
        maxContextLength: 200000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.003, // $3 per 1M tokens
        outputTokenCost: 0.015, // $15 per 1M tokens
        description: 'Claude 3.7 Sonnet - Extended thinking with hybrid reasoning'
      },
      'claude-3-5-sonnet-20241022': {
        maxContextLength: 200000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.003, // $3 per 1M tokens
        outputTokenCost: 0.015, // $15 per 1M tokens
        description: 'Claude 3.5 Sonnet - Latest high-performance model'
      },
      'claude-3-5-haiku-20241022': {
        maxContextLength: 200000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.00025, // $0.25 per 1M tokens
        outputTokenCost: 0.00125, // $1.25 per 1M tokens
        description: 'Claude 3.5 Haiku - Fast and cost-effective'
      },
      'claude-3-opus-20240229': {
        maxContextLength: 200000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.015, // $15 per 1M tokens
        outputTokenCost: 0.075, // $75 per 1M tokens
        description: 'Claude 3 Opus - Most capable model'
      },
      'claude-3-sonnet-20240229': {
        maxContextLength: 200000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.003, // $3 per 1M tokens
        outputTokenCost: 0.015, // $15 per 1M tokens
        description: 'Claude 3 Sonnet - Balanced performance'
      },
      'claude-3-haiku-20240307': {
        maxContextLength: 200000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.00025, // $0.25 per 1M tokens
        outputTokenCost: 0.00125, // $1.25 per 1M tokens
        description: 'Claude 3 Haiku - Fast and economical'
      }
    };
  }

  private async makeAnthropicRequest(
    endpoint: string,
    body: AnthropicRequest | null,
    method: string = 'POST',
    request?: LLMRequest
  ): Promise<Response> {
    const headers: Record<string, string> = {
      'x-api-key': this.apiKey,
      'anthropic-version': this.version,
      'Content-Type': 'application/json',
      ...this.getAIGatewayHeaders(request)
    };

    const options: RequestInit = {
      method,
      headers
    };

    if (body && method !== 'GET') {
      options.body = JSON.stringify(body);
    }

    return this.makeRequest(`${this.baseUrl}${endpoint}`, options);
  }

  private formatRequest(request: LLMRequest): AnthropicRequest {
    const messages: AnthropicMessage[] = [];
    const jsonMode = request.response_format?.type === 'json_object';

    // Convert messages (skip system messages as they go in separate field)
    for (const message of request.messages) {
      if (message.role === 'system') continue;

      const anthropicMessage: AnthropicMessage = {
        role: message.role as 'user' | 'assistant',
        content: this.formatMessageContent(message.content, message.role === 'user' ? request.images : undefined)
      };

      // Handle tool calls and results
      if (message.toolCalls && message.toolCalls.length > 0) {
        anthropicMessage.content = message.toolCalls.map(tc => ({
          type: 'tool_use' as const,
          id: tc.id,
          name: tc.function.name,
          input: JSON.parse(tc.function.arguments) as Record<string, unknown>
        }));
      }

      if (message.toolResults && message.toolResults.length > 0) {
        anthropicMessage.content = message.toolResults.map(tr => ({
          type: 'tool_result' as const,
          id: tr.id,
          content: tr.output,
          is_error: !!tr.error
        }));
      }

      messages.push(anthropicMessage);
    }

    // Add prefilled assistant message for JSON mode
    if (jsonMode) {
      messages.push({ role: 'assistant', content: '{' });
    }

    const anthropicRequest: AnthropicRequest = {
      model: request.model || 'claude-haiku-4-5-20251001',
      messages,
      max_tokens: request.maxTokens || 1000,
      temperature: request.temperature,
      stream: request.stream
    };

    // Add system prompt if provided
    const systemMessage = request.messages.find(m => m.role === 'system');
    const jsonInstruction = '\n\nYou must respond with valid JSON only. No markdown fences, no commentary, no text outside the JSON.';

    if (request.systemPrompt) {
      anthropicRequest.system = jsonMode
        ? request.systemPrompt + jsonInstruction
        : request.systemPrompt;
    } else if (systemMessage) {
      anthropicRequest.system = jsonMode
        ? systemMessage.content + jsonInstruction
        : systemMessage.content;
    } else if (jsonMode) {
      anthropicRequest.system = jsonInstruction.trimStart();
    }

    // Add tools if provided
    if (request.tools && request.tools.length > 0) {
      anthropicRequest.tools = request.tools.map(tool => ({
        name: tool.function.name,
        description: tool.function.description,
        input_schema: tool.function.parameters as Record<string, unknown>
      }));

      if (request.toolChoice) {
        anthropicRequest.tool_choice = request.toolChoice;
      }
    }

    return anthropicRequest;
  }

  private formatMessageContent(
    text: string,
    images?: LLMRequest['images']
  ): string | AnthropicContentBlock[] {
    if (!images || images.length === 0) {
      return text;
    }

    return [
      { type: 'text', text },
      ...images.map(image => {
        if (image.url) {
          return {
            type: 'text' as const,
            text: `[Image URL: ${image.url}]`
          };
        }

        return {
          type: 'image' as const,
          source: {
            type: 'base64' as const,
            media_type: image.mimeType || 'image/jpeg',
            data: image.data || ''
          }
        };
      })
    ];
  }

  private formatResponse(
    data: AnthropicResponse,
    responseTime: number
  ): LLMResponse {
    // Extract text content from response
    const textContent = data.content
      .filter(block => block.type === 'text')
      .map(block => block.text)
      .join('');

    const usage = {
      inputTokens: data.usage.input_tokens,
      outputTokens: data.usage.output_tokens,
      totalTokens: data.usage.input_tokens + data.usage.output_tokens,
      cost: this.calculateCost(
        data.usage.input_tokens,
        data.usage.output_tokens,
        data.model
      )
    };

    const response: LLMResponse = {
      id: data.id,
      message: textContent,
      content: textContent,
      usage,
      model: data.model,
      provider: this.name,
      responseTime,
      finishReason: this.mapStopReason(data.stop_reason),
      metadata: {
        stopSequence: data.stop_sequence
      }
    };

    // Extract tool calls if present. id/name/input are guaranteed non-null
    // for tool_use blocks by ANTHROPIC_RESPONSE_SCHEMA's discriminated-union
    // validation — if they were missing, validateSchema would have thrown
    // SchemaDriftError before we got here.
    const toolUses = data.content.filter(block => block.type === 'tool_use');
    if (toolUses.length > 0) {
      const raw: ToolCall[] = toolUses.map(tool => ({
        id: tool.id as string,
        type: 'function' as const,
        function: {
          name: tool.name as string,
          arguments: JSON.stringify(tool.input)
        }
      }));
      response.toolCalls = this.validateToolCalls(raw);
    }

    return response;
  }

  private mapStopReason(stopReason: string): 'stop' | 'length' | 'tool_calls' | 'content_filter' {
    switch (stopReason) {
      case 'end_turn':
        return 'stop';
      case 'max_tokens':
        return 'length';
      case 'tool_use':
        return 'tool_calls';
      default:
        return 'stop';
    }
  }

  /**
   * Stream response support
   */
  async streamResponse(request: LLMRequest): Promise<ReadableStream<string>> {
    this.validateRequest(request);

    const anthropicRequest = { ...this.formatRequest(request), stream: true };

    return new ReadableStream({
      start: async (controller) => {
        try {
          const response = await this.makeAnthropicRequest('/v1/messages', anthropicRequest, 'POST', request);

          if (!response.ok) {
            throw await LLMErrorFactory.fromFetchResponse('anthropic', response);
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

                  if (parsed.type === 'content_block_delta' && parsed.delta?.text) {
                    controller.enqueue(parsed.delta.text);
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
   * Tool usage support
   */
  async generateWithTools(
    request: LLMRequest & { tools: Tool[] }
  ): Promise<LLMResponse> {
    return this.generateResponse(request);
  }
}
