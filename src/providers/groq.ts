/**
 * Groq Provider
 * Implementation for Groq fast inference models (OpenAI-compatible API)
 */

import type { LLMRequest, LLMResponse, GroqConfig, ModelCapabilities, ProviderBalance, ToolCall } from '../types';
import { BaseProvider } from './base';
import {
  LLMErrorFactory,
  AuthenticationError,
  ConfigurationError,
  SchemaDriftError
} from '../errors';
import { validateSchema, type SchemaField } from '../utils/schema-validator';

// Groq serves the OpenAI /chat/completions contract — same envelope shape as
// OpenAI. Kept as a separate constant (not imported from openai.ts) because
// each provider's envelope is an independent API surface; shared drift would
// be a correlated outage signal, not a single bug.
const GROQ_RESPONSE_SCHEMA: SchemaField[] = [
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

interface GroqMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: Array<{ id: string; type: 'function'; function: { name: string; arguments: string } }>;
  tool_call_id?: string;
}

interface GroqTool {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

interface GroqRequest {
  model: string;
  messages: GroqMessage[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  response_format?: { type: 'json_object' | 'text' };
  tools?: GroqTool[];
  tool_choice?: 'auto' | 'none' | { type: 'function'; function: { name: string } };
  seed?: number;
}

interface GroqResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string | null;
      tool_calls?: Array<{
        id: string;
        type: 'function';
        function: { name: string; arguments: string };
      }>;
    };
    finish_reason: 'stop' | 'length' | 'content_filter' | 'tool_calls';
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  system_fingerprint?: string;
}

// Models that support tool calling
const TOOL_CAPABLE_MODELS = new Set([
  'openai/gpt-oss-120b',
  'llama-3.3-70b-versatile',
]);

export class GroqProvider extends BaseProvider {
  name = 'groq';
  models = [
    'llama-3.3-70b-versatile',
    'llama-3.1-8b-instant',
    'openai/gpt-oss-120b',
  ];
  supportsStreaming = true;
  supportsTools = true;
  supportsBatching = false;

  private apiKey: string;
  private baseUrl: string;

  constructor(config: GroqConfig) {
    super(config);

    if (!config.apiKey) {
      throw new AuthenticationError('groq', 'Groq API key is required');
    }

    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl || 'https://api.groq.com/openai/v1';
  }

  async generateResponse(request: LLMRequest): Promise<LLMResponse> {
    this.validateRequest(request);

    const startTime = Date.now();

    try {
      const response = await this.executeWithResiliency(async () => {
        const groqRequest = this.formatRequest(request);
        const httpResponse = await this.makeGroqRequest('/chat/completions', groqRequest, 'POST', request);

        if (!httpResponse.ok) {
          throw await LLMErrorFactory.fromFetchResponse('groq', httpResponse);
        }

        const data = await httpResponse.json() as unknown;
        validateSchema('groq', data, GROQ_RESPONSE_SCHEMA);
        return this.formatResponse(data as GroqResponse, Date.now() - startTime);
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
    const model = request.model || 'llama-3.3-70b-versatile';
    const capabilities = this.getModelCapabilities()[model];

    if (!capabilities) return 0;

    const inputTokens = request.messages.reduce((sum, msg) =>
      sum + Math.ceil(msg.content.length / 4), 0
    );
    const outputTokens = request.maxTokens || 1000;

    return this.calculateCost(inputTokens, outputTokens, model);
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.makeGroqRequest('/models', null, 'GET');
      return response.ok;
    } catch {
      return false;
    }
  }

  async getProviderBalance(): Promise<ProviderBalance> {
    return {
      provider: this.name,
      status: 'unavailable',
      source: 'not_supported',
      message: 'Groq does not expose a public billing or credit-balance API; use CreditLedger reporting for local quota state.'
    };
  }

  protected getModelCapabilities(): Record<string, ModelCapabilities> {
    return {
      'llama-3.3-70b-versatile': {
        maxContextLength: 128000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.00059, // $0.59 per 1M tokens
        outputTokenCost: 0.00079, // $0.79 per 1M tokens
        description: 'Llama 3.3 70B Versatile - High-quality fast inference on Groq'
      },
      'llama-3.1-8b-instant': {
        maxContextLength: 128000,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: false,
        inputTokenCost: 0.00005, // $0.05 per 1M tokens
        outputTokenCost: 0.00008, // $0.08 per 1M tokens
        description: 'Llama 3.1 8B Instant - Ultra-fast inference on Groq'
      },
      'openai/gpt-oss-120b': {
        maxContextLength: 128000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.00015, // $0.15 per 1M tokens (cached: $0.075/MTok)
        outputTokenCost: 0.0006,  // $0.60 per 1M tokens
        description: 'GPT-OSS 120B - OpenAI-compatible tool calling on Groq'
      }
    };
  }

  /**
   * Stream response support (OpenAI-compatible SSE format)
   */
  async streamResponse(request: LLMRequest): Promise<ReadableStream<string>> {
    this.validateRequest(request);

    const groqRequest = { ...this.formatRequest(request), stream: true };

    return new ReadableStream({
      start: async (controller) => {
        try {
          const response = await this.makeGroqRequest('/chat/completions', groqRequest, 'POST', request);

          if (!response.ok) {
            throw await LLMErrorFactory.fromFetchResponse('groq', response);
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
                  const content = parsed.choices?.[0]?.delta?.content;

                  if (content) {
                    controller.enqueue(content);
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

  private async makeGroqRequest(
    endpoint: string,
    body: GroqRequest | null,
    method: string = 'POST',
    request?: LLMRequest
  ): Promise<Response> {
    const headers: Record<string, string> = {
      'Authorization': `Bearer ${this.apiKey}`,
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

  private formatRequest(request: LLMRequest): GroqRequest {
    const messages: GroqMessage[] = [];
    const model = request.model || 'llama-3.3-70b-versatile';
    const usesTools =
      (request.tools?.length ?? 0) > 0 ||
      request.messages.some(message =>
        (message.toolCalls?.length ?? 0) > 0 || (message.toolResults?.length ?? 0) > 0
      );
    const jsonMode = request.response_format?.type === 'json_object';
    const jsonInstruction = '\n\nYou must respond with valid JSON only. No markdown fences, no commentary, no text outside the JSON.';

    if (usesTools && !TOOL_CAPABLE_MODELS.has(model)) {
      throw new ConfigurationError(
        this.name,
        `Model '${model}' does not support tool calling on Groq`
      );
    }

    if (request.systemPrompt) {
      messages.push({
        role: 'system',
        content: jsonMode ? request.systemPrompt + jsonInstruction : request.systemPrompt
      });
    } else if (jsonMode) {
      messages.push({
        role: 'system',
        content: jsonInstruction.trimStart()
      });
    }

    for (const message of request.messages) {
      if (message.role === 'system' && request.systemPrompt) {
        continue;
      }

      const msg: GroqMessage = {
        role: message.role,
        content: message.content
      };

      // Carry tool calls for multi-turn tool conversations
      if (message.toolCalls) {
        msg.tool_calls = message.toolCalls.map(tc => ({
          id: tc.id,
          type: 'function' as const,
          function: { name: tc.function.name, arguments: tc.function.arguments }
        }));
      }
      if (message.toolResults) {
        // Tool results come as separate messages in OpenAI format
        for (const tr of message.toolResults) {
          messages.push({ role: 'tool', content: tr.output, tool_call_id: tr.id });
        }
        continue; // Don't push the original message — tool results replace it
      }

      messages.push(msg);
    }

    const groqRequest: GroqRequest = {
      model,
      messages,
      temperature: request.temperature,
      max_tokens: request.maxTokens,
      stream: request.stream,
      seed: request.seed
    };

    // Pass through response_format if provided
    if (request.response_format) {
      groqRequest.response_format = request.response_format;
    }

    // Add tools if provided. Unsupported tool models are rejected above.
    if (request.tools && request.tools.length > 0) {
      groqRequest.tools = request.tools.map(t => ({
        type: 'function',
        function: {
          name: t.function.name,
          description: t.function.description,
          parameters: t.function.parameters as Record<string, unknown>,
        }
      }));
      if (request.toolChoice) {
        groqRequest.tool_choice = request.toolChoice;
      }
    }

    return groqRequest;
  }

  private formatResponse(
    data: GroqResponse,
    responseTime: number
  ): LLMResponse {
    const choice = data.choices[0];
    if (!choice) {
      throw new SchemaDriftError('groq', 'choices[0]', 'object', 'undefined');
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

    // Extract tool calls if present (validated at provider boundary)
    let toolCalls: ToolCall[] | undefined;
    if (choice.message.tool_calls && choice.message.tool_calls.length > 0) {
      const raw: ToolCall[] = choice.message.tool_calls.map(tc => ({
        id: tc.id,
        type: 'function' as const,
        function: { name: tc.function.name, arguments: tc.function.arguments }
      }));
      toolCalls = this.validateToolCalls(raw);
    }

    return {
      id: data.id,
      message: content,
      content,
      usage,
      model: data.model,
      provider: this.name,
      responseTime,
      finishReason: choice.finish_reason,
      toolCalls,
      metadata: {
        systemFingerprint: data.system_fingerprint,
        created: data.created
      }
    };
  }
}
