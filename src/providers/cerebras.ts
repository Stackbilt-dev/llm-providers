/**
 * Cerebras Provider
 * Implementation for Cerebras fast inference models (OpenAI-compatible API)
 */

import type { LLMRequest, LLMResponse, CerebrasConfig, ModelCapabilities, ToolCall } from '../types';
import { BaseProvider } from './base';
import {
  LLMErrorFactory,
  AuthenticationError,
  ConfigurationError,
  SchemaDriftError
} from '../errors';
import { validateSchema, type SchemaField } from '../utils/schema-validator';

// Cerebras serves the OpenAI /chat/completions contract. See groq.ts for the
// rationale on keeping each OpenAI-compat provider's schema as its own
// constant rather than a shared import.
const CEREBRAS_RESPONSE_SCHEMA: SchemaField[] = [
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

interface CerebrasMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: Array<{ id: string; type: 'function'; function: { name: string; arguments: string } }>;
  tool_call_id?: string;
}

interface CerebrasTool {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

interface CerebrasRequest {
  model: string;
  messages: CerebrasMessage[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  tools?: CerebrasTool[];
  tool_choice?: 'auto' | 'none' | { type: 'function'; function: { name: string } };
  seed?: number;
}

interface CerebrasResponse {
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
  'zai-glm-4.7',
  'qwen-3-235b-a22b-instruct-2507',
]);

export class CerebrasProvider extends BaseProvider {
  name = 'cerebras';
  models = [
    'llama-3.1-8b',
    'llama-3.3-70b',
    'zai-glm-4.7',
    'qwen-3-235b-a22b-instruct-2507',
  ];
  supportsStreaming = true;
  supportsTools = true;
  supportsBatching = false;

  private apiKey: string;
  private baseUrl: string;

  constructor(config: CerebrasConfig) {
    super(config);

    if (!config.apiKey) {
      throw new AuthenticationError('cerebras', 'Cerebras API key is required');
    }

    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl || 'https://api.cerebras.ai/v1';
  }

  async generateResponse(request: LLMRequest): Promise<LLMResponse> {
    this.validateRequest(request);

    const startTime = Date.now();

    try {
      const response = await this.executeWithResiliency(async () => {
        const cerebrasRequest = this.formatRequest(request);
        const httpResponse = await this.makeCerebrasRequest('/chat/completions', cerebrasRequest, 'POST', request);

        if (!httpResponse.ok) {
          throw await LLMErrorFactory.fromFetchResponse('cerebras', httpResponse);
        }

        const data = await httpResponse.json() as unknown;
        validateSchema('cerebras', data, CEREBRAS_RESPONSE_SCHEMA);
        return this.formatResponse(data as CerebrasResponse, Date.now() - startTime);
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
    const model = request.model || 'llama-3.1-8b';
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
      const response = await this.makeCerebrasRequest('/models', null, 'GET');
      return response.ok;
    } catch {
      return false;
    }
  }

  protected getModelCapabilities(): Record<string, ModelCapabilities> {
    return {
      'llama-3.1-8b': {
        maxContextLength: 128000,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: false,
        inputTokenCost: 0.0001, // $0.10 per 1M tokens
        outputTokenCost: 0.0001, // $0.10 per 1M tokens
        description: 'Llama 3.1 8B - Fast inference on Cerebras'
      },
      'llama-3.3-70b': {
        maxContextLength: 128000,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: false,
        inputTokenCost: 0.0006, // $0.60 per 1M tokens
        outputTokenCost: 0.0006, // $0.60 per 1M tokens
        description: 'Llama 3.3 70B - High-quality fast inference on Cerebras'
      },
      'zai-glm-4.7': {
        maxContextLength: 131000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.00225, // $2.25 per 1M tokens
        outputTokenCost: 0.00275, // $2.75 per 1M tokens
        description: 'ZAI-GLM 4.7 355B - Reasoning mode, tool calling, structured outputs (Preview)'
      },
      'qwen-3-235b-a22b-instruct-2507': {
        maxContextLength: 131000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.0006, // $0.60 per 1M tokens
        outputTokenCost: 0.0012, // $1.20 per 1M tokens
        description: 'Qwen 3 235B MoE (22B active) - Tool calling, structured outputs (Preview)'
      }
    };
  }

  /**
   * Stream response support (OpenAI-compatible SSE format)
   */
  async streamResponse(request: LLMRequest): Promise<ReadableStream<string>> {
    this.validateRequest(request);

    const cerebrasRequest = { ...this.formatRequest(request), stream: true };

    return new ReadableStream({
      start: async (controller) => {
        try {
          const response = await this.makeCerebrasRequest('/chat/completions', cerebrasRequest, 'POST', request);

          if (!response.ok) {
            throw await LLMErrorFactory.fromFetchResponse('cerebras', response);
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

  private async makeCerebrasRequest(
    endpoint: string,
    body: CerebrasRequest | null,
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

  private formatRequest(request: LLMRequest): CerebrasRequest {
    const messages: CerebrasMessage[] = [];
    const model = request.model || 'llama-3.1-8b';
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
        `Model '${model}' does not support tool calling on Cerebras`
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

      const msg: CerebrasMessage = {
        role: message.role,
        content: message.content
      };

      // Carry tool calls/results for multi-turn tool conversations
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

    const result: CerebrasRequest = {
      model,
      messages,
      temperature: request.temperature,
      max_tokens: request.maxTokens,
      stream: request.stream,
      seed: request.seed
    };

    // Add tools if provided. Unsupported tool models are rejected above.
    if (request.tools && request.tools.length > 0) {
      result.tools = request.tools.map(t => ({
        type: 'function',
        function: {
          name: t.function.name,
          description: t.function.description,
          parameters: t.function.parameters as Record<string, unknown>,
        }
      }));
      if (request.toolChoice) {
        result.tool_choice = request.toolChoice;
      }
    }

    return result;
  }

  private formatResponse(
    data: CerebrasResponse,
    responseTime: number
  ): LLMResponse {
    const choice = data.choices[0];
    if (!choice) {
      throw new SchemaDriftError('cerebras', 'choices[0]', 'object', 'undefined');
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
