/**
 * NVIDIA NIM Provider
 * Implementation for NVIDIA NIM inference models (OpenAI-compatible API)
 */

import type { LLMRequest, LLMResponse, NvidiaConfig, ModelCapabilities, ProviderBalance, ToolCall, TokenUsage } from '../types.js';
import { BaseProvider } from './base.js';
import {
  LLMErrorFactory,
  AuthenticationError,
  ConfigurationError,
  SchemaDriftError
} from '../errors.js';
import { getProviderDefaultModel } from '../model-catalog.js';
import { validateSchema, type SchemaField } from '../utils/schema-validator.js';
import { attachStreamUsage, createStreamUsageTracker } from '../utils/stream-usage.js';

// NVIDIA NIM serves the OpenAI /chat/completions contract. See groq.ts for the
// rationale on keeping each OpenAI-compat provider's schema as its own
// constant rather than a shared import.
const NVIDIA_RESPONSE_SCHEMA: SchemaField[] = [
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

interface NvidiaMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: Array<{ id: string; type: 'function'; function: { name: string; arguments: string } }>;
  tool_call_id?: string;
}

interface NvidiaTool {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

interface NvidiaRequest {
  model: string;
  messages: NvidiaMessage[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  stream_options?: { include_usage?: boolean };
  response_format?:
    | { type: 'json_object' | 'text' }
    | { type: 'json_schema'; json_schema: { name: string; schema: Record<string, unknown>; strict?: boolean } };
  tools?: NvidiaTool[];
  tool_choice?: 'auto' | 'none' | { type: 'function'; function: { name: string } };
  seed?: number;
}

interface NvidiaResponse {
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
      }> | null;
    };
    finish_reason: 'stop' | 'length' | 'content_filter' | 'tool_calls';
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    // null on NVIDIA NIM — not an object like Groq/Cerebras
    prompt_tokens_details?: { cached_tokens?: number } | null;
  };
  system_fingerprint?: string | null;
}

// Models that support tool/function calling on NVIDIA NIM.
// Verified via live API probe; add new models here as NVIDIA certifies them.
// DeepSeek models omitted — returned 502 during tool-call verification; re-add once confirmed.
const TOOL_CAPABLE_MODELS = new Set([
  'meta/llama-3.1-70b-instruct',
  'meta/llama-3.1-8b-instruct',
  'meta/llama-3.3-70b-instruct',
  'meta/llama-4-maverick-17b-128e-instruct',
  'nvidia/llama-3.1-nemotron-70b-instruct',
  'nvidia/llama-3.3-nemotron-super-49b-v1',
  'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'mistralai/mistral-large-2-instruct',
]);

export class NvidiaProvider extends BaseProvider {
  name = 'nvidia';
  models = [
    'meta/llama-3.3-70b-instruct',
    'meta/llama-4-maverick-17b-128e-instruct',
    'nvidia/llama-3.1-nemotron-70b-instruct',
    'nvidia/llama-3.3-nemotron-super-49b-v1',
    'deepseek-ai/deepseek-v4-flash',
    'meta/llama-3.1-70b-instruct',
    'nvidia/llama-3.1-nemotron-ultra-253b-v1',
    'mistralai/mistral-large-2-instruct',
    'deepseek-ai/deepseek-v4-pro',
  ];
  supportsStreaming = true;
  supportsTools = true;
  supportsBatching = false;

  private apiKey: string;
  private baseUrl: string;

  constructor(config: NvidiaConfig) {
    super(config);

    if (!config.apiKey) {
      throw new AuthenticationError('nvidia', 'NVIDIA API key is required');
    }

    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl || 'https://integrate.api.nvidia.com/v1';
  }

  async generateResponse(request: LLMRequest): Promise<LLMResponse> {
    this.validateRequest(request);

    const startTime = Date.now();

    try {
      const response = await this.executeWithResiliency(async () => {
        const nvidiaRequest = this.formatRequest(request);
        const httpResponse = await this.makeNvidiaRequest('/chat/completions', nvidiaRequest, 'POST', request);

        if (!httpResponse.ok) {
          throw await LLMErrorFactory.fromFetchResponse('nvidia', httpResponse);
        }

        const data = await httpResponse.json() as unknown;
        validateSchema('nvidia', data, NVIDIA_RESPONSE_SCHEMA);
        return this.formatResponse(data as NvidiaResponse, Date.now() - startTime);
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

    const inputTokens = request.messages.reduce((sum, msg) =>
      sum + Math.ceil(msg.content.length / 4), 0
    );
    const outputTokens = request.maxTokens || 1000;

    return this.calculateCost(inputTokens, outputTokens, model);
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.makeNvidiaRequest('/models', null, 'GET');
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
      message: 'NVIDIA NIM does not expose a public billing or credit-balance API; use CreditLedger reporting for local quota state.'
    };
  }

  protected getModelCapabilities(): Record<string, ModelCapabilities> {
    // Costs are zero placeholders — NVIDIA NIM dev-tier runs on credits;
    // production pricing varies by model and deployment. Update via CreditLedger.
    return {
      'meta/llama-3.3-70b-instruct': {
        maxContextLength: 128000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'Meta Llama 3.3 70B Instruct on NVIDIA NIM'
      },
      'meta/llama-4-maverick-17b-128e-instruct': {
        maxContextLength: 1048576,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'Meta Llama 4 Maverick 17B 128E MoE Instruct on NVIDIA NIM'
      },
      'nvidia/llama-3.1-nemotron-70b-instruct': {
        maxContextLength: 128000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'NVIDIA Llama 3.1 Nemotron 70B Instruct — NVIDIA-optimized for accuracy'
      },
      'nvidia/llama-3.3-nemotron-super-49b-v1': {
        maxContextLength: 128000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'NVIDIA Llama 3.3 Nemotron Super 49B v1 — efficiency-optimized'
      },
      'deepseek-ai/deepseek-v4-flash': {
        maxContextLength: 65536,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: false,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'DeepSeek V4 Flash on NVIDIA NIM'
      },
      'meta/llama-3.1-70b-instruct': {
        maxContextLength: 128000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'Meta Llama 3.1 70B Instruct on NVIDIA NIM'
      },
      'nvidia/llama-3.1-nemotron-ultra-253b-v1': {
        maxContextLength: 128000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'NVIDIA Llama 3.1 Nemotron Ultra 253B v1 — maximum quality'
      },
      'mistralai/mistral-large-2-instruct': {
        maxContextLength: 131072,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'Mistral Large 2 Instruct on NVIDIA NIM'
      },
      'deepseek-ai/deepseek-v4-pro': {
        maxContextLength: 65536,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: false,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'DeepSeek V4 Pro on NVIDIA NIM'
      },
    };
  }

  async streamResponse(request: LLMRequest): Promise<ReadableStream<string>> {
    this.validateRequest(request);

    const nvidiaRequest = {
      ...this.formatRequest(request),
      stream: true,
      stream_options: { include_usage: true }
    };
    const hooks = this.config.hooks;
    const providerName = this.name;
    const usageTracker = createStreamUsageTracker();
    const model = nvidiaRequest.model;
    let finalUsage: TokenUsage | undefined;

    const stream = new ReadableStream<string>({
      start: async (controller) => {
        try {
          const response = await this.makeNvidiaRequest('/chat/completions', nvidiaRequest, 'POST', request);

          if (!response.ok) {
            throw await LLMErrorFactory.fromFetchResponse('nvidia', response);
          }

          const reader = response.body?.getReader();
          if (!reader) {
            throw new Error('No response body');
          }

          const decoder = new TextDecoder();
          let buffer = '';

          const emitDrift = (path: string, expected: string, actual: string): void => {
            hooks?.onSchemaDrift?.({
              provider: providerName,
              model: request.model,
              requestId: request.requestId,
              path,
              expected,
              actual,
              timestamp: Date.now(),
            });
          };

          while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (!line.startsWith('data: ')) continue;

              const data = line.slice(6).trim();
              if (data === '[DONE]' || data === '') {
                if (data === '[DONE]') {
                  usageTracker.resolve(finalUsage);
                  controller.close();
                  return;
                }
                continue;
              }

              let parsed: unknown;
              try {
                parsed = JSON.parse(data);
              } catch {
                const err = new SchemaDriftError('nvidia', 'sse.chunk', 'valid-json', 'malformed-json');
                emitDrift('sse.chunk', 'valid-json', 'malformed-json');
                controller.error(err);
                reader.cancel().catch(() => {});
                return;
              }

              if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) continue;
              const chunk = parsed as Record<string, unknown>;
              const parsedUsage = this.parseOpenAICompatibleStreamUsage(chunk['usage'], model);
              if (parsedUsage) finalUsage = parsedUsage;

              const choices = chunk['choices'];
              if (!Array.isArray(choices) || choices.length === 0) continue;
              const delta = (choices[0] as Record<string, unknown>)['delta'];
              if (!delta || typeof delta !== 'object' || Array.isArray(delta)) continue;
              const content = (delta as Record<string, unknown>)['content'];

              if (content === undefined || content === null) continue;
              if (typeof content !== 'string') {
                const actual = String(typeof content);
                emitDrift('sse.choices[0].delta.content', 'string', actual);
                controller.error(new SchemaDriftError('nvidia', 'sse.choices[0].delta.content', 'string', actual));
                reader.cancel().catch(() => {});
                return;
              }

              if (content) controller.enqueue(content);
            }
          }

          usageTracker.resolve(finalUsage);
          controller.close();
        } catch (error) {
          usageTracker.resolve(undefined);
          controller.error(error);
        }
      }
    });

    return attachStreamUsage(stream, usageTracker.promise);
  }

  private async makeNvidiaRequest(
    endpoint: string,
    body: NvidiaRequest | null,
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

  private formatRequest(request: LLMRequest): NvidiaRequest {
    const messages: NvidiaMessage[] = [];
    const model = request.model || this.getDefaultModel(request);
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
        `Model '${model}' does not support tool calling on NVIDIA NIM`
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

      const msg: NvidiaMessage = {
        role: message.role,
        content: message.content
      };

      if (message.toolCalls) {
        msg.tool_calls = message.toolCalls.map(tc => ({
          id: tc.id,
          type: 'function' as const,
          function: { name: tc.function.name, arguments: tc.function.arguments }
        }));
      }
      if (message.toolResults) {
        for (const tr of message.toolResults) {
          messages.push({ role: 'tool', content: tr.output, tool_call_id: tr.id });
        }
        continue;
      }

      messages.push(msg);
    }

    const nvidiaRequest: NvidiaRequest = {
      model,
      messages,
      temperature: request.temperature,
      max_tokens: request.maxTokens,
      stream: request.stream,
      seed: request.seed
    };

    if (request.response_format) {
      nvidiaRequest.response_format = request.response_format;
    }

    if (request.tools && request.tools.length > 0) {
      nvidiaRequest.tools = request.tools.map(t => ({
        type: 'function',
        function: {
          name: t.function.name,
          description: t.function.description,
          parameters: t.function.parameters as Record<string, unknown>,
        }
      }));
      if (request.toolChoice) {
        nvidiaRequest.tool_choice = request.toolChoice;
      }
    }

    return nvidiaRequest;
  }

  private formatResponse(
    data: NvidiaResponse,
    responseTime: number
  ): LLMResponse {
    const choice = data.choices[0];
    if (!choice) {
      throw new SchemaDriftError('nvidia', 'choices[0]', 'object', 'undefined');
    }

    const content = choice.message.content || '';
    const usage: TokenUsage = {
      inputTokens: data.usage.prompt_tokens,
      outputTokens: data.usage.completion_tokens,
      totalTokens: data.usage.total_tokens,
      cost: this.calculateCost(
        data.usage.prompt_tokens,
        data.usage.completion_tokens,
        data.model
      )
    };
    // NVIDIA NIM returns prompt_tokens_details as null; handle null gracefully.
    const cachedTokens = data.usage.prompt_tokens_details?.cached_tokens;
    if (typeof cachedTokens === 'number') {
      usage.cachedInputTokens = cachedTokens;
    }

    // Extract tool calls if present. NVIDIA NIM returns `tool_calls: []` (empty
    // array) when no tools are used, unlike Groq which omits the field entirely.
    // Filter to function-type variants before dereferencing `tc.function` — same
    // forward-compat rationale as groq.ts.
    let toolCalls: ToolCall[] | undefined;
    const rawToolCalls = choice.message.tool_calls;
    const functionCalls = Array.isArray(rawToolCalls)
      ? rawToolCalls.filter(tc => tc.type === 'function')
      : [];
    if (functionCalls.length > 0) {
      const raw: ToolCall[] = functionCalls.map(tc => ({
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

  private getDefaultModel(request: LLMRequest): string {
    return getProviderDefaultModel('nvidia', request);
  }
}
