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
  ToolCall,
  TokenUsage,
  ProviderExecutionOptions
} from '../types.js';
import { BaseProvider, resolveCfGateway } from './base.js';
import {
  LLMErrorFactory,
  AuthenticationError,
  ConfigurationError,
  ModelNotFoundError,
  RateLimitError,
  SchemaDriftError
} from '../errors.js';
import { getProviderDefaultModel, resolveModelId } from '../model-catalog.js';
import { validateSchema, type SchemaField } from '../utils/schema-validator.js';
import { joinReasoning } from '../utils/reasoning.js';

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
        thinking: [
          { path: 'thinking', type: 'string' },
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
  { path: 'stop_sequence', type: 'string-or-null', optional: true },
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

interface AnthropicCacheControl {
  type: 'ephemeral';
}

interface AnthropicSystemBlock {
  type: 'text';
  text: string;
  cache_control?: AnthropicCacheControl;
}

interface AnthropicTool {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
  cache_control?: AnthropicCacheControl;
}

type AnthropicToolChoice =
  | { type: 'auto' }
  | { type: 'any' }
  | { type: 'tool'; name: string };

interface AnthropicRequest {
  model: string;
  messages: AnthropicMessage[];
  system?: string | AnthropicSystemBlock[];
  max_tokens: number;
  temperature?: number;
  stream?: boolean;
  tools?: AnthropicTool[];
  tool_choice?: AnthropicToolChoice;
}

interface AnthropicResponseContentBlock {
  type: 'text' | 'tool_use' | 'thinking';
  text?: string;
  thinking?: string;
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
  stop_sequence?: string | null;
  usage: {
    input_tokens: number;
    output_tokens: number;
    /** Tokens read from an existing cache_control breakpoint (cache hit). */
    cache_read_input_tokens?: number;
    /** Tokens written to a new cache_control breakpoint (cache miss/create). */
    cache_creation_input_tokens?: number;
    /** Split of cache writes by TTL; 1-hour writes are billed at a higher rate. */
    cache_creation?: { ephemeral_5m_input_tokens?: number; ephemeral_1h_input_tokens?: number };
  };
}

/** Models that return a 400 for a non-default temperature, top_p or top_k (Claude 4.7 and later). */
const FIXED_SAMPLING_MODELS = new Set(['claude-fable-5-1', 'claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5']);

/** Cache-read price as a fraction of the input price, where it is not the standard 0.1. */
const CACHE_READ_MULTIPLIER: Record<string, number> = {
  'claude-fable-5-1': 0.025,
  'claude-opus-5-5': 0.05,
  'claude-sonnet-5-5': 0.05,
};

/** Prompt size (input plus cache tokens) above which a model bills at its long-prompt rate. */
const LONG_PROMPT_THRESHOLD: Record<string, number> = { 'claude-haiku-5-5': 100_000 };
const LONG_PROMPT_MULTIPLIER = 5;

/**
 * Default max_tokens when the request sets none. Fable 5.1 and Opus 5.5 always
 * think, and thinking counts against max_tokens, so 1000 could leave no room for
 * the answer. 16,000 stays below the size where non-streaming requests risk
 * long-request timeouts.
 */
const ALWAYS_THINKING_DEFAULT_MAX_TOKENS: Record<string, number> = {
  'claude-fable-5-1': 16_000,
  'claude-opus-5-5': 16_000,
};

function defaultMaxTokens(model: string): number {
  return ALWAYS_THINKING_DEFAULT_MAX_TOKENS[model] ?? 1000;
}

export class AnthropicProvider extends BaseProvider {
  name = 'anthropic';
  models = [
    'claude-fable-5-1',
    'claude-opus-5-5',
    'claude-sonnet-5-5',
    'claude-haiku-5-5',
    'claude-opus-4-6',
    'claude-sonnet-4-6',
    'claude-opus-4-20250514',
    'claude-sonnet-4-20250514',
    'claude-haiku-4-5-20251001',
    'claude-3-7-sonnet-20250219',
    'claude-3-5-sonnet-20241022',
    'claude-3-5-haiku-20241022',
    'claude-3-opus-20240229',
    'claude-3-sonnet-20240229'
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
    const resolved = resolveCfGateway({
      provider: 'anthropic',
      baseUrl: config.baseUrl,
      cfGateway: config.cfGateway,
      suffix: 'anthropic',
      defaultBaseUrl: 'https://api.anthropic.com',
    });
    this.baseUrl = resolved.resolvedBaseUrl;
    this.cfGatewayActive = resolved.cfGatewayActive;
    this.version = config.version || '2023-06-01';
  }

  async generateResponse(request: LLMRequest, options?: ProviderExecutionOptions): Promise<LLMResponse> {
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
      }, options);

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
    const model = resolveModelId(request.model || this.getDefaultModel(request));
    const capabilities = this.getModelCapabilities()[model];

    if (!capabilities) return 0;

    // Estimate input tokens (rough approximation)
    const inputTokens = request.messages.reduce((sum, msg) =>
      sum + Math.ceil(msg.content.length / 4), 0
    );
    const outputTokens = request.maxTokens || defaultMaxTokens(model);

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
    return {
      provider: this.name,
      status: 'unavailable',
      source: 'not_supported',
      message: 'Anthropic credit-balance reporting requires the Admin API (admin.anthropic.com) with a separate admin key — not available through standard inference keys. Use CreditLedger for local quota state.'
    };
  }

  protected getModelCapabilities(): Record<string, ModelCapabilities> {
    return {
      'claude-fable-5-1': {
        maxContextLength: 1000000,
        supportsStreaming: true,
        supportsTools: true,
        supportsVision: true,
        supportsBatching: false,
        supportsPromptCache: true,
        inputTokenCost: 0.01,
        outputTokenCost: 0.05,
        description: 'Claude Fable 5.1 — top tier for demanding reasoning and long-horizon agentic work; adaptive thinking always on'
      },
      'claude-opus-5-5': {
        maxContextLength: 1000000,
        supportsStreaming: true,
        supportsTools: true,
        supportsVision: true,
        supportsBatching: false,
        supportsPromptCache: true,
        inputTokenCost: 0.004,
        outputTokenCost: 0.02,
        description: 'Claude Opus 5.5 — Anthropic\'s recommended starting model; adaptive thinking always on'
      },
      'claude-sonnet-5-5': {
        maxContextLength: 1000000,
        supportsStreaming: true,
        supportsTools: true,
        supportsVision: true,
        supportsBatching: false,
        supportsPromptCache: true,
        inputTokenCost: 0.002,
        outputTokenCost: 0.01,
        description: 'Claude Sonnet 5.5 — balanced speed and intelligence'
      },
      'claude-haiku-5-5': {
        maxContextLength: 1000000,
        supportsStreaming: true,
        supportsTools: true,
        supportsVision: true,
        supportsBatching: false,
        supportsPromptCache: true,
        inputTokenCost: 0.0001,
        outputTokenCost: 0.0005,
        description: 'Claude Haiku 5.5 — fastest tier; prompts over 100K tokens bill at 5x these rates'
      },
      'claude-opus-4-6': {
        maxContextLength: 1000000,
        supportsStreaming: true,
        supportsTools: true,
        supportsVision: true,
        supportsBatching: false,
        supportsPromptCache: true,
        inputTokenCost: 0.005, // $5 per 1M tokens
        outputTokenCost: 0.025, // $25 per 1M tokens
        description: 'Claude Opus 4.6 - legacy; retirement not sooner than 2027-02-05'
      },
      'claude-sonnet-4-6': {
        maxContextLength: 1000000,
        supportsStreaming: true,
        supportsTools: true,
        supportsVision: true,
        supportsBatching: false,
        supportsPromptCache: true,
        inputTokenCost: 0.003, // $3 per 1M tokens
        outputTokenCost: 0.015, // $15 per 1M tokens
        description: 'Claude Sonnet 4.6 - legacy; retirement not sooner than 2027-02-17'
      },
      'claude-opus-4-20250514': {
        maxContextLength: 200000,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.015, // $15 per 1M tokens
        outputTokenCost: 0.075, // $75 per 1M tokens
        description: 'Claude Opus 4 - retired on the Claude API 2026-06-15'
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
        supportsPromptCache: true,
        inputTokenCost: 0.001, // $1 per 1M tokens
        outputTokenCost: 0.005, // $5 per 1M tokens
        description: 'Claude Haiku 4.5 - legacy; retirement not sooner than 2026-10-15'
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
        supportsPromptCache: true,
        inputTokenCost: 0.0008, // $0.80 per 1M tokens
        outputTokenCost: 0.004, // $4 per 1M tokens
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
      }
    };
  }

  private async makeAnthropicRequest(
    endpoint: string,
    body: AnthropicRequest | null,
    method: string = 'POST',
    request?: LLMRequest,
    signal?: AbortSignal
  ): Promise<Response> {
    const headers: Record<string, string> = {
      'x-api-key': this.apiKey,
      'anthropic-version': this.version,
      'Content-Type': 'application/json',
      ...this.getAIGatewayHeaders(request),
      ...this.getCfGatewayHeaders(request)
    };

    const options: RequestInit = {
      method,
      signal,
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

    const model = resolveModelId(request.model || this.getDefaultModel(request));
    const anthropicRequest: AnthropicRequest = {
      model,
      messages,
      max_tokens: request.maxTokens || defaultMaxTokens(model),
      stream: request.stream
    };
    // Claude 4.7 and later reject a non-default temperature with a 400.
    if (!FIXED_SAMPLING_MODELS.has(anthropicRequest.model)) {
      anthropicRequest.temperature = request.temperature;
    }

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
    if (request.tools && request.tools.length > 0 && request.toolChoice !== 'none') {
      anthropicRequest.tools = request.tools.map(tool => ({
        name: tool.function.name,
        description: tool.function.description,
        input_schema: tool.function.parameters as Record<string, unknown>
      }));

      if (request.toolChoice) {
        if (request.toolChoice === 'auto') {
          anthropicRequest.tool_choice = { type: 'auto' };
        } else if (request.toolChoice === 'required') {
          anthropicRequest.tool_choice = { type: 'any' };
        } else if (typeof request.toolChoice === 'object') {
          anthropicRequest.tool_choice = {
            type: 'tool',
            name: request.toolChoice.function.name
          };
        }
      }
    }

    // Apply cache_control breakpoints when caller opts into provider-prefix caching.
    // Anthropic supports up to 4 breakpoints per request; we mark the system prompt
    // and (optionally) the last tool definition so the stable prefix is cached.
    const cacheStrategy = request.cache?.strategy;
    if (cacheStrategy === 'provider-prefix' || cacheStrategy === 'both') {
      const hint = request.cache?.cacheablePrefix ?? 'auto';

      // Wrap system as a content-block array so we can attach cache_control
      if (anthropicRequest.system && (hint === 'auto' || hint === 'system')) {
        const systemText = typeof anthropicRequest.system === 'string'
          ? anthropicRequest.system
          : anthropicRequest.system.map(b => b.text).join('');
        anthropicRequest.system = [{ type: 'text', text: systemText, cache_control: { type: 'ephemeral' } }];
      }

      // Mark the last tool definition as a breakpoint so the tools block is cached
      if (anthropicRequest.tools && anthropicRequest.tools.length > 0 && (hint === 'auto' || hint === 'tools')) {
        const tools = [...anthropicRequest.tools];
        tools[tools.length - 1] = { ...tools[tools.length - 1], cache_control: { type: 'ephemeral' } };
        anthropicRequest.tools = tools;
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
        /**
         * Anthropic image blocks require base64 bytes. Passing URL images as
         * placeholder text would make a vision request succeed without the
         * model seeing the image, so fail before dispatch instead.
         */
        if (image.url) {
          throw new ConfigurationError(
            this.name,
            'Anthropic does not support image URLs; convert the image to base64 data or use an OpenAI-compatible provider for URL-based vision.'
          );
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
    const reasoning = joinReasoning(data.content
      .filter(block => block.type === 'thinking')
      .map(block => block.thinking));

    const usage: TokenUsage = {
      inputTokens: data.usage.input_tokens,
      outputTokens: data.usage.output_tokens,
      totalTokens: data.usage.input_tokens + data.usage.output_tokens,
      cost: this.costWithCache(
        data.usage.input_tokens,
        data.usage.output_tokens,
        data.model,
        data.usage.cache_read_input_tokens,
        data.usage.cache_creation_input_tokens,
        data.usage.cache_creation?.ephemeral_1h_input_tokens
      ),
      costProvenance: 'catalog_estimate',
      tokenProvenance: 'provider_reported',
    };
    if (typeof data.usage.cache_read_input_tokens === 'number') {
      usage.cacheReadInputTokens = data.usage.cache_read_input_tokens;
    }
    if (typeof data.usage.cache_creation_input_tokens === 'number') {
      usage.cacheCreationInputTokens = data.usage.cache_creation_input_tokens;
      usage.cacheWriteInputTokens = data.usage.cache_creation_input_tokens;
    }

    const response: LLMResponse = {
      id: data.id,
      reasoning,
      message: textContent,
      content: textContent,
      usage,
      model: data.model,
      provider: this.name,
      responseTime,
      finishReason: this.mapStopReason(data.stop_reason),
      metadata: {
        ...(reasoning ? { reasoning } : {}),
        stopSequence: data.stop_sequence ?? undefined
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
   * Stream response support with SSE chunk schema validation.
   *
   * Each non-DONE data frame is validated against the expected Anthropic SSE
   * shape. Malformed JSON or a drifted content_block_delta envelope errors the
   * stream with SchemaDriftError (triggering factory fallback) rather than
   * silently producing truncated output.
   *
   * Unknown event types (additive upstream changes) are skipped for
   * forward-compatibility. Only the event types the parser actively reads
   * are validated.
   */
  async streamResponse(request: LLMRequest): Promise<ReadableStream<string>> {
    this.validateRequest(request);

    const anthropicRequest = { ...this.formatRequest(request), stream: true };
    const model = anthropicRequest.model;
    let sawUsage = false;
    let inputTokens = 0;
    let outputTokens = 0;
    let cacheReadInputTokens: number | undefined;
    let cacheCreationInputTokens: number | undefined;
    let cacheCreation1hInputTokens: number | undefined;

    // message_start and message_delta usage are cumulative snapshots; the latest value of each field wins.
    const readUsage = (usage: unknown): void => {
      if (!usage || typeof usage !== 'object' || Array.isArray(usage)) return;
      const usageObj = usage as Record<string, unknown>;
      if (typeof usageObj['input_tokens'] === 'number') {
        inputTokens = usageObj['input_tokens'];
        sawUsage = true;
      }
      if (typeof usageObj['output_tokens'] === 'number') {
        outputTokens = usageObj['output_tokens'];
        sawUsage = true;
      }
      if (typeof usageObj['cache_read_input_tokens'] === 'number') {
        cacheReadInputTokens = usageObj['cache_read_input_tokens'];
      }
      if (typeof usageObj['cache_creation_input_tokens'] === 'number') {
        cacheCreationInputTokens = usageObj['cache_creation_input_tokens'];
      }
      const split = usageObj['cache_creation'];
      if (split && typeof split === 'object' && !Array.isArray(split)
        && typeof (split as Record<string, unknown>)['ephemeral_1h_input_tokens'] === 'number') {
        cacheCreation1hInputTokens = (split as Record<string, number>)['ephemeral_1h_input_tokens'];
      }
    };

    const currentUsage = (): TokenUsage | undefined => {
      if (!sawUsage) return undefined;

      const usage: TokenUsage = {
        inputTokens,
        outputTokens,
        totalTokens: inputTokens + outputTokens,
        cost: this.costWithCache(inputTokens, outputTokens, model,
          cacheReadInputTokens, cacheCreationInputTokens, cacheCreation1hInputTokens),
        costProvenance: 'catalog_estimate',
        tokenProvenance: 'provider_reported',
      };

      if (cacheReadInputTokens !== undefined) usage.cacheReadInputTokens = cacheReadInputTokens;
      if (cacheCreationInputTokens !== undefined) {
        usage.cacheCreationInputTokens = cacheCreationInputTokens;
        usage.cacheWriteInputTokens = cacheCreationInputTokens;
      }

      return usage;
    };

    return this.createStream({
      provider: this.name,
      timeoutMs: this.config.timeout,
      open: async signal => {
        const response = await this.makeAnthropicRequest('/v1/messages', anthropicRequest, 'POST', request, signal);
        if (!response.ok) throw await LLMErrorFactory.fromFetchResponse(this.name, response);
        if (!response.body) throw new Error('No response body');
        return response.body;
      },
      event: value => {
        if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
        const event = value as Record<string, unknown>;
        if (event.type === 'error' || event.error) throw LLMErrorFactory.fromHttpResponse(this.name, 500, event);
        if (event.type === 'message_stop') return { done: true };
        if (event['type'] === 'message_start') {
          const message = event['message'];
          const usage = message && typeof message === 'object' && !Array.isArray(message)
            ? (message as Record<string, unknown>)['usage']
            : undefined;
          readUsage(usage);
          return {};
        }

        if (event['type'] === 'message_delta') {
          readUsage(event['usage']);
          return {};
        }

        if (event['type'] !== 'content_block_delta') return {};

        const delta = event['delta'];
        if (!delta || typeof delta !== 'object' || Array.isArray(delta)) {
          const actual = delta === null ? 'null' : Array.isArray(delta) ? 'array' : String(typeof delta);
          throw new SchemaDriftError('anthropic', 'content_block_delta.delta', 'object', actual);
        }

        const deltaObj = delta as Record<string, unknown>;
        // text_delta: the only delta type that produces content text.
        // input_json_delta (tool use) and other future delta types are skipped.
        if (deltaObj['type'] !== 'text_delta') return {};

        const text = deltaObj['text'];
        if (typeof text !== 'string') {
          const actual = text === null ? 'null' : String(typeof text);
          throw new SchemaDriftError('anthropic', 'content_block_delta.delta.text', 'string', actual);
        }

        return { text };
      },
      usage: currentUsage,
      onError: error => {
        if (error instanceof SchemaDriftError) this.config.hooks?.onSchemaDrift?.({
          provider: this.name, model, requestId: request.requestId,
          path: error.path, expected: error.expected, actual: error.actual, timestamp: Date.now(),
        });
      },
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

  private getDefaultModel(request: LLMRequest): string {
    return getProviderDefaultModel('anthropic', request);
  }

  /** Accept the invalid model IDs earlier releases published, validating the real one. */
  protected validateRequest(request: LLMRequest): void {
    super.validateRequest(request.model ? { ...request, model: resolveModelId(request.model) } : request);
  }

  /**
   * Token cost including prompt caching. `input` excludes cached tokens on the
   * Claude API. Multipliers on the base input rate: reads 0.1x (lower on some
   * models, see CACHE_READ_MULTIPLIER), 5-minute writes 1.25x, 1-hour writes 2x
   * (platform.claude.com/docs/en/about-claude/pricing, checked 2026-10-15).
   * Haiku 5.5 bills every rate at 5x when the prompt, cache included, exceeds 100K tokens.
   */
  private costWithCache(
    input: number, output: number, model: string,
    cacheRead = 0, cacheWrite = 0, cacheWrite1h = 0
  ): number {
    const base = this.calculateCost(input, output, model);
    const capabilities = this.getModelCapabilities()[model];
    if (!capabilities) return base;
    const write1h = Math.min(cacheWrite1h, cacheWrite);
    const readMultiplier = CACHE_READ_MULTIPLIER[model] ?? 0.1;
    const cachedInputUnits = cacheRead * readMultiplier + (cacheWrite - write1h) * 1.25 + write1h * 2;
    const cost = base + (cachedInputUnits / 1000) * capabilities.inputTokenCost;
    const longPrompt = input + cacheRead + cacheWrite > LONG_PROMPT_THRESHOLD[model];
    return longPrompt ? cost * LONG_PROMPT_MULTIPLIER : cost;
  }
}
