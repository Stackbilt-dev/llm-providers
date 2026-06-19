/**
 * Groq Provider
 * Implementation for Groq fast inference models (OpenAI-compatible API)
 */

import type { LLMRequest, LLMResponse, GroqConfig, ModelCapabilities, ProviderBalance, ToolCall, TokenUsage, BuiltInTool, BuiltInToolType, BuiltInToolResult } from '../types.js';
import { BaseProvider } from './base.js';
import {
  LLMErrorFactory,
  AuthenticationError,
  ConfigurationError,
  SchemaDriftError
} from '../errors.js';
import { getProviderDefaultModel, getCatalogEntry, modelSupportsBuiltInTools } from '../model-catalog.js';
import { validateSchema, type SchemaField } from '../utils/schema-validator.js';
import { attachStreamUsage, createStreamUsageTracker } from '../utils/stream-usage.js';
import { joinReasoning } from '../utils/reasoning.js';

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
        { path: 'message.content', type: 'string-or-null', optional: true },
        { path: 'message.reasoning', type: 'string', optional: true },
        { path: 'message.reasoning_content', type: 'string', optional: true },
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
        // Built-in tool executions (issue #69 S5). Validated SHALLOW on purpose:
        // only the always-present `type` is checked. `search_results.results`
        // sub-fields ({title,url,content,score}) are NOT validated here —
        // SchemaDriftError routes through the fallback chain, and the fallback
        // host (Cerebras gpt-oss) doesn't run built-in tools, so a false drift
        // on a citation sub-field (sampled n=1 in the S0 spike) would silently
        // degrade a working search response into a tool-less one. The parser
        // soft-degrades instead; citation-field coverage lives in a parser unit
        // test (the binding note's accepted alternative to a deep fixture).
        {
          path: 'message.executed_tools',
          type: 'array',
          optional: true,
          items: {
            shape: [
              { path: 'type', type: 'string' },
            ],
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

interface GroqFunctionTool {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

// Built-in tool entry on the OpenAI-compatible `tools` array, used by the
// gpt-oss path (e.g. `{ type: 'browser_search' }`). `type` is the provider's
// native wire identifier, not our normalized `BuiltInToolType`.
interface GroqBuiltInTool {
  type: string;
}

type GroqTool = GroqFunctionTool | GroqBuiltInTool;

interface GroqRequest {
  model: string;
  messages: GroqMessage[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  stream_options?: { include_usage?: boolean };
  response_format?:
    | { type: 'json_object' | 'text' }
    | { type: 'json_schema'; json_schema: { name: string; schema: Record<string, unknown>; strict?: boolean } };
  tools?: GroqTool[];
  tool_choice?: 'auto' | 'none' | { type: 'function'; function: { name: string } };
  // Compound systems (groq/compound*) configure built-in tools here rather than
  // on the OpenAI-style `tools` array. `enabled_tools` takes Groq's compound
  // vocabulary, which equals our normalized BuiltInToolType identifiers.
  compound_custom?: { tools: { enabled_tools: string[] } };
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
      content?: string | null;
      // The model's internal reasoning (exposes built-in search queries).
      // Present on both compound and gpt-oss when built-in tools run.
      reasoning?: string;
      reasoning_content?: string;
      tool_calls?: Array<{
        id: string;
        type: 'function';
        function: { name: string; arguments: string };
      }>;
      // Server-side built-in tool executions (issue #69). Open-ended `type`
      // (compound: 'search'; gpt-oss: 'browser_search'/'browser.open'/…); only
      // search executions carry `search_results.results`. Verified live in S0.
      executed_tools?: Array<{
        index?: number;
        type: string;
        name?: string;
        arguments?: string;
        output?: string;
        search_results?: {
          results?: Array<{ title: string; url: string; content: string; score: number }>;
        };
      }>;
    };
    finish_reason: 'stop' | 'length' | 'content_filter' | 'tool_calls';
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    /** Automatic prompt cache hit tokens (Groq-supported models). */
    prompt_tokens_details?: {
      cached_tokens?: number;
    };
  };
  system_fingerprint?: string;
}

// Models that support tool calling
const TOOL_CAPABLE_MODELS = new Set([
  'openai/gpt-oss-120b',
  'llama-3.3-70b-versatile',
]);

// Compound systems configure built-in tools via `compound_custom.tools`, taking
// the normalized identifiers verbatim. Every other capable model (gpt-oss) uses
// the OpenAI-style `tools` array with a different vocabulary.
const COMPOUND_MODELS = new Set(['groq/compound', 'groq/compound-mini']);

// Translate normalized BuiltInToolType → gpt-oss native wire identifier. Only
// the tools the catalog advertises for gpt-oss appear here; the capability gate
// (modelSupportsBuiltInTools) runs first, so a missing key after the gate passes
// is an internal map/catalog drift, not a caller error.
const GPT_OSS_BUILTIN_WIRE: Partial<Record<BuiltInToolType, string>> = {
  web_search: 'browser_search',
  code_interpreter: 'code_interpreter',
};

export class GroqProvider extends BaseProvider {
  name = 'groq';
  models = [
    'llama-3.3-70b-versatile',
    'llama-3.1-8b-instant',
    'openai/gpt-oss-120b',
    'groq/compound',
    'groq/compound-mini',
  ];
  supportsStreaming = true;
  supportsTools = true;
  supportsBatching = false;
  supportsVision = false;

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
      },
      // Compound systems: token costs are estimates (roll-up of the backing
      // models); built-in tool surcharges are billed separately and not
      // token-tracked. See model-catalog.ts for the routing rationale.
      'groq/compound': {
        maxContextLength: 131072,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.00015,
        outputTokenCost: 0.0006,
        description: 'Groq Compound - agentic system with server-side built-in tools'
      },
      'groq/compound-mini': {
        maxContextLength: 131072,
        supportsStreaming: true,
        supportsTools: true,
        supportsBatching: false,
        inputTokenCost: 0.0001,
        outputTokenCost: 0.0004,
        description: 'Groq Compound Mini - lower-latency built-in-tools system'
      }
    };
  }

  /**
   * Stream response support (OpenAI-compatible SSE format)
   */
  async streamResponse(request: LLMRequest): Promise<ReadableStream<string>> {
    this.validateRequest(request);

    const groqRequest = {
      ...this.formatRequest(request),
      stream: true,
      stream_options: { include_usage: true }
    };
    const hooks = this.config.hooks;
    const providerName = this.name;
    const usageTracker = createStreamUsageTracker();
    const model = groqRequest.model;
    let finalUsage: TokenUsage | undefined;

    const stream = new ReadableStream<string>({
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
                const err = new SchemaDriftError('groq', 'sse.chunk', 'valid-json', 'malformed-json');
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
                controller.error(new SchemaDriftError('groq', 'sse.choices[0].delta.content', 'string', actual));
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

    // Built-in tools (web_search etc.) — gated against the catalog and forked by
    // model family. Runs independently of the function-tool path above; gpt-oss
    // can carry both at once on the same `tools` array.
    if (request.builtInTools && request.builtInTools.length > 0) {
      this.applyBuiltInTools(groqRequest, model, request.builtInTools);
    }

    return groqRequest;
  }

  /**
   * Gate and serialize built-in tools onto a Groq request.
   *
   * Capability is checked against the catalog (`modelSupportsBuiltInTools`) so
   * the adapter and catalog can't drift. The wire shape then forks by family:
   * compound systems take the normalized identifiers on
   * `compound_custom.tools.enabled_tools`; gpt-oss takes OpenAI-style
   * `{ type }` entries on the shared `tools` array, translated to its native
   * vocabulary (`web_search` → `browser_search`).
   */
  private applyBuiltInTools(
    groqRequest: GroqRequest,
    model: string,
    builtInTools: BuiltInTool[]
  ): void {
    for (const tool of builtInTools) {
      if (!modelSupportsBuiltInTools(model, 'groq', tool.type)) {
        const supported = getCatalogEntry(model)?.capabilities.supportsBuiltInTools ?? [];
        const detail = supported.length > 0
          ? `Model '${model}' supports built-in tools [${supported.join(', ')}] but not '${tool.type}'.`
          : `Model '${model}' does not support built-in tools on Groq.`;
        throw new ConfigurationError(
          this.name,
          `${detail} Built-in tools require groq/compound or groq/compound-mini ` +
          `(all tools), or openai/gpt-oss-120b (web_search, code_interpreter).`
        );
      }
    }

    if (COMPOUND_MODELS.has(model)) {
      // Compound's enabled_tools vocabulary equals the normalized identifiers —
      // no translation. Dedupe to keep the wire payload clean.
      groqRequest.compound_custom = {
        tools: { enabled_tools: [...new Set(builtInTools.map(t => t.type))] }
      };
      return;
    }

    // gpt-oss path: merge built-in entries into the OpenAI-style tools array.
    const builtInEntries: GroqBuiltInTool[] = builtInTools.map(tool => {
      const wireType = GPT_OSS_BUILTIN_WIRE[tool.type];
      if (!wireType) {
        // Gate passed but no wire mapping — catalog advertises a tool the
        // translation map doesn't cover. Internal drift, not a caller error.
        throw new ConfigurationError(
          this.name,
          `Internal: no Groq wire mapping for built-in tool '${tool.type}' on '${model}'`
        );
      }
      return { type: wireType };
    });
    groqRequest.tools = [...(groqRequest.tools ?? []), ...builtInEntries];
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
    const reasoning = joinReasoning([choice.message.reasoning, choice.message.reasoning_content]);
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
    const cachedTokens = data.usage.prompt_tokens_details?.cached_tokens;
    if (typeof cachedTokens === 'number') {
      usage.cachedInputTokens = cachedTokens;
    }

    // Extract tool calls if present (validated at provider boundary).
    // Filter to function-type variants before dereferencing `tc.function`:
    // the schema discriminator treats unknown `type` values as forward-compat
    // (skipped, not drift), so a future `code_interpreter`-shaped variant
    // may arrive without the `function` field we expect. Dropping at the map
    // boundary keeps unknown variants invisible rather than surfacing a bare
    // TypeError that bypasses the drift/fallback machinery.
    let toolCalls: ToolCall[] | undefined;
    const functionCalls = choice.message.tool_calls?.filter(tc => tc.type === 'function');
    if (functionCalls && functionCalls.length > 0) {
      const raw: ToolCall[] = functionCalls.map(tc => ({
        id: tc.id,
        type: 'function' as const,
        function: { name: tc.function.name, arguments: tc.function.arguments }
      }));
      toolCalls = this.validateToolCalls(raw);
    }

    const builtInToolResults = this.extractBuiltInToolResults(choice.message.executed_tools);

    return {
      id: data.id,
      reasoning,
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
        created: data.created,
        // Surface only when present, to keep metadata clean for plain responses.
        ...(builtInToolResults ? { builtInToolResults } : {}),
        ...(reasoning ? { reasoning } : {}),
      }
    };
  }

  /**
   * Map Groq's `message.executed_tools[]` → normalized `BuiltInToolResult[]`
   * (issue #69, verified live in S0).
   *
   * Keeps only executions that carry a non-empty `search_results.results` and
   * flattens those into `results`, preserving the per-execution `type` / `name`
   * / `arguments`. Non-search executions (e.g. `code_interpreter`) have no
   * `search_results` and so drop out by design — that's the locked spec, not a
   * bug. Citation sub-fields are mapped as-is: any field the provider omits
   * surfaces as `undefined` (soft degrade) rather than throwing, since the
   * schema deliberately doesn't guard them (consumers HEAD-probe URLs anyway).
   */
  private extractBuiltInToolResults(
    executed: GroqResponse['choices'][number]['message']['executed_tools']
  ): BuiltInToolResult[] | undefined {
    if (!executed || executed.length === 0) return undefined;

    const out: BuiltInToolResult[] = [];
    for (const exec of executed) {
      const results = exec.search_results?.results;
      if (!Array.isArray(results) || results.length === 0) continue;

      const entry: BuiltInToolResult = {
        type: exec.type,
        results: results.map(r => ({
          title: r.title,
          url: r.url,
          content: r.content,
          score: r.score,
        })),
      };
      if (exec.name !== undefined) entry.name = exec.name;
      if (exec.arguments !== undefined) entry.arguments = exec.arguments;
      out.push(entry);
    }

    return out.length > 0 ? out : undefined;
  }

  private getDefaultModel(request: LLMRequest): string {
    return getProviderDefaultModel('groq', request);
  }
}
