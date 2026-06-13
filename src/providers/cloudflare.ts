/**
 * Cloudflare AI Provider
 * Implementation for Cloudflare Workers AI with cost optimization
 */

import type {
  LLMRequest,
  LLMResponse,
  LLMImageInput,
  CloudflareConfig,
  CloudflareAIGatewayOptions,
  ModelCapabilities,
  TokenUsage,
  ToolCall
} from '../types.js';
import { BaseProvider } from './base.js';
import {
  ConfigurationError,
  InvalidRequestError,
  ModelNotFoundError
} from '../errors.js';
import { getProviderDefaultModel } from '../model-catalog.js';
import { validateSchema, type SchemaField } from '../utils/schema-validator.js';
import { attachStreamUsage, createStreamUsageTracker } from '../utils/stream-usage.js';

/**
 * Minimum structural fields the Cloudflare Workers AI parser reads.
 * All fields are optional because Workers AI returns different shapes
 * depending on the model and endpoint (chat/completion/vision).
 * Validates type correctness when a field IS present, without requiring
 * any particular shape.
 */
const CLOUDFLARE_RESPONSE_SCHEMA: SchemaField[] = [
  { path: 'response', type: 'string-or-number', optional: true },
  { path: 'id', type: 'string', optional: true },
  { path: 'model', type: 'string', optional: true },
  { path: 'output_text', type: 'string', optional: true },
  {
    path: 'choices',
    type: 'array',
    optional: true,
    items: {
      shape: [
        { path: 'message', type: 'object' },
        { path: 'message.content', type: 'string-or-null', optional: true },
      ]
    }
  },
  { path: 'output', type: 'array', optional: true },
  // Anthropic-via-CF format: top-level content array and stop_reason
  { path: 'content', type: 'array', optional: true },
  { path: 'stop_reason', type: 'string', optional: true },
  { path: 'usage', type: 'object', optional: true },
];

interface CloudflareContentPart {
  type: 'text' | 'image_url';
  text?: string;
  image_url?: { url: string };
}

interface CloudflareMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null | CloudflareContentPart[];
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

interface CloudflareRequest {
  messages: CloudflareMessage[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  tools?: LLMRequest['tools'];
  tool_choice?: LLMRequest['toolChoice'];
  /** LoRA adapter name or UUID (Workers AI fine-tune). Forwarded as-is to ai.run(). */
  lora?: string;
  top_p?: number;
  frequency_penalty?: number;
  /** Anthropic-format models served via the CF binding use a top-level system field. */
  system?: string;
}

interface CloudflareRunOptions {
  gateway?: CloudflareAIGatewayOptions;
  extraHeaders?: Record<string, string>;
}

/** Workers AI returns various response shapes depending on the model. */
interface WorkersAIToolCall {
  id?: string;
  type?: string;
  function?: { name?: string; arguments?: string | Record<string, unknown> };
}

interface WorkersAIChoice {
  index?: number;
  message?: {
    role?: string;
    content?: string | null | Array<{ type?: string; text?: string }>;
    reasoning_content?: string;
    tool_calls?: WorkersAIToolCall[];
  };
  finish_reason?: string;
}

interface WorkersAIOutputItem {
  type?: string;
  text?: string;
  name?: string;
  call_id?: string;
  id?: string;
  arguments?: string | Record<string, unknown>;
  content?: Array<{ type?: string; text?: string }>;
}

interface WorkersAIUsage {
  prompt_tokens?: number;
  input_tokens?: number;
  completion_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  cached_tokens?: number;
  prompt_tokens_details?: { cached_tokens?: number };
  input_tokens_details?: { cached_tokens?: number };
}

interface WorkersAIResult {
  response?: string;
  id?: string;
  model?: string;
  choices?: WorkersAIChoice[];
  output?: WorkersAIOutputItem[];
  output_text?: string;
  usage?: WorkersAIUsage;
  result?: WorkersAIResult; // wrapped responses
  // Anthropic-via-CF format fields
  content?: Array<{ type?: string; text?: string }> | string;
  stop_reason?: string;
  type?: string;
  role?: string;
}

// Models that require the raw { image, prompt } binding format rather than chat/image_url.
// Add any new CF vision models here if they exhibit the same null-content symptom via the binding.
// (The chat path returns choices[0].message.content === null through the Workers AI binding,
// silently producing "".)
const LLAMA_VISION_RAW_MODELS = new Set([
  '@cf/meta/llama-3.2-11b-vision-instruct'
]);

// Cloudflare-managed third-party models that use the Anthropic messages API format
// rather than the standard Workers AI chat format. These expect { messages, system?,
// max_tokens } and return { id, type, role, content[], stop_reason, usage }.
const ANTHROPIC_VIA_CF_MODELS = new Set([
  'anthropic/claude-opus-4.8',
]);

export class CloudflareProvider extends BaseProvider {
  name = 'cloudflare';
  models = [
    '@cf/meta/llama-3.1-8b-instruct',
    '@cf/meta/llama-3.1-70b-instruct',
    '@cf/meta/llama-3-8b-instruct',
    '@cf/meta/llama-2-7b-chat-int8',
    '@cf/microsoft/phi-2',
    '@cf/mistral/mistral-7b-instruct-v0.1',
    '@cf/openchat/openchat-3.5-0106',
    '@cf/openai/gpt-oss-120b',
    '@cf/moonshotai/kimi-k2.6',
    '@cf/zai-org/glm-4.7-flash',
    'deepseek/deepseek-v4-pro',
    '@cf/tinyllama/tinyllama-1.1b-chat-v1.0',
    '@cf/qwen/qwen1.5-0.5b-chat',
    '@cf/qwen/qwen1.5-1.8b-chat',
    '@cf/qwen/qwen1.5-14b-chat-awq',
    '@cf/qwen/qwen1.5-7b-chat-awq',
    '@cf/google/gemma-4-26b-a4b-it',
    '@cf/meta/llama-3.3-70b-instruct-fp8-fast',
    '@cf/meta/llama-3.2-1b-instruct',
    '@cf/meta/llama-3.2-3b-instruct',
    '@cf/openai/gpt-oss-20b',
    '@cf/qwen/qwen2.5-coder-32b-instruct',
    '@cf/qwen/qwen3-30b-a3b-fp8',
    '@cf/mistralai/mistral-small-3.1-24b-instruct',
    '@cf/moonshotai/kimi-k2.7-code',
    '@cf/meta/llama-4-scout-17b-16e-instruct',
    '@cf/meta/llama-3.2-11b-vision-instruct',
    '@cf/nvidia/nemotron-3-120b-a12b',
    '@cf/deepseek-ai/deepseek-r1-distill-qwen-32b',
    '@cf/qwen/qwq-32b',
    'anthropic/claude-opus-4.8',
  ];
  supportsStreaming = true;
  supportsTools = true;
  supportsBatching = true;
  supportsVision = true;

  private ai: Ai;
  private accountId?: string;
  private gateway?: CloudflareAIGatewayOptions;

  constructor(config: CloudflareConfig) {
    super(config);

    if (!config.ai) {
      throw new ConfigurationError('cloudflare', 'Cloudflare AI binding is required');
    }

    this.ai = config.ai;
    this.accountId = config.accountId;
    this.gateway = config.gateway;
  }

  async generateResponse(request: LLMRequest): Promise<LLMResponse> {
    this.validateRequest(request);

    const startTime = Date.now();

    try {
      const response = await this.executeWithResiliency(async () => {
        const model = request.model || this.getRecommendedModel(request);

        if (!this.models.includes(model)) {
          throw new ModelNotFoundError('cloudflare', model);
        }

        // llama-3.2-11b vision requires the raw Workers AI binding format.
        // The chat/image_url path returns null content via the binding.
        if (LLAMA_VISION_RAW_MODELS.has(model) && (request.images?.length ?? 0) > 0) {
          const result = await this.runLlamaVisionRaw(request, model);
          return this.formatResponse(result as WorkersAIResult, model, request, Date.now() - startTime);
        }

        const cloudflareRequest = this.formatRequest(request, model);
        const result = await this.runModel(model, cloudflareRequest, request);

        return this.formatResponse(result as WorkersAIResult, model, request, Date.now() - startTime);
      });

      this.updateMetrics(response.responseTime, true, response.usage?.cost || 0);
      this.logRequest(request, response);

      return response;
    } catch (error) {
      const responseTime = Date.now() - startTime;
      this.updateMetrics(responseTime, false);
      this.logRequest(request, undefined, error as Error);
      throw this.wrapCFError(error);
    }
  }

  validateConfig(): boolean {
    return !!(this.ai);
  }

  getModels(): string[] {
    return [...this.models];
  }

  estimateCost(request: LLMRequest): number {
    // Cloudflare AI is essentially "free" as it's included in Workers compute
    // But we can estimate the computational cost
    const model = request.model || this.getRecommendedModel(request);
    const capabilities = this.getModelCapabilities()[model];

    if (!capabilities) return 0;

    // Very low cost since it's included in Workers compute
    const inputTokens = request.messages.reduce((sum, msg) =>
      sum + Math.ceil(msg.content.length / 4), 0
    );
    const outputTokens = request.maxTokens || 1000;

    return this.calculateCost(inputTokens, outputTokens, model);
  }

  async healthCheck(): Promise<boolean> {
    try {
      const testRequest: CloudflareRequest = {
        messages: [{ role: 'user', content: 'Hi' }],
        max_tokens: 1
      };

      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Ai.run() requires branded model types
      await (this.ai as { run(model: string, input: unknown): Promise<unknown> }).run(
        this.getRecommendedModel({ messages: [{ role: 'user', content: 'Hi' }], maxTokens: 1 }),
        testRequest
      );
      return true;
    } catch {
      return false;
    }
  }

  protected getModelCapabilities(): Record<string, ModelCapabilities> {
    return {
      '@cf/meta/llama-3.1-8b-instruct': {
        maxContextLength: 8192,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: true,
        inputTokenCost: 0.0000001, // Essentially free
        outputTokenCost: 0.0000001,
        description: 'LLaMA 3.1 8B - Fast and efficient'
      },
      '@cf/meta/llama-3.1-70b-instruct': {
        maxContextLength: 8192,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: true,
        inputTokenCost: 0.0000005, // Slightly higher compute cost
        outputTokenCost: 0.0000005,
        description: 'LLaMA 3.1 70B - High performance'
      },
      '@cf/meta/llama-3-8b-instruct': {
        maxContextLength: 8192,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: true,
        inputTokenCost: 0.0000001,
        outputTokenCost: 0.0000001,
        description: 'LLaMA 3 8B - Reliable performance'
      },
      '@cf/meta/llama-2-7b-chat-int8': {
        maxContextLength: 4096,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: true,
        inputTokenCost: 0.00000005,
        outputTokenCost: 0.00000005,
        description: 'LLaMA 2 7B - Quantized for speed'
      },
      '@cf/microsoft/phi-2': {
        maxContextLength: 2048,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: true,
        inputTokenCost: 0.00000002,
        outputTokenCost: 0.00000002,
        description: 'Phi-2 - Small but capable'
      },
      '@cf/mistral/mistral-7b-instruct-v0.1': {
        maxContextLength: 8192,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: true,
        inputTokenCost: 0.0000001,
        outputTokenCost: 0.0000001,
        description: 'Mistral 7B - Balanced performance'
      },
      '@cf/openchat/openchat-3.5-0106': {
        maxContextLength: 8192,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: true,
        inputTokenCost: 0.0000001,
        outputTokenCost: 0.0000001,
        description: 'OpenChat 3.5 - Conversation optimized'
      },
      '@cf/openai/gpt-oss-120b': {
        maxContextLength: 128000,
        supportsStreaming: true,
        supportsTools: true,
        toolCalling: true,
        supportsBatching: true,
        inputTokenCost: 0.00035,  // $0.35/MTok — matches workers-ai-chat.ts GPT_OSS_RATES
        outputTokenCost: 0.00075, // $0.75/MTok
        description: 'GPT-OSS 120B - OpenAI-format tool calling on Workers AI'
      },
      '@cf/moonshotai/kimi-k2.6': {
        maxContextLength: 262100,
        supportsStreaming: true,
        supportsTools: true,
        toolCalling: true,
        supportsVision: true,
        supportsBatching: true,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'Kimi K2.6 — multi-turn tools, vision, structured outputs, 262K context'
      },
      '@cf/zai-org/glm-4.7-flash': {
        maxContextLength: 131072,
        supportsStreaming: true,
        supportsTools: true,
        toolCalling: true,
        supportsBatching: true,
        inputTokenCost: 0,
        outputTokenCost: 0,
        thinkingModel: true,
        description: 'GLM-4.7-Flash — chain-of-thought reasoning model; outputs thinking traces, not suitable for direct-response routing'
      },
      'deepseek/deepseek-v4-pro': {
        maxContextLength: 128000,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: true,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'DeepSeek V4 Pro - high-capability reasoning and coding model on Workers AI'
      },
      '@cf/tinyllama/tinyllama-1.1b-chat-v1.0': {
        maxContextLength: 2048,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: true,
        inputTokenCost: 0.00000001,
        outputTokenCost: 0.00000001,
        description: 'TinyLlama - Ultra fast and lightweight'
      },
      '@cf/qwen/qwen1.5-0.5b-chat': {
        maxContextLength: 2048,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: true,
        inputTokenCost: 0.00000001,
        outputTokenCost: 0.00000001,
        description: 'Qwen 1.5 0.5B - Compact and efficient'
      },
      '@cf/qwen/qwen1.5-1.8b-chat': {
        maxContextLength: 4096,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: true,
        inputTokenCost: 0.00000002,
        outputTokenCost: 0.00000002,
        description: 'Qwen 1.5 1.8B - Good balance'
      },
      '@cf/qwen/qwen1.5-14b-chat-awq': {
        maxContextLength: 8192,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: true,
        inputTokenCost: 0.0000002,
        outputTokenCost: 0.0000002,
        description: 'Qwen 1.5 14B - High capability'
      },
      '@cf/qwen/qwen1.5-7b-chat-awq': {
        maxContextLength: 8192,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: true,
        inputTokenCost: 0.0000001,
        outputTokenCost: 0.0000001,
        description: 'Qwen 1.5 7B - Optimized performance'
      },
      '@cf/google/gemma-4-26b-a4b-it': {
        maxContextLength: 256000,
        supportsStreaming: true,
        supportsTools: true,
        toolCalling: true,
        supportsVision: true,
        supportsBatching: true,
        inputTokenCost: 0.0000001,
        outputTokenCost: 0.0000003,
        description: 'Gemma 4 26B — vision + tools + reasoning, 256K context'
      },
      '@cf/meta/llama-4-scout-17b-16e-instruct': {
        maxContextLength: 131000,
        supportsStreaming: true,
        supportsTools: true,
        toolCalling: true,
        supportsVision: true,
        supportsBatching: true,
        inputTokenCost: 0.0000003,
        outputTokenCost: 0.0000009,
        description: 'Llama 4 Scout 17B — natively multimodal, tool calling'
      },
      '@cf/meta/llama-3.2-11b-vision-instruct': {
        maxContextLength: 128000,
        supportsStreaming: true,
        supportsTools: false,
        supportsVision: true,
        supportsBatching: true,
        inputTokenCost: 0.0000005,
        outputTokenCost: 0.0000005,
        description: 'Llama 3.2 11B Vision — image understanding'
      },
      '@cf/meta/llama-3.3-70b-instruct-fp8-fast': {
        maxContextLength: 131072,
        supportsStreaming: true,
        supportsTools: true,
        toolCalling: true,
        supportsBatching: true,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'Llama 3.3 70B FP8 Fast — best quality/cost on Workers AI, primary COST_EFFECTIVE choice'
      },
      '@cf/openai/gpt-oss-20b': {
        maxContextLength: 131072,
        supportsStreaming: true,
        supportsTools: true,
        toolCalling: true,
        supportsBatching: true,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'GPT-OSS 20B — lightweight tool-calling model'
      },
      '@cf/qwen/qwen2.5-coder-32b-instruct': {
        maxContextLength: 32768,
        supportsStreaming: true,
        supportsTools: true,
        toolCalling: true,
        supportsBatching: true,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'Qwen 2.5 Coder 32B — purpose-built for code generation'
      },
      '@cf/mistralai/mistral-small-3.1-24b-instruct': {
        maxContextLength: 131072,
        supportsStreaming: true,
        supportsTools: true,
        toolCalling: true,
        supportsVision: true,
        supportsBatching: true,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'Mistral Small 3.1 24B — vision + tool-calling, strong balanced model'
      },
      '@cf/qwen/qwen3-30b-a3b-fp8': {
        maxContextLength: 40960,
        supportsStreaming: true,
        supportsTools: true,
        toolCalling: true,
        supportsBatching: false,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'Qwen3 30B FP8 — state-of-the-art Qwen3 generation'
      },
      '@cf/meta/llama-3.2-1b-instruct': {
        maxContextLength: 131072,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: true,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'Llama 3.2 1B — ultra-cheap tiny model'
      },
      '@cf/meta/llama-3.2-3b-instruct': {
        maxContextLength: 131072,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: true,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'Llama 3.2 3B — cheap small model'
      },
      '@cf/moonshotai/kimi-k2.7-code': {
        maxContextLength: 131072,
        supportsStreaming: true,
        supportsTools: true,
        toolCalling: true,
        supportsBatching: false,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'Kimi K2.7 Code — code-focused variant of Kimi K2.6'
      },
      '@cf/nvidia/nemotron-3-120b-a12b': {
        maxContextLength: 256000,
        supportsStreaming: true,
        supportsTools: true,
        toolCalling: true,
        supportsBatching: true,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'NVIDIA Nemotron-3 120B — hybrid MoE, parallel function calling, 256K context, multi-agent focus'
      },
      '@cf/deepseek-ai/deepseek-r1-distill-qwen-32b': {
        maxContextLength: 80000,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: true,
        inputTokenCost: 0,
        outputTokenCost: 0,
        thinkingModel: true,
        description: 'DeepSeek-R1-Distill-Qwen-32B — chain-of-thought reasoning model distilled from DeepSeek-R1; outputs thinking traces'
      },
      '@cf/qwen/qwq-32b': {
        maxContextLength: 24000,
        supportsStreaming: true,
        supportsTools: false,
        supportsBatching: true,
        inputTokenCost: 0,
        outputTokenCost: 0,
        thinkingModel: true,
        description: 'QwQ-32B — native thinking/reasoning model; outputs chain-of-thought traces; competes with o1-mini on hard reasoning tasks'
      },
      'anthropic/claude-opus-4.8': {
        maxContextLength: 1000000,
        supportsStreaming: false,
        supportsTools: false,
        supportsBatching: false,
        inputTokenCost: 0,
        outputTokenCost: 0,
        description: 'Cloudflare-managed Anthropic Claude Opus 4.8 — 1M context frontier model; billing via Cloudflare dashboard'
      },
    };
  }

  private async runLlamaVisionRaw(request: LLMRequest, model: string): Promise<WorkersAIResult> {
    if (request.images!.length > 1) {
      throw new ConfigurationError(
        this.name,
        `${model} supports exactly one image via the raw binding format — ${request.images!.length} were provided.`
      );
    }

    const image = request.images![0];

    let imageBytes: number[];
    if (image.data) {
      imageBytes = Array.from(Uint8Array.from(atob(image.data), c => c.charCodeAt(0)));
    } else if (image.url?.startsWith('data:')) {
      const b64 = image.url.split(',')[1] ?? '';
      imageBytes = Array.from(Uint8Array.from(atob(b64), c => c.charCodeAt(0)));
    } else {
      throw new ConfigurationError(
        this.name,
        `${model} requires base64 image data or a data: URL — HTTP URLs are not supported.`
      );
    }

    const systemPrefix = request.systemPrompt ? `${request.systemPrompt}\n\n` : '';
    let lastUserText = '';
    for (let i = request.messages.length - 1; i >= 0; i--) {
      if (request.messages[i].role === 'user') {
        const raw = request.messages[i].content;
        lastUserText = typeof raw === 'string'
          ? raw
          : Array.isArray(raw)
            ? (raw as Array<{ type?: string; text?: string }>)
                .filter(p => p.type === 'text')
                .map(p => p.text ?? '')
                .join(' ')
            : '';
        break;
      }
    }

    return this.runModel(model, {
      image: imageBytes,
      prompt: `${systemPrefix}${lastUserText}`,
      max_tokens: request.maxTokens ?? 512
    }, request) as Promise<WorkersAIResult>;
  }

  private runModel(model: string, input: unknown, request: LLMRequest): Promise<unknown> {
    const options = this.buildRunOptions(request);
    const ai = this.ai as {
      run(model: string, input: unknown, options?: CloudflareRunOptions): Promise<unknown>;
    };

    return options ? ai.run(model, input, options) : ai.run(model, input);
  }

  private buildRunOptions(request: LLMRequest): CloudflareRunOptions | undefined {
    const options: CloudflareRunOptions = {};
    const cacheStrategy = request.cache?.strategy;

    if (
      request.cache?.sessionId &&
      (cacheStrategy === 'provider-prefix' || cacheStrategy === 'both')
    ) {
      options.extraHeaders = { 'x-session-affinity': request.cache.sessionId };
    }

    const gateway = this.buildGatewayOptions(request);
    if (gateway) {
      options.gateway = gateway;
    }

    return Object.keys(options).length > 0 ? options : undefined;
  }

  private buildGatewayOptions(request: LLMRequest): CloudflareAIGatewayOptions | undefined {
    if (!this.gateway) {
      return undefined;
    }

    const metadata: CloudflareAIGatewayOptions['metadata'] = {
      ...(this.gateway.metadata ?? {}),
      ...(request.gatewayMetadata?.customMetadata ?? {}),
      ...(request.gatewayMetadata?.requestId ? { requestId: request.gatewayMetadata.requestId } : {}),
      ...(request.requestId ? { llmRequestId: request.requestId } : {}),
      ...(request.tenantId ? { tenantId: request.tenantId } : {}),
    };

    return {
      ...this.gateway,
      ...(request.gatewayMetadata?.cacheKey ? { cacheKey: request.gatewayMetadata.cacheKey } : {}),
      ...(typeof request.gatewayMetadata?.cacheTtl === 'number' ? { cacheTtl: request.gatewayMetadata.cacheTtl } : {}),
      ...(Object.keys(metadata).length > 0 ? { metadata } : {}),
    };
  }

  private formatRequest(request: LLMRequest, model: string): CloudflareRequest {
    // Anthropic-via-CF models use a different wire format: system is a top-level
    // field, messages must not contain a system role, and tool calling is not
    // yet verified. Route these through a dedicated formatter.
    if (ANTHROPIC_VIA_CF_MODELS.has(model)) {
      return this.formatAnthropicViaCloudflareRequest(request, model);
    }

    const capabilities = this.getModelCapabilities()[model];
    const usesTools =
      (request.tools?.length ?? 0) > 0 ||
      request.messages.some(message =>
        (message.toolCalls?.length ?? 0) > 0 || (message.toolResults?.length ?? 0) > 0
      );

    if (usesTools && !capabilities?.supportsTools) {
      throw new ConfigurationError(
        this.name,
        `Model '${model}' does not support tool calling on Cloudflare Workers AI`
      );
    }

    const hasImages = (request.images?.length ?? 0) > 0;
    if (hasImages && !capabilities?.supportsVision) {
      throw new ConfigurationError(
        this.name,
        `Model '${model}' does not support image input. Use a vision-capable model like @cf/google/gemma-4-26b-a4b-it, @cf/meta/llama-4-scout-17b-16e-instruct, or @cf/meta/llama-3.2-11b-vision-instruct.`
      );
    }

    const messages: CloudflareMessage[] = [];
    const jsonMode = request.response_format?.type === 'json_object';
    const jsonInstruction = '\n\nYou must respond with valid JSON only. No markdown fences, no commentary, no text outside the JSON.';

    // Add system prompt if provided
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

    // Convert messages
    for (const message of request.messages) {
      if (message.role === 'system' && request.systemPrompt) {
        continue;
      }

      const cloudflareMessage: CloudflareMessage = {
        role: message.role as CloudflareMessage['role'],
        content: message.content
      };

      if (message.toolCalls && message.toolCalls.length > 0) {
        cloudflareMessage.tool_calls = message.toolCalls.map(toolCall => ({
          id: toolCall.id,
          type: toolCall.type,
          function: toolCall.function
        }));
        cloudflareMessage.content = null;
      }

      messages.push(cloudflareMessage);

      if (message.toolResults && message.toolResults.length > 0) {
        for (const toolResult of message.toolResults) {
          messages.push({
            role: 'tool',
            content: toolResult.error
              ? JSON.stringify({
                  output: toolResult.output,
                  error: toolResult.error
                })
              : toolResult.output,
            tool_call_id: toolResult.id
          });
        }
      }
    }

    if (hasImages) {
      this.attachImagesToLastUserMessage(messages, request.images!, model);
    }

    const cloudflareRequest: CloudflareRequest = {
      messages,
      temperature: request.temperature,
      max_tokens: request.maxTokens,
      stream: request.stream
    };

    if (request.lora !== undefined) {
      cloudflareRequest.lora = request.lora;
    }

    if (request.topP !== undefined) {
      cloudflareRequest.top_p = request.topP;
    }

    if (request.frequencyPenalty !== undefined) {
      cloudflareRequest.frequency_penalty = request.frequencyPenalty;
    }

    if (request.tools && request.tools.length > 0) {
      cloudflareRequest.tools = request.tools.map(tool => ({
        type: tool.type,
        function: tool.function
      }));

      if (request.toolChoice) {
        cloudflareRequest.tool_choice = request.toolChoice;
      }
    }

    return cloudflareRequest;
  }

  private formatAnthropicViaCloudflareRequest(request: LLMRequest, model: string): CloudflareRequest {
    const capabilities = this.getModelCapabilities()[model];

    if ((request.tools?.length ?? 0) > 0) {
      throw new ConfigurationError(
        this.name,
        `Tool calling for '${model}' via the Cloudflare binding has not been verified — ` +
        `use the direct Anthropic provider for tool-enabled requests.`
      );
    }

    if ((request.images?.length ?? 0) > 0 && !capabilities?.supportsVision) {
      throw new ConfigurationError(
        this.name,
        `'${model}' vision support via the Cloudflare binding has not been verified.`
      );
    }

    // Anthropic format: system is a top-level field, not a role in messages.
    const messages: CloudflareMessage[] = request.messages
      .filter(m => m.role !== 'system')
      .map(m => ({
        role: m.role as CloudflareMessage['role'],
        content: m.content,
      }));

    const cfRequest: CloudflareRequest = {
      messages,
      max_tokens: request.maxTokens,
    };

    const systemParts: string[] = [];
    const systemMessage = request.messages.find(m => m.role === 'system');
    if (systemMessage) systemParts.push(systemMessage.content as string);
    if (request.systemPrompt) systemParts.push(request.systemPrompt);
    if (systemParts.length > 0) cfRequest.system = systemParts.join('\n\n');

    if (request.temperature !== undefined) cfRequest.temperature = request.temperature;
    if (request.topP !== undefined) cfRequest.top_p = request.topP;

    return cfRequest;
  }

  private attachImagesToLastUserMessage(
    messages: CloudflareMessage[],
    images: NonNullable<LLMRequest['images']>,
    model: string
  ): void {
    const lastUserIndex = (() => {
      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === 'user') return i;
      }
      return -1;
    })();

    if (lastUserIndex === -1) {
      throw new ConfigurationError(
        this.name,
        `Vision request must include at least one user message (model: ${model})`
      );
    }

    const existing = messages[lastUserIndex].content;
    const text = typeof existing === 'string' ? existing : '';

    const parts: CloudflareContentPart[] = [{ type: 'text', text }];
    for (const image of images) {
      const url = this.buildImageDataUrl(image, model);
      parts.push({ type: 'image_url', image_url: { url } });
    }

    messages[lastUserIndex].content = parts;
  }

  private buildImageDataUrl(image: LLMImageInput, model: string): string {
    if (image.data) {
      const mime = image.mimeType ?? 'image/jpeg';
      return `data:${mime};base64,${image.data}`;
    }
    if (image.url?.startsWith('data:')) {
      return image.url;
    }
    throw new ConfigurationError(
      this.name,
      `Cloudflare vision models (${model}) require base64 image data or a data: URL. HTTP image URLs are not supported — fetch the image and pass bytes in image.data.`
    );
  }

  private formatResponse(
    result: WorkersAIResult,
    model: string,
    request: LLMRequest,
    responseTime: number
  ): LLMResponse {
    const payload = this.unwrapResult(result);
    // Only validate when the payload is an object — some models return a raw
    // string which is handled by extractText as a valid response shape.
    if (payload !== null && typeof payload === 'object' && !Array.isArray(payload)) {
      validateSchema('cloudflare', payload, CLOUDFLARE_RESPONSE_SCHEMA);
    }
    const content = this.extractText(result);
    const toolCalls = this.extractToolCalls(result);
    const usage = this.extractUsage(result, model, request, content);

    const response: LLMResponse = {
      id: payload?.id,
      message: content,
      content,
      usage,
      model: payload?.model || model,
      provider: this.name,
      responseTime,
      finishReason: this.extractFinishReason(result, toolCalls),
      metadata: {
        cloudflareAI: true,
        accountId: this.accountId
      }
    };

    if (toolCalls.length > 0) {
      response.toolCalls = this.validateToolCalls(toolCalls);
    }

    return response;
  }

  private extractText(result: WorkersAIResult): string {
    const payload = this.unwrapResult(result);

    if (typeof payload === 'string') {
      return payload;
    }

    if (typeof payload?.response === 'string') {
      return payload.response;
    }
    // CF sometimes returns numeric responses (e.g. "4" for "2+2") as a bare number.
    if (typeof payload?.response === 'number') {
      return String(payload.response);
    }

    const chatContent = payload?.choices?.[0]?.message?.content;
    if (typeof chatContent === 'string') {
      return chatContent;
    }

    const reasoningContent = payload?.choices?.[0]?.message?.reasoning_content;
    if (typeof reasoningContent === 'string' && reasoningContent.length > 0) {
      return reasoningContent;
    }

    if (chatContent === null) {
      return '';
    }

    // Anthropic-via-CF format: top-level content array (e.g. anthropic/claude-opus-4.8)
    if (Array.isArray(payload?.content)) {
      return (payload.content as Array<{ type?: string; text?: string }>)
        .filter(p => p?.type === 'text' && typeof p.text === 'string')
        .map(p => p.text ?? '')
        .join('');
    }
    if (typeof payload?.content === 'string') {
      return payload.content;
    }

    if (Array.isArray(chatContent)) {
      return chatContent
        .map((part: { type?: string; text?: string }) => (typeof part?.text === 'string' ? part.text : ''))
        .join('');
    }

    if (typeof payload?.output_text === 'string') {
      return payload.output_text;
    }

    if (Array.isArray(payload?.output)) {
      return payload.output
        .flatMap((item: WorkersAIOutputItem) => {
          if (item?.type === 'message' && Array.isArray(item.content)) {
            return item.content
              .map((part: { type?: string; text?: string }) =>
                part?.type === 'output_text' || part?.type === 'text' ? part.text ?? '' : ''
              )
              .filter(Boolean);
          }

          if ((item?.type === 'output_text' || item?.type === 'text') && typeof item.text === 'string') {
            return [item.text];
          }

          return [];
        })
        .join('');
    }

    return JSON.stringify(payload ?? '');
  }

  private extractToolCalls(result: WorkersAIResult): ToolCall[] {
    const payload = this.unwrapResult(result);
    const choiceToolCalls = payload?.choices?.[0]?.message?.tool_calls;
    if (Array.isArray(choiceToolCalls) && choiceToolCalls.length > 0) {
      return choiceToolCalls.map((toolCall: WorkersAIToolCall, index: number) => ({
        id: toolCall.id || `call_${index}`,
        type: 'function' as const,
        function: {
          name: toolCall.function?.name || 'unknown',
          arguments: this.stringifyArguments(toolCall.function?.arguments)
        }
      }));
    }

    if (Array.isArray(payload?.output)) {
      return payload.output
        .filter((item: WorkersAIOutputItem) => item?.type === 'function_call' && item.name)
        .map((item: WorkersAIOutputItem, index: number) => ({
          id: item.call_id || item.id || `call_${index}`,
          type: 'function' as const,
          function: {
            name: item.name!,
            arguments: this.stringifyArguments(item.arguments)
          }
        }));
    }

    return [];
  }

  private extractUsage(
    result: WorkersAIResult,
    model: string,
    request: LLMRequest,
    content: string
  ): TokenUsage {
    const payload = this.unwrapResult(result);
    const usage = payload?.usage;
    const inputTokens = usage?.prompt_tokens ?? usage?.input_tokens;
    const outputTokens = usage?.completion_tokens ?? usage?.output_tokens;
    const totalTokens = usage?.total_tokens;

    if (
      typeof inputTokens === 'number' ||
      typeof outputTokens === 'number' ||
      typeof totalTokens === 'number'
    ) {
      const normalizedInputTokens =
        typeof inputTokens === 'number'
          ? inputTokens
          : Math.max((totalTokens ?? 0) - (outputTokens ?? 0), 0);
      const normalizedOutputTokens =
        typeof outputTokens === 'number'
          ? outputTokens
          : Math.max((totalTokens ?? 0) - normalizedInputTokens, 0);
      const normalizedTotalTokens =
        typeof totalTokens === 'number'
          ? totalTokens
          : normalizedInputTokens + normalizedOutputTokens;

      const normalizedUsage: TokenUsage = {
        inputTokens: normalizedInputTokens,
        outputTokens: normalizedOutputTokens,
        totalTokens: normalizedTotalTokens,
        cost: this.calculateCost(normalizedInputTokens, normalizedOutputTokens, model)
      };

      const cachedInputTokens = this.extractCachedInputTokens(usage);
      if (typeof cachedInputTokens === 'number') {
        normalizedUsage.cachedInputTokens = cachedInputTokens;
      }

      return normalizedUsage;
    }

    const estimatedInputTokens =
      (request.systemPrompt ? Math.ceil(request.systemPrompt.length / 4) : 0) +
      request.messages.reduce((sum, message) => sum + Math.ceil(message.content.length / 4), 0);
    const estimatedOutputTokens = Math.ceil(content.length / 4);

    return {
      inputTokens: estimatedInputTokens,
      outputTokens: estimatedOutputTokens,
      totalTokens: estimatedInputTokens + estimatedOutputTokens,
      cost: this.calculateCost(estimatedInputTokens, estimatedOutputTokens, model)
    };
  }

  private extractCachedInputTokens(usage: WorkersAIUsage | undefined): number | undefined {
    if (!usage) {
      return undefined;
    }

    const cachedTokens =
      usage.prompt_tokens_details?.cached_tokens ??
      usage.input_tokens_details?.cached_tokens ??
      usage.cached_tokens;

    return typeof cachedTokens === 'number' ? cachedTokens : undefined;
  }

  private extractFinishReason(
    result: WorkersAIResult,
    toolCalls: ToolCall[]
  ): 'stop' | 'length' | 'tool_calls' | 'content_filter' {
    const payload = this.unwrapResult(result);
    const finishReason = payload?.choices?.[0]?.finish_reason;
    if (
      finishReason === 'stop' ||
      finishReason === 'length' ||
      finishReason === 'tool_calls' ||
      finishReason === 'content_filter'
    ) {
      return finishReason;
    }

    // Anthropic-via-CF format: stop_reason field
    const stopReason = payload?.stop_reason;
    if (stopReason === 'end_turn') return 'stop';
    if (stopReason === 'max_tokens') return 'length';
    if (stopReason === 'tool_use') return 'tool_calls';

    if (toolCalls.length > 0) {
      return 'tool_calls';
    }

    return 'stop';
  }

  private stringifyArguments(argumentsValue: unknown): string {
    if (typeof argumentsValue === 'string') {
      return argumentsValue;
    }

    return JSON.stringify(argumentsValue ?? {});
  }

  private unwrapResult(result: WorkersAIResult): WorkersAIResult | undefined {
    if (result && typeof result === 'object' && 'result' in result && result.result) {
      return result.result;
    }

    return result;
  }

  // CF's Workers AI binding throws AiError with "Bad input: ..." for schema validation
  // failures. Wrapping these as InvalidRequestError lets callers distinguish bad-input
  // (non-retryable) from transient failures without parsing raw error messages.
  private wrapCFError(error: unknown): unknown {
    if (error instanceof Error && error.message.includes('Bad input')) {
      return new InvalidRequestError(this.name, error.message);
    }
    return error;
  }

  /**
   * Stream response support
   */
  async streamResponse(request: LLMRequest): Promise<ReadableStream<string>> {
    this.validateRequest(request);

    const model = request.model || this.getRecommendedModel(request);
    const cloudflareRequest = { ...this.formatRequest(request, model), stream: true };
    const usageTracker = createStreamUsageTracker();
    let finalUsage: TokenUsage | undefined;
    let streamedContent = '';

    const resolveUsage = () => {
      usageTracker.resolve(
        finalUsage ?? this.extractUsage({ response: streamedContent }, model, request, streamedContent)
      );
    };

    const stream = new ReadableStream<string>({
      start: async (controller) => {
        try {
          const stream = await this.runModel(model, cloudflareRequest, request);

          if (stream instanceof ReadableStream) {
            const reader = stream.getReader();

            while (true) {
              const { done, value } = await reader.read();

              if (done) {
                break;
              }

              // Handle different chunk formats
              if (typeof value === 'string') {
                streamedContent += value;
                controller.enqueue(value);
              } else if (value != null && typeof value === 'object') {
                const chunk = value as WorkersAIResult;
                if (chunk.usage) {
                  finalUsage = this.extractUsage(chunk, model, request, streamedContent);
                }
                if (typeof chunk.response === 'string') {
                  streamedContent += chunk.response;
                  controller.enqueue(chunk.response);
                }
              }
            }
          } else {
            // Non-streaming response, send all at once
            const content = typeof stream === 'string' ? stream : this.extractText(stream as WorkersAIResult);
            if (typeof stream === 'object' && stream !== null && !Array.isArray(stream)) {
              const result = stream as WorkersAIResult;
              if (result.usage) {
                finalUsage = this.extractUsage(result, model, request, content);
              }
            }
            streamedContent += content;
            controller.enqueue(content);
          }
          resolveUsage();
          controller.close();
        } catch (error) {
          usageTracker.resolve(undefined);
          controller.error(this.wrapCFError(error));
        }
      }
    });

    return attachStreamUsage(stream, usageTracker.promise);
  }

  /**
   * Batch processing support
   */
  async processBatch(requests: LLMRequest[]): Promise<LLMResponse[]> {
    // Process requests concurrently but with rate limiting
    const batchSize = 5; // Process 5 at a time to avoid overwhelming
    const responses: LLMResponse[] = [];

    for (let i = 0; i < requests.length; i += batchSize) {
      const batch = requests.slice(i, i + batchSize);

      const batchPromises = batch.map(async (request) => {
        try {
          return await this.generateResponse(request);
        } catch (error) {
          // Return error response for failed requests
          return {
            message: '',
            usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0, cost: 0 },
            model: request.model || this.getRecommendedModel(request),
            provider: this.name,
            responseTime: 0,
            metadata: { error: (error as Error).message }
          };
        }
      });

      const batchResults = await Promise.all(batchPromises);
      responses.push(...batchResults);
    }

    return responses;
  }

  /**
   * Get recommended model for cost optimization
   */
  getRecommendedModel(request: LLMRequest): string {
    return getProviderDefaultModel('cloudflare', request);
  }

  /**
   * Cost optimization features
   */
  async generateWithCostOptimization(request: LLMRequest): Promise<LLMResponse> {
    // Automatically select the most cost-effective model
    const optimizedRequest = {
      ...request,
      model: request.model || this.getRecommendedModel(request)
    };

    return this.generateResponse(optimizedRequest);
  }
}
