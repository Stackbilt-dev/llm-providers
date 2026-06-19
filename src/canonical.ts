import type {
  BuiltInTool,
  CacheHints,
  GatewayMetadata,
  LLMImageInput,
  LLMMessage,
  LLMRequest,
  LLMResponse,
  TokenUsage,
  Tool,
  ToolCall,
  ToolResult,
} from './types.js';
import type { ModelRecommendationUseCase, ProviderName } from './model-catalog.js';

export interface CanonicalMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
  toolCalls?: ToolCall[];
  toolResults?: ToolResult[];
}

export interface CanonicalMediaPart {
  type: 'image';
  data?: string;
  url?: string;
  mimeType?: string;
}

export type CanonicalTool = Tool;

export type CanonicalToolMode =
  | 'auto'
  | 'none'
  | 'required'
  | { toolName: string };

export interface CanonicalOutputFormat {
  kind: 'text' | 'json_object' | 'json_schema';
  schemaName?: string;
  schema?: Record<string, unknown>;
  strict?: boolean;
}

export interface CanonicalSamplingOptions {
  temperature?: number;
  maxTokens?: number;
  seed?: number;
}

export interface CanonicalRequirements {
  toolCalling?: boolean;
  structuredOutput?: boolean;
  vision?: boolean;
  longContext?: boolean;
  builtInTools?: boolean;
}

export interface CanonicalProviderOptions {
  cloudflare?: {
    lora?: string;
    topP?: number;
    frequencyPenalty?: number;
  };
  cerebras?: {
    reasoning?: LLMRequest['reasoning'];
    prediction?: string;
  };
  groq?: Record<string, unknown>;
  nvidia?: Record<string, unknown>;
  openai?: Record<string, unknown>;
  anthropic?: Record<string, unknown>;
}

export interface CanonicalRequestMetadata {
  requestId?: string;
  tenantId?: string;
  gateway?: GatewayMetadata;
  cache?: CacheHints;
  custom?: Record<string, unknown>;
}

export interface CanonicalLLMRequest {
  messages: CanonicalMessage[];
  system?: string;
  model?: string;
  stream?: boolean;
  sampling?: CanonicalSamplingOptions;
  tools?: CanonicalTool[];
  toolMode?: CanonicalToolMode;
  builtInTools?: BuiltInTool[];
  output?: CanonicalOutputFormat;
  media?: CanonicalMediaPart[];
  workload?: ModelRecommendationUseCase;
  requirements?: CanonicalRequirements;
  providerOptions?: CanonicalProviderOptions;
  metadata?: CanonicalRequestMetadata;
}

export interface CanonicalFallbackHop {
  provider: ProviderName;
  model?: string;
  reason?: string;
}

export interface CanonicalDegradation {
  capability: string;
  action: 'stripped' | 'downgraded' | 'emulated' | 'failed';
  reason: string;
}

export interface CanonicalRoutingMetadata {
  selectedProvider: ProviderName;
  selectedModel: string;
  selectionReason?: string;
  fallbackChain?: CanonicalFallbackHop[];
  degradations?: CanonicalDegradation[];
}

export interface CanonicalResponseError {
  class: string;
  provider?: ProviderName;
  retryable?: boolean;
  message: string;
}

export interface CanonicalLLMResponse {
  id?: string;
  reasoning?: string;
  message: string;
  model: string;
  provider: ProviderName;
  finishReason?: 'stop' | 'length' | 'tool_calls' | 'content_filter' | 'error';
  toolCalls?: ToolCall[];
  usage: TokenUsage;
  responseTime: number;
  routing?: CanonicalRoutingMetadata;
  error?: CanonicalResponseError;
  metadata?: Record<string, unknown>;
}

const MODEL_USE_CASES = new Set<string>([
  'COST_EFFECTIVE',
  'HIGH_PERFORMANCE',
  'BALANCED',
  'TOOL_CALLING',
  'LONG_CONTEXT',
  'VISION',
  'RESEARCH',
]);

export function normalizeLLMRequest(request: LLMRequest | CanonicalLLMRequest): CanonicalLLMRequest {
  if (isCanonicalLLMRequest(request)) {
    return normalizeCanonicalInput(request);
  }

  const legacyMetadata = copyRecord(request.metadata);
  const systemMessage = request.messages.find(message => message.role === 'system');
  const output = normalizeOutputFormat(request.response_format);
  const media = request.images?.map(imageToMediaPart);
  const providerOptions = normalizeProviderOptions(request);
  const workload = normalizeWorkload(legacyMetadata?.useCase);
  const requirements = normalizeRequirements(request, output, media, workload);

  return {
    messages: request.messages
      .filter(message => message.role !== 'system')
      .map(copyMessage),
    system: request.systemPrompt ?? systemMessage?.content,
    model: request.model,
    stream: request.stream,
    sampling: {
      temperature: request.temperature,
      maxTokens: request.maxTokens,
      seed: request.seed,
    },
    tools: request.tools?.map(copyTool),
    toolMode: normalizeToolMode(request.toolChoice),
    builtInTools: request.builtInTools?.map(tool => ({ ...tool })),
    output,
    media,
    workload,
    requirements,
    providerOptions,
    metadata: {
      requestId: request.requestId,
      tenantId: request.tenantId,
      gateway: request.gatewayMetadata,
      cache: request.cache,
      custom: legacyMetadata,
    },
  };
}

export function canonicalToLLMRequest(request: CanonicalLLMRequest): LLMRequest {
  const customMetadata = copyRecord(request.metadata?.custom) ?? {};
  if (request.workload && customMetadata.useCase === undefined) {
    customMetadata.useCase = request.workload;
  }

  const legacy: LLMRequest = {
    messages: request.messages.map(copyMessage),
    model: request.model,
    stream: request.stream,
    systemPrompt: request.system,
    temperature: request.sampling?.temperature,
    maxTokens: request.sampling?.maxTokens,
    seed: request.sampling?.seed,
    tools: request.tools?.map(copyTool),
    builtInTools: request.builtInTools?.map(tool => ({ ...tool })),
    toolChoice: canonicalToolModeToLegacy(request.toolMode),
    response_format: canonicalOutputToLegacy(request.output),
    images: request.media?.map(mediaPartToImage),
    tenantId: request.metadata?.tenantId,
    requestId: request.metadata?.requestId,
    gatewayMetadata: request.metadata?.gateway,
    cache: request.metadata?.cache,
    metadata: Object.keys(customMetadata).length > 0 ? customMetadata : undefined,
  };

  const cloudflare = request.providerOptions?.cloudflare;
  if (cloudflare) {
    legacy.lora = cloudflare.lora;
    legacy.topP = cloudflare.topP;
    legacy.frequencyPenalty = cloudflare.frequencyPenalty;
  }

  const cerebras = request.providerOptions?.cerebras;
  if (cerebras) {
    legacy.reasoning = cerebras.reasoning;
    legacy.prediction = cerebras.prediction;
  }

  return legacy;
}

export function normalizeLLMResponse(
  response: LLMResponse,
  options: {
    routing?: CanonicalRoutingMetadata;
    error?: CanonicalResponseError;
  } = {}
): CanonicalLLMResponse {
  return {
    id: response.id,
    reasoning: response.reasoning,
    message: response.message,
    model: response.model,
    provider: response.provider as ProviderName,
    finishReason: response.finishReason,
    toolCalls: response.toolCalls,
    usage: response.usage,
    responseTime: response.responseTime,
    routing: options.routing ?? {
      selectedProvider: response.provider as ProviderName,
      selectedModel: response.model,
    },
    error: options.error,
    metadata: response.metadata,
  };
}

function isCanonicalLLMRequest(request: LLMRequest | CanonicalLLMRequest): request is CanonicalLLMRequest {
  return (
    'sampling' in request ||
    'output' in request ||
    'media' in request ||
    'providerOptions' in request ||
    'requirements' in request ||
    'workload' in request ||
    'system' in request ||
    'toolMode' in request
  );
}

function normalizeCanonicalInput(request: CanonicalLLMRequest): CanonicalLLMRequest {
  return {
    ...request,
    messages: request.messages.map(copyMessage),
    sampling: request.sampling ? { ...request.sampling } : undefined,
    tools: request.tools?.map(copyTool),
    builtInTools: request.builtInTools?.map(tool => ({ ...tool })),
    output: request.output ? { ...request.output } : { kind: 'text' },
    media: request.media?.map(part => ({ ...part })),
    requirements: request.requirements ? { ...request.requirements } : undefined,
    providerOptions: copyProviderOptions(request.providerOptions),
    metadata: request.metadata ? {
      ...request.metadata,
      custom: copyRecord(request.metadata.custom),
    } : undefined,
  };
}

function normalizeOutputFormat(format: LLMRequest['response_format'] | undefined): CanonicalOutputFormat {
  if (!format) {
    return { kind: 'text' };
  }

  if (format.type === 'json_schema') {
    return {
      kind: 'json_schema',
      schemaName: format.json_schema.name,
      schema: format.json_schema.schema,
      strict: format.json_schema.strict,
    };
  }

  return { kind: format.type };
}

function canonicalOutputToLegacy(output: CanonicalOutputFormat | undefined): LLMRequest['response_format'] {
  if (!output || output.kind === 'text') {
    return output?.kind === 'text' ? { type: 'text' } : undefined;
  }

  if (output.kind === 'json_schema') {
    return {
      type: 'json_schema',
      json_schema: {
        name: output.schemaName ?? 'response',
        schema: output.schema ?? {},
        strict: output.strict,
      },
    };
  }

  return { type: output.kind };
}

function normalizeToolMode(toolChoice: LLMRequest['toolChoice'] | undefined): CanonicalToolMode | undefined {
  if (!toolChoice) return undefined;
  if (toolChoice === 'auto' || toolChoice === 'none') return toolChoice;
  return { toolName: toolChoice.function.name };
}

function canonicalToolModeToLegacy(toolMode: CanonicalToolMode | undefined): LLMRequest['toolChoice'] {
  if (!toolMode) return undefined;
  if (toolMode === 'auto' || toolMode === 'none') return toolMode;
  if (toolMode === 'required') return 'auto';
  return { type: 'function', function: { name: toolMode.toolName } };
}

function normalizeProviderOptions(request: LLMRequest): CanonicalProviderOptions | undefined {
  const cloudflare: NonNullable<CanonicalProviderOptions['cloudflare']> = {};
  if (request.lora !== undefined) cloudflare.lora = request.lora;
  if (request.topP !== undefined) cloudflare.topP = request.topP;
  if (request.frequencyPenalty !== undefined) cloudflare.frequencyPenalty = request.frequencyPenalty;

  const cerebras: NonNullable<CanonicalProviderOptions['cerebras']> = {};
  if (request.reasoning !== undefined) cerebras.reasoning = request.reasoning;
  if (request.prediction !== undefined) cerebras.prediction = request.prediction;

  const providerOptions: CanonicalProviderOptions = {};
  if (Object.keys(cloudflare).length > 0) providerOptions.cloudflare = cloudflare;
  if (Object.keys(cerebras).length > 0) providerOptions.cerebras = cerebras;

  return Object.keys(providerOptions).length > 0 ? providerOptions : undefined;
}

function normalizeRequirements(
  request: LLMRequest,
  output: CanonicalOutputFormat,
  media: CanonicalMediaPart[] | undefined,
  workload: ModelRecommendationUseCase | undefined
): CanonicalRequirements | undefined {
  const requirements: CanonicalRequirements = {};
  const hasToolProtocol =
    (request.tools?.length ?? 0) > 0 ||
    request.messages.some(message =>
      (message.toolCalls?.length ?? 0) > 0 || (message.toolResults?.length ?? 0) > 0
    );

  if (hasToolProtocol) requirements.toolCalling = true;
  if (output.kind !== 'text') requirements.structuredOutput = true;
  if ((media?.length ?? 0) > 0) requirements.vision = true;
  if ((request.builtInTools?.length ?? 0) > 0) requirements.builtInTools = true;
  if (workload === 'LONG_CONTEXT') requirements.longContext = true;

  return Object.keys(requirements).length > 0 ? requirements : undefined;
}

function normalizeWorkload(value: unknown): ModelRecommendationUseCase | undefined {
  if (typeof value !== 'string') return undefined;
  const normalized = value.toUpperCase();
  return MODEL_USE_CASES.has(normalized)
    ? normalized as ModelRecommendationUseCase
    : undefined;
}

function copyMessage(message: LLMMessage | CanonicalMessage): CanonicalMessage {
  return {
    role: message.role,
    content: message.content,
    timestamp: message.timestamp,
    toolCalls: message.toolCalls?.map(copyToolCall),
    toolResults: message.toolResults?.map(copyToolResult),
  };
}

function copyTool(tool: Tool): Tool {
  return {
    type: tool.type,
    function: {
      name: tool.function.name,
      description: tool.function.description,
      parameters: {
        type: tool.function.parameters.type,
        properties: { ...tool.function.parameters.properties },
        required: tool.function.parameters.required
          ? [...tool.function.parameters.required]
          : undefined,
      },
    },
  };
}

function copyToolCall(toolCall: ToolCall): ToolCall {
  return {
    id: toolCall.id,
    type: toolCall.type,
    function: { ...toolCall.function },
  };
}

function copyToolResult(toolResult: ToolResult): ToolResult {
  return { ...toolResult };
}

function imageToMediaPart(image: LLMImageInput): CanonicalMediaPart {
  return {
    type: 'image',
    data: image.data,
    url: image.url,
    mimeType: image.mimeType,
  };
}

function mediaPartToImage(part: CanonicalMediaPart): LLMImageInput {
  return {
    data: part.data,
    url: part.url,
    mimeType: part.mimeType,
  };
}

function copyRecord(value: Record<string, unknown> | undefined): Record<string, unknown> | undefined {
  return value ? { ...value } : undefined;
}

function copyProviderOptions(options: CanonicalProviderOptions | undefined): CanonicalProviderOptions | undefined {
  if (!options) return undefined;
  return {
    cloudflare: options.cloudflare ? { ...options.cloudflare } : undefined,
    cerebras: options.cerebras ? { ...options.cerebras } : undefined,
    groq: copyRecord(options.groq),
    nvidia: copyRecord(options.nvidia),
    openai: copyRecord(options.openai),
    anthropic: copyRecord(options.anthropic),
  };
}
