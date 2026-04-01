/**
 * Cloudflare AI Provider
 * Implementation for Cloudflare Workers AI with cost optimization
 */

import type {
  LLMRequest,
  LLMResponse,
  CloudflareConfig,
  ModelCapabilities,
  TokenUsage,
  ToolCall
} from '../types';
import { BaseProvider } from './base';
import {
  ConfigurationError,
  ModelNotFoundError
} from '../errors';

interface CloudflareMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
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
}

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
    '@cf/tinyllama/tinyllama-1.1b-chat-v1.0',
    '@cf/qwen/qwen1.5-0.5b-chat',
    '@cf/qwen/qwen1.5-1.8b-chat',
    '@cf/qwen/qwen1.5-14b-chat-awq',
    '@cf/qwen/qwen1.5-7b-chat-awq'
  ];
  supportsStreaming = true;
  supportsTools = true;
  supportsBatching = true;

  private ai: Ai;
  private accountId?: string;

  constructor(config: CloudflareConfig) {
    super(config);

    if (!config.ai) {
      throw new ConfigurationError('cloudflare', 'Cloudflare AI binding is required');
    }

    this.ai = config.ai;
    this.accountId = config.accountId;
  }

  async generateResponse(request: LLMRequest): Promise<LLMResponse> {
    this.validateRequest(request);
    
    const startTime = Date.now();

    try {
      const response = await this.executeWithResiliency(async () => {
        const model = request.model || '@cf/meta/llama-3.1-8b-instruct';
        const cloudflareRequest = this.formatRequest(request, model);
        
        // Validate model is supported
        if (!this.models.includes(model)) {
          throw new ModelNotFoundError('cloudflare', model);
        }

        const result = await this.ai.run(model as any, cloudflareRequest);
        
        return this.formatResponse(result, model, request, Date.now() - startTime);
      });

      this.updateMetrics(response.responseTime, true, response.usage?.cost || 0);
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
    return !!(this.ai);
  }

  getModels(): string[] {
    return [...this.models];
  }

  estimateCost(request: LLMRequest): number {
    // Cloudflare AI is essentially "free" as it's included in Workers compute
    // But we can estimate the computational cost
    const model = request.model || '@cf/meta/llama-3.1-8b-instruct';
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

      await this.ai.run('@cf/meta/llama-3.1-8b-instruct' as any, testRequest);
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
        inputTokenCost: 0.0000008,
        outputTokenCost: 0.0000008,
        description: 'GPT-OSS 120B - OpenAI-format tool calling on Workers AI'
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
      }
    };
  }

  private formatRequest(request: LLMRequest, model: string): CloudflareRequest {
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

    const cloudflareRequest: CloudflareRequest = {
      messages,
      temperature: request.temperature,
      max_tokens: request.maxTokens,
      stream: request.stream
    };

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

  private formatResponse(
    result: any,
    model: string,
    request: LLMRequest,
    responseTime: number
  ): LLMResponse {
    const payload = this.unwrapResult(result);
    const content = this.extractText(result);
    const toolCalls = this.extractToolCalls(result);
    const usage = this.extractUsage(result, model, request, content);

    const response: LLMResponse = {
      id: typeof payload === 'object' && payload !== null ? payload.id : undefined,
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
      response.toolCalls = toolCalls;
    }

    return response;
  }

  private extractText(result: any): string {
    const payload = this.unwrapResult(result);

    if (typeof payload === 'string') {
      return payload;
    }

    if (typeof payload?.response === 'string') {
      return payload.response;
    }

    const chatContent = payload?.choices?.[0]?.message?.content;
    if (typeof chatContent === 'string') {
      return chatContent;
    }

    if (chatContent === null) {
      return '';
    }

    if (Array.isArray(chatContent)) {
      return chatContent
        .map((part: any) => (typeof part?.text === 'string' ? part.text : ''))
        .join('');
    }

    if (typeof payload?.output_text === 'string') {
      return payload.output_text;
    }

    if (Array.isArray(payload?.output)) {
      return payload.output
        .flatMap((item: any) => {
          if (item?.type === 'message' && Array.isArray(item.content)) {
            return item.content
              .map((part: any) =>
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

  private extractToolCalls(result: any): ToolCall[] {
    const payload = this.unwrapResult(result);
    const choiceToolCalls = payload?.choices?.[0]?.message?.tool_calls;
    if (Array.isArray(choiceToolCalls) && choiceToolCalls.length > 0) {
      return choiceToolCalls.map((toolCall: any, index: number) => ({
        id: toolCall.id || `call_${index}`,
        type: 'function',
        function: {
          name: toolCall.function?.name || 'unknown',
          arguments: this.stringifyArguments(toolCall.function?.arguments)
        }
      }));
    }

    if (Array.isArray(payload?.output)) {
      return payload.output
        .filter((item: any) => item?.type === 'function_call' && item.name)
        .map((item: any, index: number) => ({
          id: item.call_id || item.id || `call_${index}`,
          type: 'function',
          function: {
            name: item.name,
            arguments: this.stringifyArguments(item.arguments)
          }
        }));
    }

    return [];
  }

  private extractUsage(
    result: any,
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

      return {
        inputTokens: normalizedInputTokens,
        outputTokens: normalizedOutputTokens,
        totalTokens: normalizedTotalTokens,
        cost: this.calculateCost(normalizedInputTokens, normalizedOutputTokens, model)
      };
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

  private extractFinishReason(
    result: any,
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

  private unwrapResult(result: any): any {
    if (result && typeof result === 'object' && 'result' in result && result.result) {
      return result.result;
    }

    return result;
  }

  /**
   * Stream response support
   */
  async streamResponse(request: LLMRequest): Promise<ReadableStream<string>> {
    this.validateRequest(request);

    const model = request.model || '@cf/meta/llama-3.1-8b-instruct';
    const cloudflareRequest = { ...this.formatRequest(request, model), stream: true };

    return new ReadableStream({
      start: async (controller) => {
        try {
          // Cloudflare AI streaming support
          const stream = await this.ai.run(model as any, cloudflareRequest);
          
          if (stream instanceof ReadableStream) {
            const reader = stream.getReader();
            
            while (true) {
              const { done, value } = await reader.read();
              
              if (done) {
                controller.close();
                break;
              }
              
              // Handle different chunk formats
              if (typeof value === 'string') {
                controller.enqueue(value);
              } else if (value?.response) {
                controller.enqueue(value.response);
              } else if (value?.delta?.content) {
                controller.enqueue(value.delta.content);
              }
            }
          } else {
            // Non-streaming response, send all at once
            const content = typeof stream === 'string' ? stream : stream?.response || '';
            controller.enqueue(content);
            controller.close();
          }
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
            usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
            model: request.model || '@cf/meta/llama-3.1-8b-instruct',
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
    const messageLength = request.messages.reduce((sum, msg) => sum + msg.content.length, 0);
    const maxTokens = request.maxTokens || 1000;

    // For short responses, use the fastest/cheapest model
    if (messageLength < 500 && maxTokens < 500) {
      return '@cf/tinyllama/tinyllama-1.1b-chat-v1.0';
    }

    // For medium complexity tasks
    if (messageLength < 2000 && maxTokens < 2000) {
      return '@cf/qwen/qwen1.5-7b-chat-awq';
    }

    // For complex tasks, use the best model
    return '@cf/meta/llama-3.1-8b-instruct';
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
