/**
 * Cloudflare AI Provider
 * Implementation for Cloudflare Workers AI with cost optimization
 */

import type { LLMRequest, LLMResponse, CloudflareConfig, ModelCapabilities } from '../types';
import { BaseProvider } from './base';
import { 
  LLMErrorFactory, 
  ConfigurationError, 
  ModelNotFoundError 
} from '../errors';

interface CloudflareMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface CloudflareRequest {
  messages: CloudflareMessage[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
}

interface CloudflareResponse {
  result: {
    response: string;
    success: boolean;
    errors?: string[];
    messages?: string[];
  };
  success: boolean;
  errors: any[];
  messages: any[];
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
    '@cf/tinyllama/tinyllama-1.1b-chat-v1.0',
    '@cf/qwen/qwen1.5-0.5b-chat',
    '@cf/qwen/qwen1.5-1.8b-chat',
    '@cf/qwen/qwen1.5-14b-chat-awq',
    '@cf/qwen/qwen1.5-7b-chat-awq'
  ];
  supportsStreaming = true;
  supportsTools = false; // Cloudflare AI doesn't support function calling yet
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
        const cloudflareRequest = this.formatRequest(request);
        const model = request.model || '@cf/meta/llama-3.1-8b-instruct';
        
        // Validate model is supported
        if (!this.models.includes(model)) {
          throw new ModelNotFoundError('cloudflare', model);
        }

        const result = await this.ai.run(model as any, cloudflareRequest);
        
        return this.formatResponse(result, model, Date.now() - startTime);
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

  private formatRequest(request: LLMRequest): CloudflareRequest {
    const messages: CloudflareMessage[] = [];

    // Add system prompt if provided
    if (request.systemPrompt) {
      messages.push({
        role: 'system',
        content: request.systemPrompt
      });
    }

    // Convert messages
    for (const message of request.messages) {
      // Skip tool-related messages as Cloudflare AI doesn't support them
      if (message.toolCalls || message.toolResults) {
        continue;
      }

      messages.push({
        role: message.role as 'system' | 'user' | 'assistant',
        content: message.content
      });
    }

    const cloudflareRequest: CloudflareRequest = {
      messages,
      temperature: request.temperature,
      max_tokens: request.maxTokens,
      stream: request.stream
    };

    return cloudflareRequest;
  }

  private formatResponse(
    result: any,
    model: string,
    responseTime: number
  ): LLMResponse {
    // Handle different response formats from Cloudflare AI
    let content = '';
    let finishReason: 'stop' | 'length' | 'tool_calls' | 'content_filter' = 'stop';

    if (typeof result === 'string') {
      content = result;
    } else if (result?.response) {
      content = result.response;
    } else if (result?.result?.response) {
      content = result.result.response;
    } else if (result?.choices?.[0]?.message?.content) {
      content = result.choices[0].message.content;
      finishReason = result.choices[0].finish_reason || 'stop';
    } else {
      content = JSON.stringify(result);
    }

    // Estimate token usage (Cloudflare AI doesn't always provide usage stats)
    const inputTokens = Math.ceil(content.length / 4); // Rough estimation
    const outputTokens = Math.ceil(content.length / 4);

    const usage = {
      inputTokens,
      outputTokens,
      totalTokens: inputTokens + outputTokens,
      cost: this.calculateCost(inputTokens, outputTokens, model)
    };

    return {
      message: content,
      content,
      usage,
      model,
      provider: this.name,
      responseTime,
      finishReason,
      metadata: {
        cloudflareAI: true,
        accountId: this.accountId
      }
    };
  }

  /**
   * Stream response support
   */
  async streamResponse(request: LLMRequest): Promise<ReadableStream<string>> {
    this.validateRequest(request);

    const cloudflareRequest = { ...this.formatRequest(request), stream: true };
    const model = request.model || '@cf/meta/llama-3.1-8b-instruct';

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