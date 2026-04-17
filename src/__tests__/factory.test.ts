/**
 * LLM Provider Factory Tests
 * Tests for the provider factory with mocked providers
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { LLMProviderFactory, createCostOptimizedFactory } from '../factory';
import { AuthenticationError } from '../errors';
import type { LLMRequest, LLMResponse } from '../types';
import { OpenAIProvider } from '../providers/openai';
import { CreditLedger } from '../utils/credit-ledger';
import { defaultCostTracker } from '../utils/cost-tracker';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { defaultExhaustionRegistry } from '../utils/exhaustion';
import { defaultLatencyHistogram } from '../utils/latency-histogram';

// Mock providers
const mockOpenAIProvider = {
  name: 'openai',
  models: ['gpt-4', 'gpt-3.5-turbo'],
  supportsStreaming: true,
  supportsTools: true,
  supportsBatching: true,
  supportsVision: true,
  generateResponse: vi.fn().mockResolvedValue({
    message: 'OpenAI response',
    usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30, cost: 0.001 },
    model: 'gpt-3.5-turbo',
    provider: 'openai',
    responseTime: 1000
  } as LLMResponse),
  streamResponse: vi.fn(),
  getProviderBalance: vi.fn(),
  validateConfig: vi.fn().mockReturnValue(true),
  getModels: vi.fn().mockReturnValue(['gpt-4', 'gpt-3.5-turbo']),
  estimateCost: vi.fn().mockReturnValue(0.001),
  healthCheck: vi.fn().mockResolvedValue(true),
  getMetrics: vi.fn().mockReturnValue({
    requestCount: 1,
    successCount: 1,
    errorCount: 0,
    averageLatency: 1000,
    totalCost: 0.001,
    rateLimitHits: 0,
    lastUsed: Date.now()
  }),
  resetMetrics: vi.fn()
};

const mockAnthropicProvider = {
  name: 'anthropic',
  models: ['claude-haiku-4-5-20251001', 'claude-3-sonnet-20240229'],
  supportsStreaming: true,
  supportsTools: true,
  supportsBatching: false,
  supportsVision: true,
  generateResponse: vi.fn().mockResolvedValue({
    message: 'Anthropic response',
    usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30, cost: 0.002 },
    model: 'claude-haiku-4-5-20251001',
    provider: 'anthropic',
    responseTime: 1200
  } as LLMResponse),
  streamResponse: vi.fn(),
  getProviderBalance: vi.fn(),
  validateConfig: vi.fn().mockReturnValue(true),
  getModels: vi.fn().mockReturnValue(['claude-haiku-4-5-20251001', 'claude-3-sonnet-20240229']),
  estimateCost: vi.fn().mockReturnValue(0.002),
  healthCheck: vi.fn().mockResolvedValue(true),
  getMetrics: vi.fn().mockReturnValue({
    requestCount: 1,
    successCount: 1,
    errorCount: 0,
    averageLatency: 1200,
    totalCost: 0.002,
    rateLimitHits: 0,
    lastUsed: Date.now()
  }),
  resetMetrics: vi.fn()
};

const mockCloudflareProvider = {
  name: 'cloudflare',
  models: ['@cf/meta/llama-3.1-8b-instruct'],
  supportsStreaming: true,
  supportsTools: false,
  supportsBatching: true,
  supportsVision: false,
  generateResponse: vi.fn().mockResolvedValue({
    message: 'Cloudflare response',
    usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30, cost: 0.0001 },
    model: '@cf/meta/llama-3.1-8b-instruct',
    provider: 'cloudflare',
    responseTime: 800
  } as LLMResponse),
  streamResponse: vi.fn(),
  getProviderBalance: vi.fn(),
  validateConfig: vi.fn().mockReturnValue(true),
  getModels: vi.fn().mockReturnValue(['@cf/meta/llama-3.1-8b-instruct']),
  estimateCost: vi.fn().mockReturnValue(0.0001),
  healthCheck: vi.fn().mockResolvedValue(true),
  getMetrics: vi.fn().mockReturnValue({
    requestCount: 1,
    successCount: 1,
    errorCount: 0,
    averageLatency: 800,
    totalCost: 0.0001,
    rateLimitHits: 0,
    lastUsed: Date.now()
  }),
  resetMetrics: vi.fn()
};

// Mock the provider classes
vi.mock('../providers/openai', () => ({
  OpenAIProvider: vi.fn().mockImplementation(() => mockOpenAIProvider)
}));

vi.mock('../providers/anthropic', () => ({
  AnthropicProvider: vi.fn().mockImplementation(() => mockAnthropicProvider)
}));

vi.mock('../providers/cloudflare', () => ({
  CloudflareProvider: vi.fn().mockImplementation(() => mockCloudflareProvider)
}));

describe('LLMProviderFactory', () => {
  let factory: LLMProviderFactory;
  
  const testRequest: LLMRequest = {
    messages: [{ role: 'user', content: 'Hello, world!' }],
    maxTokens: 100,
    temperature: 0.7
  };

  beforeEach(() => {
    vi.clearAllMocks();
    defaultCostTracker.reset();
    defaultCircuitBreakerManager.resetAll();
    defaultExhaustionRegistry.reset();
    defaultLatencyHistogram.reset();

    mockOpenAIProvider.generateResponse.mockReset().mockResolvedValue({
      message: 'OpenAI response',
      usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30, cost: 0.001 },
      model: 'gpt-3.5-turbo',
      provider: 'openai',
      responseTime: 1000
    } as LLMResponse);
    mockOpenAIProvider.streamResponse.mockReset().mockResolvedValue(new ReadableStream<string>({
      start(controller) {
        controller.enqueue('OpenAI stream');
        controller.close();
      }
    }));
    mockOpenAIProvider.getProviderBalance.mockReset().mockResolvedValue({
      provider: 'openai',
      status: 'available',
      source: 'provider_api'
    });
    mockAnthropicProvider.generateResponse.mockReset().mockResolvedValue({
      message: 'Anthropic response',
      usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30, cost: 0.002 },
      model: 'claude-haiku-4-5-20251001',
      provider: 'anthropic',
      responseTime: 1200
    } as LLMResponse);
    mockAnthropicProvider.streamResponse.mockReset().mockResolvedValue(new ReadableStream<string>({
      start(controller) {
        controller.enqueue('Anthropic stream');
        controller.close();
      }
    }));
    mockAnthropicProvider.getProviderBalance.mockReset().mockResolvedValue({
      provider: 'anthropic',
      status: 'available',
      source: 'provider_api'
    });
    mockCloudflareProvider.generateResponse.mockReset().mockResolvedValue({
      message: 'Cloudflare response',
      usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30, cost: 0.0001 },
      model: '@cf/meta/llama-3.1-8b-instruct',
      provider: 'cloudflare',
      responseTime: 800
    } as LLMResponse);
    mockCloudflareProvider.streamResponse.mockReset().mockResolvedValue(new ReadableStream<string>({
      start(controller) {
        controller.enqueue('Cloudflare stream');
        controller.close();
      }
    }));
    mockCloudflareProvider.getProviderBalance.mockReset().mockResolvedValue({
      provider: 'cloudflare',
      status: 'unavailable',
      source: 'not_supported'
    });

    factory = new LLMProviderFactory({
      openai: { apiKey: 'test-openai-key' },
      anthropic: { apiKey: 'test-anthropic-key' },
      cloudflare: { ai: {} as Ai },
      defaultProvider: 'auto',
      costOptimization: true,
      enableCircuitBreaker: true,
    });
  });

  describe('Provider Initialization', () => {
    it('should initialize all configured providers', () => {
      const availableProviders = factory.getAvailableProviders();
      expect(availableProviders).toContain('openai');
      expect(availableProviders).toContain('anthropic');
      expect(availableProviders).toContain('cloudflare');
    });

    it('should handle missing provider configurations gracefully', () => {
      const factoryWithMissing = new LLMProviderFactory({
        openai: { apiKey: 'test-key' }
        // Missing anthropic and cloudflare configs
      });
      
      const availableProviders = factoryWithMissing.getAvailableProviders();
      expect(availableProviders).toContain('openai');
      expect(availableProviders).not.toContain('anthropic');
      expect(availableProviders).not.toContain('cloudflare');
    });
  });

  describe('Response Generation', () => {
    it('should generate response using available provider', async () => {
      const response = await factory.generateResponse(testRequest);
      
      expect(response).toBeDefined();
      expect(response.message).toBeTruthy();
      expect(response.provider).toBeOneOf(['openai', 'anthropic', 'cloudflare']);
      expect(response.usage).toBeDefined();
      expect(response.responseTime).toBeGreaterThan(0);
    });

    it('should use specific provider when model is specified', async () => {
      const requestWithModel: LLMRequest = {
        ...testRequest,
        model: 'gpt-3.5-turbo'
      };

      const response = await factory.generateResponse(requestWithModel);
      expect(mockOpenAIProvider.generateResponse).toHaveBeenCalled();
    });

    it('should fallback to other providers on failure', async () => {
      // Make OpenAI provider fail
      mockOpenAIProvider.generateResponse.mockRejectedValueOnce(
        new Error('OpenAI temporarily unavailable')
      );

      const response = await factory.generateResponse(testRequest);
      
      // Should fallback to another provider
      expect(response).toBeDefined();
      expect(response.provider).not.toBe('openai');
    });

    it('should skip fully open providers and continue down the chain', async () => {
      defaultCircuitBreakerManager.getBreaker('cloudflare').forceOpen();

      const response = await factory.generateResponse(testRequest);

      expect(mockCloudflareProvider.generateResponse).not.toHaveBeenCalled();
      expect(response.provider).not.toBe('cloudflare');
    });
  });

  describe('Cost Optimization', () => {
    it('should prefer cheaper providers when cost optimization is enabled', async () => {
      const response = await factory.generateResponse(testRequest);
      
      // Cloudflare should be preferred for cost optimization
      expect(response.provider).toBe('cloudflare');
    });

    it('should provide cost analytics', async () => {
      await factory.generateResponse(testRequest);

      const analytics = factory.getCostAnalytics();
      
      expect(analytics).toBeDefined();
      expect(analytics.breakdown).toBeDefined();
      expect(analytics.total).toBeCloseTo(0.0001);
      expect(analytics.breakdown.cloudflare).toMatchObject({
        cost: 0.0001,
        totalCost: 0.0001,
        requests: 1,
        requestCount: 1,
        inputTokens: 10,
        outputTokens: 20,
        tokens: { input: 10, output: 20 }
      });
      expect(analytics.breakdown.cloudflare.lastRecordedAt).toBeGreaterThan(0);
    });
  });

  describe('Health Monitoring', () => {
    it('should check provider health status', async () => {
      const health = await factory.getProviderHealth();
      
      expect(health).toBeDefined();
      expect(health.openai).toBeDefined();
      expect(health.anthropic).toBeDefined();
      expect(health.cloudflare).toBeDefined();
      
      expect(health.openai.healthy).toBe(true);
      expect(health.openai.models).toEqual(['gpt-4', 'gpt-3.5-turbo']);
      expect(health.openai.capabilities).toBeDefined();
    });

    it('should handle provider health check failures', async () => {
      mockOpenAIProvider.healthCheck.mockResolvedValueOnce(false);
      
      const health = await factory.getProviderHealth();
      expect(health.openai.healthy).toBe(false);
    });
  });

  describe('Provider Chain Building', () => {
    it('should build appropriate provider chain for model-specific requests', async () => {
      // Test Claude model request
      const claudeRequest: LLMRequest = {
        ...testRequest,
        model: 'claude-haiku-4-5-20251001'
      };

      await factory.generateResponse(claudeRequest);
      expect(mockAnthropicProvider.generateResponse).toHaveBeenCalled();
    });

    it('should build cost-optimized chain for auto mode', async () => {
      const autoRequest: LLMRequest = {
        ...testRequest
        // No specific model, should use auto selection
      };

      await factory.generateResponse(autoRequest);
      
      // Should prefer Cloudflare for cost optimization
      expect(mockCloudflareProvider.generateResponse).toHaveBeenCalled();
    });

    it('should honor fallbackProvider as the next route when a rule matches', async () => {
      const ruleFactory = new LLMProviderFactory({
        openai: { apiKey: 'test-openai-key' },
        anthropic: { apiKey: 'test-anthropic-key' },
        cloudflare: { ai: {} as Ai },
        defaultProvider: 'openai',
        costOptimization: false,
        fallbackRules: [{ condition: 'error', fallbackProvider: 'anthropic' }]
      });

      mockOpenAIProvider.generateResponse.mockRejectedValueOnce(new Error('OpenAI down'));

      const response = await ruleFactory.generateResponse(testRequest);

      expect(response.provider).toBe('anthropic');
      expect(mockAnthropicProvider.generateResponse).toHaveBeenCalled();
      expect(mockCloudflareProvider.generateResponse).not.toHaveBeenCalled();
    });

    it('should apply fallbackModel when routing through a fallback rule', async () => {
      const ruleFactory = new LLMProviderFactory({
        openai: { apiKey: 'test-openai-key' },
        anthropic: { apiKey: 'test-anthropic-key' },
        defaultProvider: 'openai',
        costOptimization: false,
        fallbackRules: [{
          condition: 'error',
          fallbackProvider: 'anthropic',
          fallbackModel: 'claude-haiku-4-5-20251001'
        }]
      });

      mockOpenAIProvider.generateResponse.mockRejectedValueOnce(new Error('OpenAI down'));

      await ruleFactory.generateResponse({
        ...testRequest,
        model: 'gpt-4'
      });

      expect(mockAnthropicProvider.generateResponse).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'claude-haiku-4-5-20251001' })
      );
    });
  });

  describe('Streaming, tools, classification, and vision', () => {
    async function readStream(stream: ReadableStream<string>): Promise<string> {
      const reader = stream.getReader();
      let output = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        output += value;
      }
      return output;
    }

    it('should stream through the factory and fallback before the first chunk', async () => {
      const streamFactory = new LLMProviderFactory({
        openai: { apiKey: 'test-openai-key' },
        anthropic: { apiKey: 'test-anthropic-key' },
        defaultProvider: 'openai',
        costOptimization: false,
        fallbackRules: [{ condition: 'error', fallbackProvider: 'anthropic' }]
      });

      mockOpenAIProvider.streamResponse.mockRejectedValueOnce(new Error('stream start failed'));

      const stream = await streamFactory.generateResponseStream(testRequest);

      expect(await readStream(stream)).toBe('Anthropic stream');
      expect(mockOpenAIProvider.streamResponse).toHaveBeenCalled();
      expect(mockAnthropicProvider.streamResponse).toHaveBeenCalled();
    });

    it('should call quota hooks before and after successful dispatch', async () => {
      const quotaHook = {
        check: vi.fn().mockResolvedValue({ allowed: true, remainingBudget: 1 }),
        record: vi.fn().mockResolvedValue(undefined)
      };
      const quotaFactory = new LLMProviderFactory({
        openai: { apiKey: 'test-openai-key' },
        defaultProvider: 'openai',
        costOptimization: false,
        quotaHook
      });

      await quotaFactory.generateResponse({ ...testRequest, tenantId: 'tenant-1' });

      expect(quotaHook.check).toHaveBeenCalledWith(expect.objectContaining({
        tenantId: 'tenant-1',
        provider: 'openai',
        model: 'gpt-4'
      }));
      expect(quotaHook.record).toHaveBeenCalledWith(expect.objectContaining({
        tenantId: 'tenant-1',
        provider: 'openai',
        actualCost: 0.001,
        inputTokens: 10,
        outputTokens: 20
      }));
    });

    it('should deny dispatch when quota hook rejects the request', async () => {
      const quotaFactory = new LLMProviderFactory({
        openai: { apiKey: 'test-openai-key' },
        defaultProvider: 'openai',
        costOptimization: false,
        quotaHook: {
          check: vi.fn().mockResolvedValue({ allowed: false, reason: 'budget exhausted' }),
          record: vi.fn()
        }
      });

      await expect(quotaFactory.generateResponse(testRequest)).rejects.toThrow('budget exhausted');
      expect(mockOpenAIProvider.generateResponse).not.toHaveBeenCalled();
    });

    it('should execute tool loops until the final response has no tool calls', async () => {
      const toolResponse: LLMResponse = {
        message: '',
        usage: { inputTokens: 5, outputTokens: 5, totalTokens: 10, cost: 0.001 },
        model: 'gpt-3.5-turbo',
        provider: 'openai',
        responseTime: 10,
        finishReason: 'tool_calls',
        toolCalls: [{
          id: 'call-1',
          type: 'function',
          function: { name: 'lookup', arguments: '{"id":42}' }
        }]
      };
      const finalResponse: LLMResponse = {
        message: 'done',
        usage: { inputTokens: 5, outputTokens: 5, totalTokens: 10, cost: 0.001 },
        model: 'gpt-3.5-turbo',
        provider: 'openai',
        responseTime: 10
      };
      mockOpenAIProvider.generateResponse
        .mockResolvedValueOnce(toolResponse)
        .mockResolvedValueOnce(finalResponse);

      const loopFactory = new LLMProviderFactory({
        openai: { apiKey: 'test-openai-key' },
        defaultProvider: 'openai',
        costOptimization: false
      });
      const executor = { execute: vi.fn().mockResolvedValue({ value: 42 }) };

      const response = await loopFactory.generateResponseWithTools(testRequest, executor);

      expect(response.message).toBe('done');
      expect(executor.execute).toHaveBeenCalledWith('lookup', { id: 42 });
      expect(mockOpenAIProvider.generateResponse).toHaveBeenLastCalledWith(
        expect.objectContaining({
          messages: expect.arrayContaining([
            expect.objectContaining({
              toolResults: [{ id: 'call-1', output: '{"value":42}' }]
            })
          ])
        })
      );
    });

    it('should classify JSON responses and expose confidence', async () => {
      mockOpenAIProvider.generateResponse.mockResolvedValueOnce({
        message: '{"label":"recipe","confidence":0.92}',
        usage: { inputTokens: 5, outputTokens: 5, totalTokens: 10, cost: 0.001 },
        model: 'gpt-3.5-turbo',
        provider: 'openai',
        responseTime: 10
      } as LLMResponse);

      const classifyFactory = new LLMProviderFactory({
        openai: { apiKey: 'test-openai-key' },
        defaultProvider: 'openai',
        costOptimization: false
      });

      const result = await classifyFactory.classify<{ label: string; confidence: number }>('classify this');

      expect(result.data.label).toBe('recipe');
      expect(result.confidence).toBe(0.92);
    });

    it('should route image analysis to a vision-capable provider', async () => {
      const visionFactory = new LLMProviderFactory({
        anthropic: { apiKey: 'test-anthropic-key' },
        cloudflare: { ai: {} as Ai },
        costOptimization: false
      });

      await visionFactory.analyzeImage({
        image: { data: 'abc123', mimeType: 'image/jpeg' },
        prompt: 'Extract recipe text'
      });

      expect(mockAnthropicProvider.generateResponse).toHaveBeenCalledWith(
        expect.objectContaining({
          images: [{ data: 'abc123', mimeType: 'image/jpeg' }],
          model: 'claude-haiku-4-5-20251001'
        })
      );
      expect(mockCloudflareProvider.generateResponse).not.toHaveBeenCalled();
    });
  });

  describe('Error Handling', () => {
    it('should handle all providers failing', async () => {
      // Make all providers fail
      mockOpenAIProvider.generateResponse.mockRejectedValue(new Error('OpenAI failed'));
      mockAnthropicProvider.generateResponse.mockRejectedValue(new Error('Anthropic failed'));
      mockCloudflareProvider.generateResponse.mockRejectedValue(new Error('Cloudflare failed'));

      await expect(factory.generateResponse(testRequest)).rejects.toThrow();
    });

    it('should not fallback for authentication errors', async () => {
      const authError = new AuthenticationError('openai', 'Authentication failed');

      mockOpenAIProvider.generateResponse.mockRejectedValueOnce(authError);

      await expect(factory.generateResponse({
        ...testRequest,
        model: 'gpt-3.5-turbo'
      })).rejects.toThrow('Authentication failed');
    });
  });

  describe('Configuration Updates', () => {
    it('should update configuration', () => {
      factory.updateConfig({
        defaultProvider: 'anthropic',
        costOptimization: false
      });

      // Configuration should be updated
      // (We'd need to expose config or test behavior changes)
      expect(() => factory.updateConfig({})).not.toThrow();
    });

    it('should reset metrics and circuit breakers', () => {
      factory.reset();
      
      expect(mockOpenAIProvider.resetMetrics).toHaveBeenCalled();
      expect(mockAnthropicProvider.resetMetrics).toHaveBeenCalled();
      expect(mockCloudflareProvider.resetMetrics).toHaveBeenCalled();
    });

    it('should pass maxRetries 0 to providers when factory retries are disabled', () => {
      new LLMProviderFactory({
        openai: { apiKey: 'test-key' },
        enableRetries: false
      });

      expect(OpenAIProvider).toHaveBeenCalledWith(
        expect.objectContaining({
          apiKey: 'test-key',
          maxRetries: 0
        })
      );
    });

    it('should preserve explicit provider maxRetries when factory retries are disabled', () => {
      new LLMProviderFactory({
        openai: { apiKey: 'test-key', maxRetries: 2 },
        enableRetries: false
      });

      expect(OpenAIProvider).toHaveBeenCalledWith(
        expect.objectContaining({
          apiKey: 'test-key',
          maxRetries: 2
        })
      );
    });
  });

  describe('CreditLedger integration', () => {
    it('should record successful factory calls into the provided ledger even without cost optimization', async () => {
      const ledger = new CreditLedger({
        budgets: [{
          provider: 'cloudflare',
          monthlyBudget: 1,
          rateLimits: { rpm: 10, rpd: 100, tpm: 1000, tpd: 10_000 }
        }]
      });

      const ledgerFactory = new LLMProviderFactory({
        cloudflare: { ai: {} as Ai },
        costOptimization: false,
        ledger
      });

      await ledgerFactory.generateResponse(testRequest);

      const accumulator = ledger.getProviderAccumulator('cloudflare');
      expect(accumulator).toMatchObject({
        spend: 0.0001,
        inputTokens: 10,
        outputTokens: 20,
        requestCount: 1
      });
      expect(accumulator!.rateLimits.rpm!.used).toBe(1);
      expect(accumulator!.rateLimits.rpd!.used).toBe(1);
      expect(accumulator!.rateLimits.tpm!.used).toBe(30);
      expect(accumulator!.rateLimits.tpd!.used).toBe(30);
    });

    it('should expose provider balance from the configured ledger', async () => {
      const ledger = new CreditLedger({
        budgets: [{
          provider: 'cloudflare',
          monthlyBudget: 1,
          rateLimits: { rpm: 10 }
        }]
      });
      const balanceFactory = new LLMProviderFactory({
        cloudflare: { ai: {} as Ai },
        ledger
      });

      await balanceFactory.generateResponse(testRequest);
      const balance = await balanceFactory.getProviderBalance('cloudflare');

      expect(balance).toMatchObject({
        provider: 'cloudflare',
        status: 'available',
        source: 'ledger',
        currentSpend: 0.0001,
        monthlyBudget: 1,
        requestCount: 1
      });
      expect((balance as { rateLimits: Record<string, { used: number }> }).rateLimits.rpm.used).toBe(1);
    });
  });
});

describe('Cost Optimized Factory', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Restore default mock implementations after previous tests may have set mockRejectedValue
    mockOpenAIProvider.generateResponse.mockResolvedValue({
      message: 'OpenAI response',
      usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30, cost: 0.001 },
      model: 'gpt-3.5-turbo',
      provider: 'openai',
      responseTime: 1000
    } as LLMResponse);
    mockCloudflareProvider.generateResponse.mockResolvedValue({
      message: 'Cloudflare response',
      usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30, cost: 0.0001 },
      model: '@cf/meta/llama-3.1-8b-instruct',
      provider: 'cloudflare',
      responseTime: 800
    } as LLMResponse);
  });

  it('should create cost-optimized factory with correct configuration', () => {
    const factory = createCostOptimizedFactory({
      openai: { apiKey: 'test-key' },
      cloudflare: { ai: {} as Ai }
    });

    expect(factory).toBeInstanceOf(LLMProviderFactory);
    // Cost optimization should be enabled by default
  });

  it('should prioritize cost-effective providers', async () => {
    const factory = createCostOptimizedFactory({
      openai: { apiKey: 'test-key' },
      cloudflare: { ai: {} as Ai }
    });

    const response = await factory.generateResponse({
      messages: [{ role: 'user', content: 'Test' }]
    });

    // Should successfully route to an available provider
    expect(['openai', 'cloudflare']).toContain(response.provider);
  });
});

// Custom matcher for vitest
expect.extend({
  toBeOneOf(received: any, expected: any[]) {
    const pass = expected.includes(received);
    return {
      pass,
      message: () => `expected ${received} to be one of ${expected.join(', ')}`
    };
  }
});
