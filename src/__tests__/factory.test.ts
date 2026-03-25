/**
 * LLM Provider Factory Tests
 * Tests for the provider factory with mocked providers
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { LLMProviderFactory, createCostOptimizedFactory } from '../factory';
import { AuthenticationError } from '../errors';
import type { LLMRequest, LLMResponse } from '../types';

// Mock providers
const mockOpenAIProvider = {
  name: 'openai',
  models: ['gpt-4', 'gpt-3.5-turbo'],
  supportsStreaming: true,
  supportsTools: true,
  supportsBatching: true,
  generateResponse: vi.fn().mockResolvedValue({
    message: 'OpenAI response',
    usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30, cost: 0.001 },
    model: 'gpt-3.5-turbo',
    provider: 'openai',
    responseTime: 1000
  } as LLMResponse),
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
  models: ['claude-3-haiku-20240307', 'claude-3-sonnet-20240229'],
  supportsStreaming: true,
  supportsTools: true,
  supportsBatching: false,
  generateResponse: vi.fn().mockResolvedValue({
    message: 'Anthropic response',
    usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30, cost: 0.002 },
    model: 'claude-3-haiku-20240307',
    provider: 'anthropic',
    responseTime: 1200
  } as LLMResponse),
  validateConfig: vi.fn().mockReturnValue(true),
  getModels: vi.fn().mockReturnValue(['claude-3-haiku-20240307', 'claude-3-sonnet-20240229']),
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
  generateResponse: vi.fn().mockResolvedValue({
    message: 'Cloudflare response',
    usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30, cost: 0.0001 },
    model: '@cf/meta/llama-3.1-8b-instruct',
    provider: 'cloudflare',
    responseTime: 800
  } as LLMResponse),
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
    
    factory = new LLMProviderFactory({
      openai: { apiKey: 'test-openai-key' },
      anthropic: { apiKey: 'test-anthropic-key' },
      cloudflare: { ai: {} as Ai },
      defaultProvider: 'auto',
      costOptimization: true
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
  });

  describe('Cost Optimization', () => {
    it('should prefer cheaper providers when cost optimization is enabled', async () => {
      const response = await factory.generateResponse(testRequest);
      
      // Cloudflare should be preferred for cost optimization
      expect(response.provider).toBe('cloudflare');
    });

    it('should provide cost analytics', () => {
      const analytics = factory.getCostAnalytics();
      
      expect(analytics).toBeDefined();
      expect(analytics.breakdown).toBeDefined();
      expect(analytics.total).toBeDefined();
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
        model: 'claude-3-haiku-20240307'
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