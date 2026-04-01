/**
 * LLMProviders.fromEnv() Tests
 * Tests for auto-discovery of providers from a Worker env object
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { LLMProviders } from '../index';
import { ConfigurationError } from '../errors';

// Mock all provider constructors so we never make real API calls
const mockValidateConfig = vi.fn().mockReturnValue(true);

vi.mock('../providers/openai', () => ({
  OpenAIProvider: vi.fn().mockImplementation(() => ({
    name: 'openai',
    models: ['gpt-4o'],
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: true,
    validateConfig: mockValidateConfig,
    generateResponse: vi.fn(),
    getModels: vi.fn().mockReturnValue(['gpt-4o']),
    estimateCost: vi.fn().mockReturnValue(0),
    healthCheck: vi.fn().mockResolvedValue(true),
    getMetrics: vi.fn().mockReturnValue({
      requestCount: 0, successCount: 0, errorCount: 0,
      averageLatency: 0, totalCost: 0, rateLimitHits: 0, lastUsed: 0
    }),
    resetMetrics: vi.fn()
  }))
}));

vi.mock('../providers/anthropic', () => ({
  AnthropicProvider: vi.fn().mockImplementation(() => ({
    name: 'anthropic',
    models: ['claude-3-haiku-20240307'],
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    validateConfig: mockValidateConfig,
    generateResponse: vi.fn(),
    getModels: vi.fn().mockReturnValue(['claude-3-haiku-20240307']),
    estimateCost: vi.fn().mockReturnValue(0),
    healthCheck: vi.fn().mockResolvedValue(true),
    getMetrics: vi.fn().mockReturnValue({
      requestCount: 0, successCount: 0, errorCount: 0,
      averageLatency: 0, totalCost: 0, rateLimitHits: 0, lastUsed: 0
    }),
    resetMetrics: vi.fn()
  }))
}));

vi.mock('../providers/cloudflare', () => ({
  CloudflareProvider: vi.fn().mockImplementation(() => ({
    name: 'cloudflare',
    models: ['@cf/meta/llama-3.1-8b-instruct'],
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: true,
    validateConfig: mockValidateConfig,
    generateResponse: vi.fn(),
    getModels: vi.fn().mockReturnValue(['@cf/meta/llama-3.1-8b-instruct']),
    estimateCost: vi.fn().mockReturnValue(0),
    healthCheck: vi.fn().mockResolvedValue(true),
    getMetrics: vi.fn().mockReturnValue({
      requestCount: 0, successCount: 0, errorCount: 0,
      averageLatency: 0, totalCost: 0, rateLimitHits: 0, lastUsed: 0
    }),
    resetMetrics: vi.fn()
  }))
}));

vi.mock('../providers/cerebras', () => ({
  CerebrasProvider: vi.fn().mockImplementation(() => ({
    name: 'cerebras',
    models: ['llama-3.1-8b'],
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    validateConfig: mockValidateConfig,
    generateResponse: vi.fn(),
    getModels: vi.fn().mockReturnValue(['llama-3.1-8b']),
    estimateCost: vi.fn().mockReturnValue(0),
    healthCheck: vi.fn().mockResolvedValue(true),
    getMetrics: vi.fn().mockReturnValue({
      requestCount: 0, successCount: 0, errorCount: 0,
      averageLatency: 0, totalCost: 0, rateLimitHits: 0, lastUsed: 0
    }),
    resetMetrics: vi.fn()
  }))
}));

vi.mock('../providers/groq', () => ({
  GroqProvider: vi.fn().mockImplementation(() => ({
    name: 'groq',
    models: ['llama-3.3-70b-versatile'],
    supportsStreaming: true,
    supportsTools: false,
    supportsBatching: false,
    validateConfig: mockValidateConfig,
    generateResponse: vi.fn(),
    getModels: vi.fn().mockReturnValue(['llama-3.3-70b-versatile']),
    estimateCost: vi.fn().mockReturnValue(0),
    healthCheck: vi.fn().mockResolvedValue(true),
    getMetrics: vi.fn().mockReturnValue({
      requestCount: 0, successCount: 0, errorCount: 0,
      averageLatency: 0, totalCost: 0, rateLimitHits: 0, lastUsed: 0
    }),
    resetMetrics: vi.fn()
  }))
}));

describe('LLMProviders.fromEnv', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockValidateConfig.mockReturnValue(true);
  });

  // --- Detection ---

  it('should detect Anthropic from ANTHROPIC_API_KEY', () => {
    const providers = LLMProviders.fromEnv({ ANTHROPIC_API_KEY: 'sk-ant-test' });
    expect(providers.getAvailableProviders()).toContain('anthropic');
  });

  it('should detect OpenAI from OPENAI_API_KEY', () => {
    const providers = LLMProviders.fromEnv({ OPENAI_API_KEY: 'sk-test' });
    expect(providers.getAvailableProviders()).toContain('openai');
  });

  it('should detect Groq from GROQ_API_KEY', () => {
    const providers = LLMProviders.fromEnv({ GROQ_API_KEY: 'gsk-test' });
    expect(providers.getAvailableProviders()).toContain('groq');
  });

  it('should detect Cerebras from CEREBRAS_API_KEY', () => {
    const providers = LLMProviders.fromEnv({ CEREBRAS_API_KEY: 'csk-test' });
    expect(providers.getAvailableProviders()).toContain('cerebras');
  });

  it('should detect Cloudflare Workers AI from AI binding object', () => {
    const providers = LLMProviders.fromEnv({ AI: {} as Ai });
    expect(providers.getAvailableProviders()).toContain('cloudflare');
  });

  it('should detect multiple providers at once', () => {
    const providers = LLMProviders.fromEnv({
      OPENAI_API_KEY: 'sk-test',
      ANTHROPIC_API_KEY: 'sk-ant-test',
      GROQ_API_KEY: 'gsk-test',
      CEREBRAS_API_KEY: 'csk-test',
      AI: {} as Ai,
    });

    const available = providers.getAvailableProviders();
    expect(available).toContain('openai');
    expect(available).toContain('anthropic');
    expect(available).toContain('groq');
    expect(available).toContain('cerebras');
    expect(available).toContain('cloudflare');
  });

  // --- Rejection of invalid values ---

  it('should ignore empty string API keys', () => {
    expect(() => LLMProviders.fromEnv({ OPENAI_API_KEY: '' })).toThrow(ConfigurationError);
  });

  it('should ignore non-string API key values', () => {
    expect(() =>
      LLMProviders.fromEnv({ OPENAI_API_KEY: 123 as unknown })
    ).toThrow(ConfigurationError);
  });

  it('should ignore AI binding when it is a string (not an object)', () => {
    expect(() =>
      LLMProviders.fromEnv({ AI: 'not-a-binding' })
    ).toThrow(ConfigurationError);
  });

  it('should ignore AI binding when it is null', () => {
    expect(() =>
      LLMProviders.fromEnv({ AI: null })
    ).toThrow(ConfigurationError);
  });

  // --- No providers ---

  it('should throw ConfigurationError when env is empty', () => {
    expect(() => LLMProviders.fromEnv({})).toThrow(ConfigurationError);
  });

  it('should throw ConfigurationError when env has no recognized keys', () => {
    expect(() => LLMProviders.fromEnv({ UNRELATED_VAR: 'hello' })).toThrow(ConfigurationError);
  });

  it('should include helpful message listing expected keys', () => {
    expect(() => LLMProviders.fromEnv({})).toThrow(/ANTHROPIC_API_KEY/);
    expect(() => LLMProviders.fromEnv({})).toThrow(/OPENAI_API_KEY/);
    expect(() => LLMProviders.fromEnv({})).toThrow(/AI binding/);
  });

  // --- Overrides ---

  it('should apply defaultProvider override', () => {
    const providers = LLMProviders.fromEnv(
      { OPENAI_API_KEY: 'sk-test', ANTHROPIC_API_KEY: 'sk-ant-test' },
      { defaultProvider: 'anthropic' }
    );
    // The instance was constructed without error; we verify through available providers
    expect(providers.getAvailableProviders()).toContain('openai');
    expect(providers.getAvailableProviders()).toContain('anthropic');
  });

  it('should apply costOptimization override', () => {
    const providers = LLMProviders.fromEnv(
      { OPENAI_API_KEY: 'sk-test' },
      { costOptimization: true }
    );
    // Cost analytics should reflect that optimization is enabled
    const analytics = providers.getCostAnalytics();
    expect(analytics).toBeDefined();
    // When cost optimization is on, analytics includes breakdown
    expect(analytics.breakdown).toBeDefined();
  });

  it('should apply enableCircuitBreaker override', () => {
    const providers = LLMProviders.fromEnv(
      { OPENAI_API_KEY: 'sk-test' },
      { enableCircuitBreaker: true }
    );
    expect(providers).toBeInstanceOf(LLMProviders);
  });

  it('should apply enableRetries override', () => {
    const providers = LLMProviders.fromEnv(
      { OPENAI_API_KEY: 'sk-test' },
      { enableRetries: true }
    );
    expect(providers).toBeInstanceOf(LLMProviders);
  });

  it('should apply fallbackRules override', () => {
    const providers = LLMProviders.fromEnv(
      { OPENAI_API_KEY: 'sk-test', ANTHROPIC_API_KEY: 'sk-ant-test' },
      { fallbackRules: [{ condition: 'error', fallbackProvider: 'anthropic' }] }
    );
    expect(providers).toBeInstanceOf(LLMProviders);
  });

  it('should work with no overrides argument', () => {
    const providers = LLMProviders.fromEnv({ OPENAI_API_KEY: 'sk-test' });
    expect(providers).toBeInstanceOf(LLMProviders);
  });

  // --- Return type ---

  it('should return a fully functional LLMProviders instance', () => {
    const providers = LLMProviders.fromEnv({ OPENAI_API_KEY: 'sk-test' });
    expect(providers).toBeInstanceOf(LLMProviders);
    expect(typeof providers.generateResponse).toBe('function');
    expect(typeof providers.getProvider).toBe('function');
    expect(typeof providers.getAvailableProviders).toBe('function');
    expect(typeof providers.getHealth).toBe('function');
    expect(typeof providers.getCostAnalytics).toBe('function');
    expect(typeof providers.reset).toBe('function');
    expect(typeof providers.updateConfig).toBe('function');
  });

  // --- Ignores unrelated env keys ---

  it('should not be affected by unrelated env keys', () => {
    const providers = LLMProviders.fromEnv({
      OPENAI_API_KEY: 'sk-test',
      DATABASE_URL: 'postgres://...',
      MY_SECRET: 'secret-value',
    });
    const available = providers.getAvailableProviders();
    expect(available).toEqual(['openai']);
  });
});
