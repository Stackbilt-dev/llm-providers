/**
 * Cloudflare AI Gateway passthrough (cfGateway) tests
 * Covers URL derivation, constructor validation, header injection/omission,
 * and the baseUrl override regression across all five HTTP providers.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { OpenAIProvider } from '../providers/openai';
import { AnthropicProvider } from '../providers/anthropic';
import { CerebrasProvider } from '../providers/cerebras';
import { GroqProvider } from '../providers/groq';
import { NvidiaProvider } from '../providers/nvidia';
import { CfGatewayInvalidConfigError } from '../errors';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import type { LLMRequest } from '../types';

const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

const ACCOUNT = 'acct-123';
const GATEWAY = 'gw-abc';

const chatCompletionBody = (model: string) => ({
  ok: true,
  json: async () => ({
    id: 'chatcmpl-123',
    object: 'chat.completion',
    created: 1700000000,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: 'Hi!' }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  }),
  headers: new Headers({ 'content-type': 'application/json' }),
});

const anthropicBody = () => ({
  ok: true,
  json: async () => ({
    id: 'msg_123',
    type: 'message',
    role: 'assistant',
    model: 'claude-3-5-haiku-20241022',
    content: [{ type: 'text', text: 'Hi!' }],
    stop_reason: 'end_turn',
    usage: { input_tokens: 10, output_tokens: 5 },
  }),
  headers: new Headers({ 'content-type': 'application/json' }),
});

const baseRequest: LLMRequest = {
  messages: [{ role: 'user', content: 'Hello' }],
  maxTokens: 100,
};

beforeEach(() => {
  vi.clearAllMocks();
  defaultCircuitBreakerManager.resetAll();
});

describe('cfGateway URL derivation', () => {
  const expectedUrl = (suffix: string) =>
    `https://gateway.ai.cloudflare.com/v1/${ACCOUNT}/${GATEWAY}/${suffix}`;

  it('derives the OpenAI gateway URL', async () => {
    mockFetch.mockResolvedValueOnce(chatCompletionBody('gpt-4o-mini'));
    const provider = new OpenAIProvider({ apiKey: 'k', cfGateway: { accountId: ACCOUNT, gatewayId: GATEWAY } });
    await provider.generateResponse({ ...baseRequest, model: 'gpt-4o-mini' });
    expect(mockFetch.mock.calls[0][0]).toBe(`${expectedUrl('openai/v1')}/chat/completions`);
  });

  it('derives the Anthropic gateway URL (no version segment)', async () => {
    mockFetch.mockResolvedValueOnce(anthropicBody());
    const provider = new AnthropicProvider({ apiKey: 'k', cfGateway: { accountId: ACCOUNT, gatewayId: GATEWAY } });
    await provider.generateResponse({ ...baseRequest, model: 'claude-3-5-haiku-20241022' });
    expect(mockFetch.mock.calls[0][0]).toBe(`${expectedUrl('anthropic')}/v1/messages`);
  });

  it('derives the Cerebras gateway URL', async () => {
    mockFetch.mockResolvedValueOnce(chatCompletionBody('llama-3.1-8b'));
    const provider = new CerebrasProvider({ apiKey: 'k', cfGateway: { accountId: ACCOUNT, gatewayId: GATEWAY } });
    await provider.generateResponse({ ...baseRequest, model: 'llama-3.1-8b' });
    expect(mockFetch.mock.calls[0][0]).toBe(`${expectedUrl('cerebras/v1')}/chat/completions`);
  });

  it('derives the Groq gateway URL', async () => {
    mockFetch.mockResolvedValueOnce(chatCompletionBody('llama-3.3-70b-versatile'));
    const provider = new GroqProvider({ apiKey: 'k', cfGateway: { accountId: ACCOUNT, gatewayId: GATEWAY } });
    await provider.generateResponse({ ...baseRequest, model: 'llama-3.3-70b-versatile' });
    expect(mockFetch.mock.calls[0][0]).toBe(`${expectedUrl('groq/openai/v1')}/chat/completions`);
  });

  it('derives the NVIDIA gateway URL', async () => {
    mockFetch.mockResolvedValueOnce(chatCompletionBody('meta/llama-3.3-70b-instruct'));
    const provider = new NvidiaProvider({ apiKey: 'k', cfGateway: { accountId: ACCOUNT, gatewayId: GATEWAY } });
    await provider.generateResponse({ ...baseRequest, model: 'meta/llama-3.3-70b-instruct' });
    expect(mockFetch.mock.calls[0][0]).toBe(`${expectedUrl('nvidia-nim/v1')}/chat/completions`);
  });
});

describe('cfGateway constructor validation', () => {
  it('throws synchronously on empty accountId', () => {
    expect(() => new OpenAIProvider({ apiKey: 'k', cfGateway: { accountId: '', gatewayId: GATEWAY } }))
      .toThrow(CfGatewayInvalidConfigError);
    try {
      new OpenAIProvider({ apiKey: 'k', cfGateway: { accountId: '', gatewayId: GATEWAY } });
    } catch (err) {
      expect((err as CfGatewayInvalidConfigError).code).toBe('CF_GATEWAY_INVALID_CONFIG');
      expect((err as CfGatewayInvalidConfigError).field).toBe('accountId');
      expect((err as CfGatewayInvalidConfigError).provider).toBe('openai');
      expect((err as Error).message).toContain('openai');
    }
  });

  it('throws synchronously on empty gatewayId', () => {
    try {
      new GroqProvider({ apiKey: 'k', cfGateway: { accountId: ACCOUNT, gatewayId: '' } });
      throw new Error('should have thrown');
    } catch (err) {
      expect(err).toBeInstanceOf(CfGatewayInvalidConfigError);
      expect((err as CfGatewayInvalidConfigError).field).toBe('gatewayId');
      expect((err as CfGatewayInvalidConfigError).provider).toBe('groq');
    }
  });
});

describe('cfGateway baseUrl override', () => {
  it('explicit baseUrl wins and disables header injection', async () => {
    mockFetch.mockResolvedValueOnce(chatCompletionBody('gpt-4o-mini'));
    const provider = new OpenAIProvider({
      apiKey: 'k',
      baseUrl: 'https://custom.example.com/v1',
      cfGateway: { accountId: ACCOUNT, gatewayId: GATEWAY },
    });
    await provider.generateResponse({
      ...baseRequest,
      model: 'gpt-4o-mini',
      gatewayMetadata: { cacheTtl: 300, cacheKey: 'k1', skipCache: false },
    });
    const [url, options] = mockFetch.mock.calls[0];
    expect(url).toBe('https://custom.example.com/v1/chat/completions');
    expect(options.headers['cf-aig-cache-ttl']).toBeUndefined();
    expect(options.headers['cf-aig-cache-key']).toBeUndefined();
    expect(options.headers['cf-aig-skip-cache']).toBeUndefined();
  });
});

describe('cfGateway header injection', () => {
  it('injects all cf-aig-* headers when gatewayMetadata fields are set', async () => {
    mockFetch.mockResolvedValueOnce(chatCompletionBody('gpt-4o-mini'));
    const provider = new OpenAIProvider({ apiKey: 'k', cfGateway: { accountId: ACCOUNT, gatewayId: GATEWAY } });
    await provider.generateResponse({
      ...baseRequest,
      model: 'gpt-4o-mini',
      gatewayMetadata: { cacheTtl: 300, cacheKey: 'my-key', skipCache: true },
    });
    const headers = mockFetch.mock.calls[0][1].headers;
    expect(headers['cf-aig-cache-ttl']).toBe('300');
    expect(headers['cf-aig-cache-key']).toBe('my-key');
    expect(headers['cf-aig-skip-cache']).toBe('true');
  });

  it('serializes skipCache=false as the string "false"', async () => {
    mockFetch.mockResolvedValueOnce(chatCompletionBody('gpt-4o-mini'));
    const provider = new OpenAIProvider({ apiKey: 'k', cfGateway: { accountId: ACCOUNT, gatewayId: GATEWAY } });
    await provider.generateResponse({
      ...baseRequest,
      model: 'gpt-4o-mini',
      gatewayMetadata: { skipCache: false },
    });
    const headers = mockFetch.mock.calls[0][1].headers;
    expect(headers['cf-aig-skip-cache']).toBe('false');
  });

  it('omits headers whose gatewayMetadata fields are undefined', async () => {
    mockFetch.mockResolvedValueOnce(chatCompletionBody('gpt-4o-mini'));
    const provider = new OpenAIProvider({ apiKey: 'k', cfGateway: { accountId: ACCOUNT, gatewayId: GATEWAY } });
    await provider.generateResponse({
      ...baseRequest,
      model: 'gpt-4o-mini',
      gatewayMetadata: { cacheTtl: 60 },
    });
    const headers = mockFetch.mock.calls[0][1].headers;
    expect(headers['cf-aig-cache-ttl']).toBe('60');
    expect(headers['cf-aig-cache-key']).toBeUndefined();
    expect(headers['cf-aig-skip-cache']).toBeUndefined();
  });

  it('omits all cf-aig-* headers when gatewayMetadata is absent', async () => {
    mockFetch.mockResolvedValueOnce(chatCompletionBody('gpt-4o-mini'));
    const provider = new OpenAIProvider({ apiKey: 'k', cfGateway: { accountId: ACCOUNT, gatewayId: GATEWAY } });
    await provider.generateResponse({ ...baseRequest, model: 'gpt-4o-mini' });
    const headers = mockFetch.mock.calls[0][1].headers;
    expect(headers['cf-aig-cache-ttl']).toBeUndefined();
    expect(headers['cf-aig-cache-key']).toBeUndefined();
    expect(headers['cf-aig-skip-cache']).toBeUndefined();
  });

  it('does not inject cf-aig-* headers when gateway is inactive (default base URL)', async () => {
    mockFetch.mockResolvedValueOnce(chatCompletionBody('gpt-4o-mini'));
    const provider = new OpenAIProvider({ apiKey: 'k' });
    await provider.generateResponse({
      ...baseRequest,
      model: 'gpt-4o-mini',
      gatewayMetadata: { cacheTtl: 300, cacheKey: 'k1', skipCache: true },
    });
    const headers = mockFetch.mock.calls[0][1].headers;
    expect(headers['cf-aig-skip-cache']).toBeUndefined();
  });
});
