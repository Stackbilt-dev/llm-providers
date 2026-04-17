/**
 * response_format Tests
 * Verify that each provider correctly translates the unified response_format field
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { OpenAIProvider } from '../providers/openai';
import { AnthropicProvider } from '../providers/anthropic';
import { CloudflareProvider } from '../providers/cloudflare';
import { CerebrasProvider } from '../providers/cerebras';
import { GroqProvider } from '../providers/groq';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import type { LLMRequest } from '../types';

// ---------- shared mocks ----------
const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

/** Helper that returns a standard OpenAI-shaped chat completion mock response */
function openAIChatCompletion(model: string, content: string = '{"ok":true}') {
  return {
    ok: true,
    json: async () => ({
      id: 'chatcmpl-test',
      object: 'chat.completion',
      created: 1700000000,
      model,
      choices: [{
        index: 0,
        message: { role: 'assistant', content },
        finish_reason: 'stop'
      }],
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
    }),
    headers: new Headers({ 'content-type': 'application/json' })
  };
}

// ---------- OpenAI ----------
describe('OpenAI response_format', () => {
  let provider: OpenAIProvider;

  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
    provider = new OpenAIProvider({ apiKey: 'test-key' });
  });

  it('should pass response_format through to the API body', async () => {
    mockFetch.mockResolvedValueOnce(openAIChatCompletion('gpt-4o-mini'));

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'Give me JSON' }],
      model: 'gpt-4o-mini',
      response_format: { type: 'json_object' }
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.response_format).toEqual({ type: 'json_object' });
  });

  it('should not include response_format when not set', async () => {
    mockFetch.mockResolvedValueOnce(openAIChatCompletion('gpt-4o-mini'));

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'Hello' }],
      model: 'gpt-4o-mini'
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.response_format).toBeUndefined();
  });

  it('should pass response_format with type text', async () => {
    mockFetch.mockResolvedValueOnce(openAIChatCompletion('gpt-4o-mini', 'plain text'));

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'Hello' }],
      model: 'gpt-4o-mini',
      response_format: { type: 'text' }
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.response_format).toEqual({ type: 'text' });
  });
});

// ---------- Groq ----------
describe('Groq response_format', () => {
  let provider: GroqProvider;

  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
    provider = new GroqProvider({ apiKey: 'test-key' });
  });

  it('should pass response_format through to the API body', async () => {
    mockFetch.mockResolvedValueOnce(openAIChatCompletion('llama-3.3-70b-versatile'));

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'Give me JSON' }],
      model: 'llama-3.3-70b-versatile',
      response_format: { type: 'json_object' }
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.response_format).toEqual({ type: 'json_object' });
  });

  it('should not include response_format when not set', async () => {
    mockFetch.mockResolvedValueOnce(openAIChatCompletion('llama-3.3-70b-versatile'));

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'Hello' }],
      model: 'llama-3.3-70b-versatile'
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.response_format).toBeUndefined();
  });
});

// ---------- Anthropic ----------
describe('Anthropic response_format', () => {
  let provider: AnthropicProvider;

  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
    provider = new AnthropicProvider({ apiKey: 'test-key' });
  });

  it('should append JSON instruction to system prompt and add prefilled assistant message', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        id: 'msg-test',
        type: 'message',
        role: 'assistant',
        content: [{ type: 'text', text: '"result": "ok"}' }],
        model: 'claude-3-5-haiku-20241022',
        stop_reason: 'end_turn',
        usage: { input_tokens: 20, output_tokens: 5 }
      }),
      headers: new Headers({ 'content-type': 'application/json' })
    });

    const response = await provider.generateResponse({
      messages: [{ role: 'user', content: 'Give me JSON' }],
      model: 'claude-3-5-haiku-20241022',
      systemPrompt: 'You are a helpful assistant.',
      response_format: { type: 'json_object' }
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);

    // System prompt should have JSON instruction appended
    expect(body.system).toContain('You are a helpful assistant.');
    expect(body.system).toContain('You must respond with valid JSON only.');

    // Messages should end with a prefilled assistant message
    const lastMessage = body.messages[body.messages.length - 1];
    expect(lastMessage.role).toBe('assistant');
    expect(lastMessage.content).toBe('{');

    // Response content should have '{' prepended
    expect(response.message).toBe('{"result": "ok"}');
    expect(response.content).toBe('{"result": "ok"}');
  });

  it('should inject system prompt even when none provided', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        id: 'msg-test',
        type: 'message',
        role: 'assistant',
        content: [{ type: 'text', text: '"data":1}' }],
        model: 'claude-3-5-haiku-20241022',
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 3 }
      }),
      headers: new Headers({ 'content-type': 'application/json' })
    });

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'Give me JSON' }],
      model: 'claude-3-5-haiku-20241022',
      response_format: { type: 'json_object' }
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.system).toContain('You must respond with valid JSON only.');
  });

  it('should not modify request when response_format is not set', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        id: 'msg-test',
        type: 'message',
        role: 'assistant',
        content: [{ type: 'text', text: 'Hello!' }],
        model: 'claude-3-5-haiku-20241022',
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 3 }
      }),
      headers: new Headers({ 'content-type': 'application/json' })
    });

    const response = await provider.generateResponse({
      messages: [{ role: 'user', content: 'Hello' }],
      model: 'claude-3-5-haiku-20241022',
      systemPrompt: 'Be helpful.'
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);

    // System prompt should be unchanged
    expect(body.system).toBe('Be helpful.');

    // No prefilled assistant message
    expect(body.messages).toHaveLength(1);
    expect(body.messages[0].role).toBe('user');

    // Response should not have '{' prepended
    expect(response.message).toBe('Hello!');
  });

  it('should not mutate the original request messages array', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        id: 'msg-test',
        type: 'message',
        role: 'assistant',
        content: [{ type: 'text', text: '"ok":true}' }],
        model: 'claude-3-5-haiku-20241022',
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 3 }
      }),
      headers: new Headers({ 'content-type': 'application/json' })
    });

    const originalMessages = [{ role: 'user' as const, content: 'test' }];
    const request: LLMRequest = {
      messages: originalMessages,
      model: 'claude-3-5-haiku-20241022',
      response_format: { type: 'json_object' }
    };

    await provider.generateResponse(request);

    // The original messages array should not be modified
    expect(originalMessages).toHaveLength(1);
    expect(originalMessages[0].role).toBe('user');
  });
});

// ---------- Cloudflare ----------
describe('Cloudflare response_format', () => {
  let provider: CloudflareProvider;
  let mockAiRun: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
    mockAiRun = vi.fn();
    provider = new CloudflareProvider({
      ai: { run: mockAiRun } as unknown as Ai
    });
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should inject JSON instruction into system prompt', async () => {
    mockAiRun.mockResolvedValueOnce({ response: '{"ok":true}' });

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'Give me JSON' }],
      model: '@cf/meta/llama-3.1-8b-instruct',
      systemPrompt: 'Be helpful.',
      response_format: { type: 'json_object' }
    });

    const [, body] = mockAiRun.mock.calls[0];
    const systemMsg = body.messages.find((m: any) => m.role === 'system');
    expect(systemMsg).toBeDefined();
    expect(systemMsg.content).toContain('Be helpful.');
    expect(systemMsg.content).toContain('You must respond with valid JSON only.');
  });

  it('should create system message when none exists', async () => {
    mockAiRun.mockResolvedValueOnce({ response: '{"ok":true}' });

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'Give me JSON' }],
      model: '@cf/meta/llama-3.1-8b-instruct',
      response_format: { type: 'json_object' }
    });

    const [, body] = mockAiRun.mock.calls[0];
    const systemMsg = body.messages.find((m: any) => m.role === 'system');
    expect(systemMsg).toBeDefined();
    expect(systemMsg.content).toContain('You must respond with valid JSON only.');
  });

  it('should not modify messages when response_format is not set', async () => {
    mockAiRun.mockResolvedValueOnce({ response: 'Hello!' });

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'Hello' }],
      model: '@cf/meta/llama-3.1-8b-instruct'
    });

    const [, body] = mockAiRun.mock.calls[0];
    const systemMsg = body.messages.find((m: any) => m.role === 'system');
    expect(systemMsg).toBeUndefined();
  });
});

// ---------- Cerebras ----------
describe('Cerebras response_format', () => {
  let provider: CerebrasProvider;

  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
    provider = new CerebrasProvider({ apiKey: 'test-key' });
  });

  it('should inject JSON instruction into system prompt', async () => {
    mockFetch.mockResolvedValueOnce(openAIChatCompletion('llama-3.1-8b'));

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'Give me JSON' }],
      model: 'llama-3.1-8b',
      systemPrompt: 'Be helpful.',
      response_format: { type: 'json_object' }
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    const systemMsg = body.messages.find((m: any) => m.role === 'system');
    expect(systemMsg).toBeDefined();
    expect(systemMsg.content).toContain('Be helpful.');
    expect(systemMsg.content).toContain('You must respond with valid JSON only.');
  });

  it('should create system message when none exists', async () => {
    mockFetch.mockResolvedValueOnce(openAIChatCompletion('llama-3.1-8b'));

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'Give me JSON' }],
      model: 'llama-3.1-8b',
      response_format: { type: 'json_object' }
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    const systemMsg = body.messages.find((m: any) => m.role === 'system');
    expect(systemMsg).toBeDefined();
    expect(systemMsg.content).toContain('You must respond with valid JSON only.');
  });

  it('should not modify messages when response_format is not set', async () => {
    mockFetch.mockResolvedValueOnce(openAIChatCompletion('llama-3.1-8b'));

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'Hello' }],
      model: 'llama-3.1-8b'
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    const systemMsg = body.messages.find((m: any) => m.role === 'system');
    expect(systemMsg).toBeUndefined();
  });
});
