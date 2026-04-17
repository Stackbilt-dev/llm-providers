/**
 * Model drift tests
 *
 * Guards against retired/unknown model IDs slipping through by verifying
 * that every entry in a provider's `models` array has a matching entry in
 * its capabilities map. A mismatch usually means a model was retired from
 * one list but not the other — catch that at CI time, not at runtime.
 */

import { describe, it, expect } from 'vitest';
import { AnthropicProvider } from '../providers/anthropic';
import { OpenAIProvider } from '../providers/openai';
import { CloudflareProvider } from '../providers/cloudflare';
import { CerebrasProvider } from '../providers/cerebras';
import { GroqProvider } from '../providers/groq';

type WithCapabilities = {
  models: string[];
  getModelCapabilities: () => Record<string, unknown>;
};

const providers: Array<[string, WithCapabilities]> = [
  ['anthropic', new AnthropicProvider({ apiKey: 'test' }) as unknown as WithCapabilities],
  ['openai', new OpenAIProvider({ apiKey: 'test' }) as unknown as WithCapabilities],
  ['cloudflare', new CloudflareProvider({ ai: { run: async () => ({}) } as never, accountId: 'test' }) as unknown as WithCapabilities],
  ['cerebras', new CerebrasProvider({ apiKey: 'test' }) as unknown as WithCapabilities],
  ['groq', new GroqProvider({ apiKey: 'test' }) as unknown as WithCapabilities]
];

describe('model drift', () => {
  it.each(providers)('%s: every advertised model has a capabilities entry', (_name, provider) => {
    const caps = provider.getModelCapabilities();
    const missing = provider.models.filter((m) => !(m in caps));
    expect(missing).toEqual([]);
  });

  it.each(providers)('%s: every capabilities entry is advertised in models[]', (_name, provider) => {
    const caps = provider.getModelCapabilities();
    const advertised = new Set(provider.models);
    const orphaned = Object.keys(caps).filter((m) => !advertised.has(m));
    expect(orphaned).toEqual([]);
  });
});
