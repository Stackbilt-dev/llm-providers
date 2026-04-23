# @stackbilt/llm-providers

A multi-provider LLM abstraction layer with automatic failover, graduated circuit breakers, cost tracking, and intelligent retry. Built for Cloudflare Workers but runs anywhere with a standard `fetch` API. Extracted from a production orchestration platform handling 80K+ LOC across multiple services.

## Features

- **Multi-provider failover** -- OpenAI, Anthropic, Cloudflare Workers AI, Cerebras, and Groq behind a single interface
- **Graduated circuit breaker** -- 4-state machine (closed / degraded / recovering / open) with probabilistic traffic routing prevents cascading failures
- **Exponential backoff retry** -- configurable delays, jitter, and per-error-class behavior
- **Cost tracking and optimization** -- per-provider cost attribution, budget alerts with CreditLedger, automatic routing to cheaper providers
- **Declarative model catalog** -- semantic model metadata drives recommendations, provider defaults, and fallback routing
- **Rate limit enforcement** -- CreditLedger tracks RPM/RPD/TPM/TPD per provider; factory skips providers that exceed limits
- **Streaming** -- SSE streaming support for all providers
- **Tool/function calling** -- OpenAI, Anthropic, Cerebras, and Cloudflare tool use with unified response format
- **Image generation** -- Cloudflare Workers AI (SDXL, FLUX) and Google Gemini
- **Health monitoring** -- per-provider health checks, metrics, and circuit breaker state
- **Structured logging** -- injectable `Logger` interface; silent by default, opt-in to console or custom loggers
- **Zero runtime dependencies** -- no transitive dependency tree to audit

## Installation

```bash
npm install @stackbilt/llm-providers
```

## Quick Start

```typescript
import { LLMProviders, MODELS } from '@stackbilt/llm-providers';

const llm = new LLMProviders({
  openai: { apiKey: process.env.OPENAI_API_KEY },
  anthropic: { apiKey: process.env.ANTHROPIC_API_KEY },
  cloudflare: { ai: env.AI }, // Cloudflare Workers AI binding
  defaultProvider: 'auto',
  costOptimization: true,
  enableCircuitBreaker: true,
});

const response = await llm.generateResponse({
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Summarize the circuit breaker pattern.' },
  ],
  maxTokens: 1000,
  temperature: 0.7,
});

console.log(response.message);
console.log(`Provider: ${response.provider}, Cost: $${response.usage.cost}`);
```

### Auto-Discovery from Environment

```typescript
import { LLMProviders } from '@stackbilt/llm-providers';

// Scans env for ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY,
// CEREBRAS_API_KEY, and AI binding — configures only what's present
const llm = LLMProviders.fromEnv(env, {
  costOptimization: true,
  enableCircuitBreaker: true,
});
```

## Providers

| Provider | Models | Streaming | Tools | Notes |
|----------|--------|-----------|-------|-------|
| **OpenAI** | GPT-4o Mini, GPT-4 Turbo, GPT-4, GPT-3.5 Turbo | Yes | Yes | Default: `gpt-4o-mini` |
| **Anthropic** | Claude Opus 4.6, Sonnet 4.6, Sonnet 4, Haiku 4.5, 3.7 Sonnet, 3.5 Sonnet/Haiku, 3 Opus/Sonnet | Yes | Yes | Default: `claude-haiku-4-5-20251001` |
| **Cloudflare** | Gemma 4 26B, Llama 4 Scout, GPT-OSS 120B, LLaMA 3.x, Mistral 7B, Qwen 1.5, TinyLlama, and more | Yes | GPT-OSS, Gemma 4, Llama 4 Scout | Default is request-aware and catalog-driven |
| **Cerebras** | LLaMA 3.1 8B, LLaMA 3.3 70B, ZAI-GLM 4.7, Qwen 3 235B | Yes | GLM/Qwen only | ~2,200 tok/s |
| **Groq** | LLaMA 3.3 70B Versatile, LLaMA 3.1 8B Instant, GPT-OSS 120B | Yes | LLaMA 3.3 70B, GPT-OSS 120B | Ultra-fast inference |

### Provider Configuration

```typescript
// OpenAI
{ apiKey: 'sk-...', organization: 'org-...', project: 'proj-...' }

// Anthropic
{ apiKey: 'sk-ant-...', version: '2023-06-01' }

// Cloudflare Workers AI
{ ai: env.AI, accountId: '...' }

// Cerebras
{ apiKey: 'csk-...' }

// Groq
{ apiKey: 'gsk_...' }
```

## Logging

The library is silent by default. Opt in to logging by passing a `Logger`:

```typescript
import { LLMProviders, consoleLogger } from '@stackbilt/llm-providers';

const llm = new LLMProviders({
  anthropic: { apiKey: '...', logger: consoleLogger },
  logger: consoleLogger, // factory-level logging
});
```

Or implement your own `Logger` interface (`debug`, `info`, `warn`, `error`).

## Circuit Breaker

Each provider gets a graduated circuit breaker that routes traffic away from failing providers with probabilistic degradation.

| State | Behavior |
|-------|----------|
| **Closed** | 100% traffic to primary. Failures increment counter. |
| **Degraded** | Traffic splits probabilistically (90% → 70% → 40% → 10%) as failures accumulate. |
| **Recovering** | Success steps traffic back up one level at a time. |
| **Open** | 0% traffic. After `resetTimeout` ms, failures decay and traffic resumes. |

Default: 5-step degradation curve `[1.0, 0.9, 0.7, 0.4, 0.1]`, 60s reset timeout, 5-minute monitoring window.

```typescript
import { CircuitBreakerManager } from '@stackbilt/llm-providers';

const manager = new CircuitBreakerManager({
  failureThreshold: 5,
  resetTimeout: 60000,
  monitoringPeriod: 300000,
  degradationCurve: [1.0, 0.9, 0.7, 0.4, 0.1],
});

const breaker = manager.getBreaker('openai');
console.log(breaker.getHealth());
```

## Cost Tracking & Budget Management

```typescript
import { CreditLedger, LLMProviders } from '@stackbilt/llm-providers';

const ledger = new CreditLedger({
  budgets: [
    { provider: 'openai', monthlyBudget: 50, rateLimits: { rpm: 60, rpd: 10000 } },
    { provider: 'anthropic', monthlyBudget: 100 },
  ],
});

// Threshold alerts fire at 80%, 90%, 95% utilization
ledger.on((event) => {
  if (event.type === 'threshold_crossed') {
    console.warn(`${event.provider}: ${event.tier} — ${event.utilizationPct.toFixed(0)}% of budget`);
  }
});

const llm = new LLMProviders({
  openai: { apiKey: '...' },
  anthropic: { apiKey: '...' },
  costOptimization: true,
  ledger, // Factory enforces rate limits and tracks spend
});
```

## Model Catalog & Runtime Selection

Model selection is driven by a declarative catalog rather than a hardcoded fallback array. The selector intersects:

- requested use case and capabilities
- configured providers
- circuit breaker state (`CLOSED`, `DEGRADED`, `RECOVERING`, `OPEN`)
- CreditLedger utilization and projected burn/depletion pressure

The catalog also distinguishes active, compatibility, and retired models. Retired IDs can remain exported for compatibility, but they are not recommendation targets.

```typescript
import {
  MODEL_CATALOG,
  MODEL_RECOMMENDATIONS,
  getRecommendedModel,
  inferUseCaseFromRequest
} from '@stackbilt/llm-providers';

const useCase = inferUseCaseFromRequest({
  messages: [{ role: 'user', content: 'Call the weather tool' }],
  tools: [{
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get weather',
      parameters: { type: 'object' }
    }
  }]
});

const model = getRecommendedModel('TOOL_CALLING', ['cloudflare', 'openai']);
```

For runtime-aware recommendations from a configured instance:

```typescript
const recommended = llm.getRecommendedModel({
  messages: [{ role: 'user', content: 'Summarize this incident' }],
  maxTokens: 800
});
```

## Fallback Rules

Customize when and how the factory falls back between providers:

```typescript
const llm = new LLMProviders({
  openai: { apiKey: '...' },
  anthropic: { apiKey: '...' },
  cloudflare: { ai: env.AI },
  cerebras: { apiKey: '...' },
  fallbackRules: [
    { condition: 'rate_limit', fallbackProvider: 'cloudflare' },
    { condition: 'cost', threshold: 10, fallbackProvider: 'cloudflare' },
    { condition: 'error', fallbackProvider: 'anthropic' },
  ],
});
```

Default provider precedence remains Cloudflare → Cerebras → Groq → Anthropic → OpenAI, but actual dispatch is catalog-driven and can be reordered at runtime by request fit, circuit-breaker state, and ledger burn-rate pressure.

## Error Handling

Structured error classes for each failure mode:

```typescript
import {
  RateLimitError,
  QuotaExceededError,
  AuthenticationError,
  CircuitBreakerOpenError,
  TimeoutError,
} from '@stackbilt/llm-providers';

try {
  await llm.generateResponse(request);
} catch (error) {
  if (error instanceof RateLimitError) {
    // Automatic retry already attempted; consider switching providers
  } else if (error instanceof CircuitBreakerOpenError) {
    // Provider is temporarily disabled
  } else if (error instanceof AuthenticationError) {
    // Check API key -- will NOT trigger fallback
  }
}
```

## Model Constants

```typescript
import { MODELS, getRecommendedModel } from '@stackbilt/llm-providers';

// Current-gen models
MODELS.CLAUDE_OPUS_4_6;         // 'claude-opus-4-6-20250618'
MODELS.CLAUDE_SONNET_4_6;       // 'claude-sonnet-4-6-20250618'
MODELS.CLAUDE_HAIKU_4_5;        // 'claude-haiku-4-5-20251001'
MODELS.GPT_4O;                  // 'gpt-4o' (deprecated / compatibility only)
MODELS.GPT_4O_MINI;             // 'gpt-4o-mini'
MODELS.CEREBRAS_ZAI_GLM_4_7;    // 'zai-glm-4.7'

// Get best active model for a use case given available providers
const model = getRecommendedModel('COST_EFFECTIVE', ['openai', 'cloudflare']);
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `LLMProviders` | High-level facade -- initialize providers, generate responses, check health |
| `LLMProviderFactory` | Lower-level factory with provider chain building, catalog-based routing, and fallback logic |
| `OpenAIProvider` | OpenAI GPT models (streaming, tools) |
| `AnthropicProvider` | Anthropic Claude models (streaming, tools) |
| `CloudflareProvider` | Cloudflare Workers AI (streaming, tools on GPT-OSS/Gemma 4/Llama 4, batch) |
| `CerebrasProvider` | Cerebras fast inference (streaming, tools on GLM/Qwen) |
| `GroqProvider` | Groq fast inference (streaming, tools on GPT-OSS/LLaMA 3.3 70B) |
| `BaseProvider` | Abstract base with shared resiliency, metrics, and cost calculation |

### Utilities

| Class | Description |
|-------|-------------|
| `CircuitBreaker` | Graduated 4-state circuit breaker with probabilistic degradation |
| `CircuitBreakerManager` | Manages circuit breakers across multiple providers |
| `RetryManager` | Exponential backoff retry with jitter |
| `CostTracker` | Per-provider cost accumulation and budget alerts |
| `CreditLedger` | Monthly budgets, rate limits, burn rate projection, threshold events |
| `CostOptimizer` | Static methods for optimal provider selection |
| `MODEL_CATALOG` | Declarative model metadata for routing and recommendation |
| `ImageProvider` | Multi-provider image generation (Cloudflare SDXL/FLUX, Google Gemini) |

### Logger

| Export | Description |
|--------|-------------|
| `Logger` | Interface: `debug`, `info`, `warn`, `error` methods |
| `noopLogger` | Silent logger (default) |
| `consoleLogger` | Forwards to `console.*` (opt-in) |

### Key Types

| Type | Description |
|------|-------------|
| `LLMRequest` | Unified request: messages, model, temperature, tools, response_format |
| `LLMResponse` | Unified response: message, usage (with cost), provider, tool calls |
| `TokenUsage` | Token counts and cost (inputTokens, outputTokens, totalTokens, cost) |
| `ProviderFactoryConfig` | Factory config: provider configs, fallback rules, ledger, logger |
| `CostAnalytics` | Cost breakdown, total, and recommendations |
| `ProviderHealthEntry` | Health status, metrics, circuit breaker state, capabilities |
| `ModelCatalogEntry` | Declarative model metadata: provider, lifecycle, capabilities, use cases |

### Factory Functions

| Function | Description |
|----------|-------------|
| `createLLMProviders(config)` | Create an `LLMProviders` instance |
| `createCostOptimizedLLMProviders(config)` | Create with cost optimization, circuit breakers, and retries enabled |
| `LLMProviders.fromEnv(env)` | Auto-discover providers from environment variables |
| `llm.getRecommendedModel(request, useCase?)` | Runtime recommendation using configured providers, health, and ledger state |
| `getRecommendedModel(useCase, providers, context?)` | Pick the best active model for a use case |
| `retry(fn, config)` | One-shot retry wrapper for any async function |

## License

Apache-2.0

---

Built by [Stackbilt](https://stackbilt.dev).
