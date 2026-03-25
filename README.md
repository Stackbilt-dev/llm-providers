# @stackbilt/llm-providers

A multi-provider LLM abstraction layer with automatic failover, circuit breakers, cost tracking, and intelligent retry. Built for Cloudflare Workers but runs anywhere with a standard `fetch` API. Extracted from a production orchestration platform handling 80K+ LOC across multiple services.

## Features

- **Multi-provider failover** -- OpenAI, Anthropic, and Cloudflare Workers AI behind a single interface
- **Circuit breaker** -- state machine (closed / open / half-open) prevents cascading failures
- **Exponential backoff retry** -- configurable delays, jitter, and per-error-class behavior
- **Cost tracking and optimization** -- per-provider cost attribution, budget alerts, automatic routing to cheaper providers
- **Streaming** -- SSE streaming support for all three providers
- **Tool/function calling** -- OpenAI and Anthropic tool use with unified response format
- **Batch processing** -- concurrent request batching with rate-limit awareness
- **Health monitoring** -- per-provider health checks, metrics, and circuit breaker state

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

## Provider Configuration

### OpenAI

```typescript
{
  apiKey: 'sk-...',
  organization: 'org-...',   // optional
  project: 'proj-...',       // optional
  baseUrl: 'https://api.openai.com/v1', // optional, for proxies
  timeout: 30000,
  maxRetries: 3,
}
```

### Anthropic

```typescript
{
  apiKey: 'sk-ant-...',
  version: '2023-06-01',     // optional
  baseUrl: 'https://api.anthropic.com', // optional
  timeout: 30000,
  maxRetries: 3,
}
```

### Cloudflare Workers AI

```typescript
{
  ai: env.AI,                // Cloudflare AI binding (required)
  accountId: '...',          // optional
  timeout: 30000,
  maxRetries: 3,
}
```

## Circuit Breaker

Each provider gets its own circuit breaker that tracks consecutive failures.

| State | Behavior |
|-------|----------|
| **Closed** | Requests pass through normally. Failures increment a counter. |
| **Open** | All requests are immediately rejected. After `resetTimeout` ms, transitions to half-open. |
| **Half-open** | A single test request is allowed through. Success closes the circuit; failure re-opens it. |

Default thresholds: 5 failures to open, 60s reset timeout, 5-minute monitoring window.

```typescript
import { CircuitBreakerManager } from '@stackbilt/llm-providers';

const manager = new CircuitBreakerManager({
  failureThreshold: 5,
  resetTimeout: 60000,
  monitoringPeriod: 300000,
});

const breaker = manager.getBreaker('openai');
console.log(breaker.getHealth());
```

## Cost Optimization

When `costOptimization: true`, the factory routes requests to the cheapest available provider. Cloudflare Workers AI is essentially free and gets top priority.

```typescript
import { createCostOptimizedLLMProviders } from '@stackbilt/llm-providers';

const llm = createCostOptimizedLLMProviders({
  openai: { apiKey: process.env.OPENAI_API_KEY },
  cloudflare: { ai: env.AI },
});

const analytics = llm.getCostAnalytics();
// { breakdown: { openai: { cost, requests, tokens }, ... }, total: 0.042, recommendations: [...] }
```

## Retry with Backoff

Transient errors (rate limits, network errors, server errors) are retried automatically with exponential backoff and jitter.

```typescript
import { RetryManager, retry } from '@stackbilt/llm-providers';

// Standalone retry for any async operation
const result = await retry(
  () => fetch('https://api.example.com/data'),
  { maxRetries: 3, initialDelay: 1000, backoffMultiplier: 2 }
);

// Or configure per-provider via RetryManager
const retryManager = new RetryManager({
  maxRetries: 5,
  initialDelay: 500,
  maxDelay: 30000,
  backoffMultiplier: 2,
});
```

## Fallback Rules

Customize when and how the factory falls back between providers:

```typescript
const llm = new LLMProviders({
  openai: { apiKey: '...' },
  anthropic: { apiKey: '...' },
  cloudflare: { ai: env.AI },
  fallbackRules: [
    { condition: 'rate_limit', fallbackProvider: 'cloudflare' },
    { condition: 'cost', threshold: 10, fallbackProvider: 'cloudflare' },
    { condition: 'error', fallbackProvider: 'anthropic' },
  ],
});
```

## Error Handling

Structured error classes for each failure mode:

```typescript
import {
  RateLimitError,
  QuotaExceededError,
  AuthenticationError,
  CircuitBreakerOpenError,
  TimeoutError,
  LLMErrorFactory,
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

Predefined model identifiers for convenience:

```typescript
import { MODELS, getRecommendedModel } from '@stackbilt/llm-providers';

MODELS.GPT_4O;              // 'gpt-4o'
MODELS.CLAUDE_3_5_SONNET;   // 'claude-3-5-sonnet-20241022'
MODELS.LLAMA_3_1_8B;        // '@cf/meta/llama-3.1-8b-instruct'

// Get best model for a use case given available providers
const model = getRecommendedModel('COST_EFFECTIVE', ['openai', 'cloudflare']);
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `LLMProviders` | High-level facade -- initialize providers, generate responses, check health |
| `LLMProviderFactory` | Lower-level factory with provider chain building and fallback logic |
| `OpenAIProvider` | OpenAI GPT models (streaming, tools, batch) |
| `AnthropicProvider` | Anthropic Claude models (streaming, tools) |
| `CloudflareProvider` | Cloudflare Workers AI (streaming, batch, cost optimization) |
| `BaseProvider` | Abstract base with shared resiliency, metrics, and cost calculation |

### Utilities

| Class | Description |
|-------|-------------|
| `CircuitBreaker` | Per-provider circuit breaker state machine |
| `CircuitBreakerManager` | Manages circuit breakers across multiple providers |
| `RetryManager` | Exponential backoff retry with jitter |
| `CostTracker` | Per-provider cost accumulation and budget alerts |
| `CostOptimizer` | Static methods for optimal provider selection |

### Key Types

| Type | Description |
|------|-------------|
| `LLMRequest` | Unified request: messages, model, temperature, tools, metadata |
| `LLMResponse` | Unified response: message, usage, provider, cost, tool calls |
| `LLMProvider` | Provider interface: generateResponse, healthCheck, estimateCost |
| `ProviderFactoryConfig` | Factory configuration: provider configs, fallback rules, flags |
| `CircuitBreakerConfig` | Failure threshold, reset timeout, monitoring period |
| `RetryConfig` | Max retries, delays, backoff multiplier, retryable error codes |
| `CostConfig` | Token costs, monthly budget, alert threshold |

### Factory Functions

| Function | Description |
|----------|-------------|
| `createLLMProviders(config)` | Create an `LLMProviders` instance |
| `createCostOptimizedLLMProviders(config)` | Create with cost optimization, circuit breakers, and retries enabled |
| `createLLMProviderFactory(config)` | Create a bare `LLMProviderFactory` |
| `createCostOptimizedFactory(config)` | Create a cost-optimized factory |
| `getRecommendedModel(useCase, providers)` | Pick the best model for a use case |
| `retry(fn, config)` | One-shot retry wrapper for any async function |

## License

Apache-2.0

---

Built by [Stackbilt](https://stackbilt.dev).
