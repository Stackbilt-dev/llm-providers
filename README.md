# @stackbilt/llm-providers

A multi-provider LLM abstraction layer with automatic failover, graduated circuit breakers, cost tracking, and intelligent retry. Built for Cloudflare Workers but runs anywhere with a standard `fetch` API. Extracted from a production orchestration platform handling 80K+ LOC across multiple services.

## Features

- **Multi-provider failover** -- OpenAI, Anthropic, Cloudflare Workers AI, Cerebras, Groq, and NVIDIA NIM behind a single interface
- **Graduated circuit breaker** -- 4-state machine (closed / degraded / recovering / open) with probabilistic traffic routing prevents cascading failures
- **Exponential backoff retry** -- configurable delays, jitter, and per-error-class behavior
- **Cost tracking and optimization** -- per-provider cost attribution, budget alerts with CreditLedger, automatic routing to cheaper providers
- **Declarative model catalog** -- semantic model metadata drives recommendations, provider defaults, and fallback routing
- **Rate limit enforcement** -- CreditLedger tracks RPM/RPD/TPM/TPD per provider; factory skips providers that exceed limits
- **Streaming with fallback** -- SSE streaming on all providers; factory-level streaming routes through the same circuit-breaker and fallback chain as non-streaming requests
- **Tool/function calling** -- OpenAI, Anthropic, Cerebras, and Cloudflare tool use with unified response format
- **Tool-use loop helper** -- `generateResponseWithTools` owns the request → parse → execute → repeat cycle with iteration caps, cost limits, abort signal support, and `onIteration` early-exit via `{ abort: true }`
- **Server-side built-in tools** -- `LLMRequest.builtInTools` (e.g. `[{ type: 'web_search' }]`) drives Groq's Compound systems and GPT-OSS 120B to run web search / code interpreter server-side; capability-gated per model and routed to the capable provider automatically
- **Routing introspection** -- `getRoutingInfo()` returns a pre-flight routing snapshot (use case, provider, model, token estimate, lifecycle, deprecation warning) without dispatching the request; pairs with `request.metadata.useCase` to pre-classify intent at the gateway and let the catalog drive dispatch
- **Deprecation annotations** -- `generateResponse()` attaches `metadata.llmProvidersDeprecationWarning` to every response that uses a non-active-lifecycle model
- **Provider-agnostic cache hints** -- `LLMRequest.cache` translates to provider-native caching (Anthropic `cache_control` breakpoints; automatic on OpenAI/Groq/Cerebras); cached token counts normalized into `TokenUsage`
- **Schema drift detection** -- envelope validation on every provider response; streaming frames validated per-chunk; `SchemaDriftError` routes through fallback chain and fires `onSchemaDrift` hook
- **Schema canary** -- `runCanaryCheck` / `extractShape` / `compareShapes` for comparing live response shapes against committed golden fixtures
- **Image generation** -- `ImageProvider` is deprecated; use [img-forge](https://github.com/Stackbilt-dev/img-forge) instead
- **Health monitoring** -- per-provider health checks, metrics, and circuit breaker state
- **Structured logging** -- injectable `Logger` interface; silent by default, opt-in to console or custom loggers
- **Zero runtime dependencies** -- no transitive dependency tree to audit

## Deprecated APIs

**`ImageProvider` is deprecated.** It was extracted from img-forge during an earlier phase when llm-providers and img-forge were more tightly coupled. img-forge is now the dedicated imagegen service with a full quality-tier registry, async orchestration, and quota metering. Use the [img-forge API](https://github.com/Stackbilt-dev/img-forge) or MCP tools for image generation. llm-providers handles text inference and vision understanding only.

`ImageProvider`, `ImageProviderConfig`, and `IMAGE_MODELS` will be removed in the next major version. The `normalizeAiResponse` utility is under audit and may be retained as a standalone export.

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
// CEREBRAS_API_KEY, NVIDIA_API_KEY, and AI binding — configures only what's present
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
| **Cerebras** | GPT-OSS 120B, ZAI-GLM 4.7, LLaMA 3.1 8B *(deprecated 2026-05-27)*, Qwen 3 235B *(deprecated 2026-05-27)* | Yes | GLM, Qwen, GPT-OSS | ~2,200 tok/s |
| **Groq** | LLaMA 3.3 70B Versatile, LLaMA 3.1 8B Instant, GPT-OSS 120B, Compound, Compound Mini | Yes | LLaMA 3.3 70B, GPT-OSS 120B | Ultra-fast inference; Compound systems run built-in tools |
| **NVIDIA NIM** | Llama 3.3/3.1 70B, Llama 4 Maverick, Nemotron 70B/49B/253B, Mistral Large 2, DeepSeek V4 Flash/Pro | Yes | Llama, Nemotron, Mistral Large 2 | Costs $0 placeholder — dev-tier credits; set real rates in `CreditLedger` |

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

// NVIDIA NIM
{ apiKey: 'nvapi-...' }
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

## Routing Introspection

`getRoutingInfo()` lets gateways and agent orchestrators inspect what the catalog engine *would* do — without dispatching the request. Use this at ingress to classify intent once, log the routing decision, enforce policies, or pre-warm the right provider.

```typescript
import { getRoutingInfo, LLMProviders } from '@stackbilt/llm-providers';

// Standalone — pass the providers you have configured
const info = getRoutingInfo(
  { messages: [{ role: 'user', content: 'Call the weather tool' }], tools: [...] },
  ['cloudflare', 'cerebras', 'groq']
);

// info.useCase              → 'TOOL_CALLING'
// info.provider             → 'cerebras'
// info.model                → 'zai-glm-4.7'
// info.estimatedInputTokens → heuristic count
// info.modelLifecycle       → 'active'
// info.deprecationWarning   → undefined (or a warning string for non-active models)
```

### metadata.useCase passthrough

After calling `getRoutingInfo()`, set `request.metadata.useCase` to the classified value before dispatching. The factory's `resolveUseCase()` reads this field directly, skipping re-inference and ensuring the gateway's classification is honored:

```typescript
const info = getRoutingInfo(request, configuredProviders);

// Attach the gateway's classification so the factory doesn't re-infer it
const annotatedRequest = {
  ...request,
  metadata: { ...request.metadata, useCase: info.useCase },
};

const response = await llm.generateResponse(annotatedRequest);
```

### Deprecation warnings on responses

When `generateResponse()` routes to a model on `compatibility` or `retired` lifecycle, the response includes a warning:

```typescript
const response = await llm.generateResponse({ model: 'llama-3.1-8b', ... });

if (response.metadata?.llmProvidersDeprecationWarning) {
  console.warn(response.metadata.llmProvidersDeprecationWarning);
  // → "llama-3.1-8b deprecates 2026-05-27 — plan migration"
}
```

This fires automatically — no opt-in required. The warning surface is the response, so any layer that handles responses (the gateway, the caller) sees it regardless of which layer made the call.

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

Default provider precedence remains Cloudflare → Cerebras → Groq → NVIDIA → Anthropic → OpenAI, but actual dispatch is catalog-driven and can be reordered at runtime by request fit, circuit-breaker state, and ledger burn-rate pressure.

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
MODELS.GROQ_COMPOUND;           // 'groq/compound' (RESEARCH; built-in tools)
MODELS.GROQ_COMPOUND_MINI;      // 'groq/compound-mini'
MODELS.NVIDIA_NEMOTRON_70B;     // 'nvidia/llama-3.1-nemotron-70b-instruct'
MODELS.NVIDIA_LLAMA_4_MAVERICK; // 'meta/llama-4-maverick-17b-128e-instruct'

// Get best active model for a use case given available providers
const model = getRecommendedModel('COST_EFFECTIVE', ['openai', 'cloudflare']);
```

## Factory-Level Streaming

`generateResponseStream` uses the same provider-selection, circuit-breaker, and exhaustion-registry path as `generateResponse`. Pre-stream HTTP errors (401, 429, 503, circuit open) fall over to the next provider before emitting the first chunk.

```typescript
const stream = await llm.generateResponseStream({
  messages: [{ role: 'user', content: 'Tell me a story.' }],
  model: 'claude-haiku-4-5-20251001',
});

const reader = stream.getReader();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  process.stdout.write(value); // string chunk
}
```

## Tool-Use Loop

`generateResponseWithTools` owns the `generateResponse → parse → execute → append → repeat` cycle. It enforces iteration caps, cumulative cost limits, and abort-signal cancellation — no boilerplate needed on the caller side.

```typescript
import { LLMProviders, ToolLoopLimitError } from '@stackbilt/llm-providers';

const result = await llm.generateResponseWithTools(
  {
    messages: [{ role: 'user', content: 'What is 2 + 2 * 3?' }],
    tools: [{
      type: 'function',
      function: {
        name: 'calculate',
        description: 'Evaluate a math expression',
        parameters: { type: 'object', properties: { expr: { type: 'string' } }, required: ['expr'] }
      }
    }],
  },
  {
    execute: async (name, args) => {
      if (name === 'calculate') return eval((args as { expr: string }).expr);
      throw new Error(`Unknown tool: ${name}`);
    }
  },
  {
    maxIterations: 5,
    maxCostUSD: 0.10,
    onIteration: (iteration, state) => {
      // Return { abort: true, reason? } to stop the loop early and throw ToolLoopAbortedError.
      // Returning void (or nothing) continues to the next iteration.
      if (state.messages.some(m => m.role === 'assistant' && m.content?.includes('DONE'))) {
        return { abort: true, reason: 'found answer' };
      }
    },
  }
);

console.log(result.message); // final assistant response after tool execution
```

## Built-in Tools (Server-Side)

Some models run tools **server-side** — the provider executes web search, code interpretation, etc. inside a single completion call, no execute callback required. Set `LLMRequest.builtInTools` with normalized identifiers; the adapter translates to each model family's native wire shape and rejects unsupported combinations at the boundary.

```typescript
import { LLMProviders, MODELS } from '@stackbilt/llm-providers';

const llm = LLMProviders.fromEnv(env);
const res = await llm.generateResponse({
  model: MODELS.GROQ_COMPOUND,                 // 'groq/compound'
  messages: [{ role: 'user', content: 'Find authoritative sources on topic X.' }],
  builtInTools: [{ type: 'web_search' }],
});
```

Normalized identifiers: `web_search`, `visit_website`, `browser_automation`, `code_interpreter`, `wolfram_alpha`.

| Model | Built-in tools | Wire shape |
|-------|----------------|------------|
| `groq/compound`, `groq/compound-mini` | all five | `compound_custom.tools.enabled_tools` (identifiers verbatim) |
| `openai/gpt-oss-120b` (Groq) | `web_search`, `code_interpreter` | OpenAI-style `tools: [{ type }]`, `web_search` → `browser_search` |
| all other models | — | rejected with `ConfigurationError` naming the capable models |

Notes:
- **Capability-aware routing.** `openai/gpt-oss-120b` is hosted by both Cerebras and Groq; only Groq runs built-in tools, so a `builtInTools` request is steered to Groq automatically. Plain requests keep the default routing.
- **Provenance.** The Compound systems are tagged `RESEARCH`-only in the catalog and are not auto-selected for generic use cases — pin the model (or request the `RESEARCH` use case) to use them, since selecting a Compound model can incur per-search surcharges.
- **Cost.** Built-in tool surcharges (e.g. web search ~$5/1k requests) are billed by the provider and are **not** attributed per-call in `TokenUsage`; track them via `CreditLedger` if needed.
- **Citations.** Structured search results surface on `LLMResponse.metadata.builtInToolResults` — `Array<{ type, name?, arguments?, results: [{ title, url, content, score }] }>`. Only executions that ran a web search appear (e.g. `code_interpreter` runs, which carry no citations, are omitted); the field is absent when no search ran. Citation sub-fields are passed through as the provider returns them — treat them as best-effort and validate URLs before use.
- **Reasoning.** When the model exposes its internal reasoning (the queries it searched), it surfaces on `LLMResponse.metadata.reasoning` as a string. Absent when the model doesn't emit it.

```typescript
const citations = res.metadata?.builtInToolResults?.[0]?.results ?? [];
// → [{ title, url, content, score }, …]
```

## Prompt Cache Hints

Pass a provider-agnostic `cache` hint on any request. The library translates it to the appropriate provider-native mechanism.

```typescript
const response = await llm.generateResponse({
  messages: [{ role: 'user', content: 'Summarize the context.' }],
  systemPrompt: 'You are an expert at analyzing long documents. [... 10KB of stable context ...]',
  model: 'claude-haiku-4-5-20251001',
  cache: {
    strategy: 'provider-prefix',   // mark the stable prefix for caching
    cacheablePrefix: 'auto',       // cache system prompt + tools (default)
  },
});

// Cached token counts are normalized in TokenUsage
console.log(response.usage.cacheReadInputTokens);    // Anthropic cache hit tokens
console.log(response.usage.cachedInputTokens);       // OpenAI / Groq / Cerebras cache hit tokens
```

| Strategy | Behavior |
|----------|----------|
| `'off'` | No caching hints sent |
| `'provider-prefix'` | Mark stable prefix for provider-side caching |
| `'response'` | Enable AI Gateway response caching (via `GatewayMetadata`) |
| `'both'` | Both prefix and response caching |

## Schema Drift Canary

Use the canary utilities to compare a live provider response against a committed golden fixture and detect API shape drift before it reaches production.

```typescript
import {
  extractShape, compareShapes, runCanaryCheck
} from '@stackbilt/llm-providers';

// 1. Load your committed golden fixture (flat path → type map)
import goldenShape from './fixtures/openai.json';

// 2. Fetch a raw response from the provider (your responsibility)
const liveResponse = await fetch('https://api.openai.com/v1/chat/completions', ...).then(r => r.json());

// 3. Check for drift
const report = runCanaryCheck('openai', goldenShape, liveResponse);

if (report.status === 'drift') {
  console.error('OpenAI response shape changed!', report.diff);
  // diff.added   — new fields (additive, usually safe)
  // diff.removed — missing fields (breaking, alert immediately)
  // diff.changed — type-changed fields (breaking, alert immediately)
}
```

Generate your initial golden fixture from a known-good response:

```typescript
import { extractShape } from '@stackbilt/llm-providers';
import fs from 'fs';

const shape = extractShape(knownGoodResponse);
fs.writeFileSync('fixtures/openai.json', JSON.stringify(shape, null, 2));
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
| `NvidiaProvider` | NVIDIA NIM inference (streaming, tools on Llama/Nemotron/Mistral) |
| `BaseProvider` | Abstract base with shared resiliency, metrics, and cost calculation |

### Utilities

| Class / Export | Description |
|----------------|-------------|
| `CircuitBreaker` | Graduated 4-state circuit breaker with probabilistic degradation |
| `CircuitBreakerManager` | Manages circuit breakers across multiple providers |
| `RetryManager` | Exponential backoff retry with jitter |
| `CostTracker` | Per-provider cost accumulation and budget alerts |
| `CreditLedger` | Monthly budgets, rate limits, burn rate projection, threshold events |
| `CostOptimizer` | Static methods for optimal provider selection |
| `MODEL_CATALOG` | Declarative model metadata for routing and recommendation |
| `ImageProvider` | **[Deprecated]** Multi-provider image generation — use [img-forge](https://github.com/Stackbilt-dev/img-forge) instead |
| `extractShape` | Walk a raw API response into a flat `path → type` shape map |
| `compareShapes` | Diff two shape maps into `{ added, removed, changed }` |
| `runCanaryCheck` | One-shot canary: extract live shape, compare against golden, return `CanaryReport` |
| `validateSchema` | Low-level envelope validator (for custom provider authors) |

### Logger

| Export | Description |
|--------|-------------|
| `Logger` | Interface: `debug`, `info`, `warn`, `error` methods |
| `noopLogger` | Silent logger (default) |
| `consoleLogger` | Forwards to `console.*` (opt-in) |

### Key Types

| Type | Description |
|------|-------------|
| `LLMRequest` | Unified request: messages, model, temperature, tools, response_format, cache, lora |
| `LLMResponse` | Unified response: message, usage (with cost), provider, tool calls |
| `TokenUsage` | Token counts, cost, and cached token fields (cachedInputTokens, cacheReadInputTokens, cacheCreationInputTokens) |
| `CacheHints` | Cache strategy, key, ttl, sessionId, cacheablePrefix for provider-agnostic prompt caching |
| `ToolExecutor` | Interface for `generateResponseWithTools`: `execute(name, args) => Promise<unknown>` |
| `ToolLoopOptions` | Loop config: maxIterations, maxCostUSD, onIteration, abortSignal |
| `ToolLoopAbortSignal` | `{ abort: true; reason?: string }` — return from `onIteration` to stop the loop and throw `ToolLoopAbortedError` |
| `RoutingInfo` | Pre-flight routing snapshot: useCase, provider, model, estimatedInputTokens, lifecycle, deprecationWarning |
| `CanaryReport` | Schema canary result: provider, status ('ok'|'drift'), diff |
| `ShapeMap` | Flat `path → JSON-type` map produced by `extractShape` |
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
| `llm.generateResponse(request)` | Generate a response with provider selection and fallback |
| `llm.generateResponseStream(request)` | Streaming generation; fallback chain active before first chunk |
| `llm.generateResponseWithTools(request, executor, opts?)` | Managed tool-use loop with caps and abort-signal support |
| `llm.getRecommendedModel(request, useCase?)` | Runtime recommendation using configured providers, health, and ledger state |
| `getRecommendedModel(useCase, providers, context?)` | Pick the best active model for a use case |
| `getRoutingInfo(request, providers?, context?)` | Pre-flight routing snapshot — use case, model, lifecycle, deprecation warning — without dispatching |
| `runCanaryCheck(provider, golden, liveResponse)` | Compare live response shape against golden fixture |
| `extractShape(obj)` | Extract flat path → type map from any object |
| `retry(fn, config)` | One-shot retry wrapper for any async function |

## Security

`@stackbilt/llm-providers` treats supply chain security as a first-class concern:

- **Zero runtime dependencies.** The published tarball contains only compiled source and the Apache-2.0 license — no transitive dependency tree to audit or compromise.
- **npm provenance attestation.** Every published version is cryptographically linked to the exact GitHub commit and CI workflow that built it. Verify on the npm registry page under "Provenance."
- **CI-only OIDC publishing.** Releases are published exclusively through GitHub Actions with short-lived OIDC tokens. No long-lived npm publish credentials exist.
- **SHA-pinned Actions.** Every workflow step references a full commit SHA, not a mutable tag. A compromised `v4` tag cannot inject code into this workflow.
- **SBOM on every commit.** The supply chain workflow generates a Software Bill of Materials artifact (`sbom-{sha}`) on every push to `main`, retained for 90 days.
- **Dependency review on every PR.** Pull requests are blocked if any newly introduced dependency has a known vulnerability or an incompatible license.
- **`--ignore-scripts` on install.** CI and publish both run `npm ci --ignore-scripts`, preventing install-time script execution from dev dependencies.
- **`npm audit` on every build.** Vulnerabilities in dev dependencies are caught before tests run and before publish.

See [SECURITY.md](./SECURITY.md) for the full security policy and vulnerability reporting.

## License

Apache-2.0

---

Built by [Stackbilt](https://stackbilt.dev).
