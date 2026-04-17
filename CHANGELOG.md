# Changelog

All notable changes to `@stackbilt/llm-providers` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/). Versions use [Semantic Versioning](https://semver.org/).

## [1.4.0] — 2026-04-17

### Deprecated
- **`MODELS.CLAUDE_3_HAIKU`** (`claude-3-haiku-20240307`) — Anthropic retires 2026-04-19. Migrate to `MODELS.CLAUDE_HAIKU_4_5` or `MODELS.CLAUDE_3_5_HAIKU`. Export retained; callers get a compile-time `@deprecated` warning.
- **`MODELS.GPT_4O`** (`gpt-4o`) — retired by OpenAI on 2026-04-03. Migrate to `MODELS.GPT_4O_MINI` or a current GPT-4 successor. Export retained; callers get a compile-time `@deprecated` warning.

### Removed
- `claude-3-haiku-20240307` — dropped from `AnthropicProvider.models[]` and its capabilities/pricing table. Calls to this ID will fail at Anthropic's cutoff; keeping it advertised would mislead consumers.
- `gpt-4o` — dropped from `OpenAIProvider.models[]` and its capabilities/pricing table.
- `gpt-4-turbo-preview` — dead alias dropped from `OpenAIProvider.models[]` (no corresponding capabilities entry; caught by the new drift test).

### Added
- **Model drift test** (`src/__tests__/model-drift.test.ts`) — asserts every provider's `models[]` array is symmetrically covered by its capabilities map. Prevents future retirement drift where a model is removed from one list but not the other. Runs across all 5 providers.

## [1.3.0] — 2026-04-16

### Added
- **Cloudflare Workers AI vision support** — `CloudflareProvider` now accepts `request.images` and routes to vision-capable models. Previously image data was silently dropped on the CF path.
- **Three new CF vision models**:
  - `@cf/google/gemma-4-26b-a4b-it` — 256K context, vision + function calling + reasoning
  - `@cf/meta/llama-4-scout-17b-16e-instruct` — natively multimodal, tool calling
  - `@cf/meta/llama-3.2-11b-vision-instruct` — image understanding
- **`CloudflareProvider.supportsVision = true`** — factory's `analyzeImage` now dispatches to CF when configured.
- **Factory default vision fallback** — `getDefaultVisionModel()` falls back to `@cf/google/gemma-4-26b-a4b-it` when neither Anthropic nor OpenAI is configured, enabling CF-only deployments to use `analyzeImage()`.

### Changed
- Images are passed to CF using the OpenAI-compatible `image_url` content-part shape (base64 data URIs). HTTP image URLs throw a helpful `ConfigurationError` — fetch the image and pass bytes in `image.data`.
- Attempting `request.images` on a non-vision CF model throws a `ConfigurationError` naming the vision-capable alternatives.

## [1.2.0] — 2026-04-01

### Added
- **Structured Logger** — `Logger` interface with `noopLogger` (silent default) and `consoleLogger` (opt-in). All components accept an optional `logger` via config. Zero `console.*` calls in production code.
- **Rate limit enforcement** — `LLMProviderFactory` now checks `CreditLedger.checkRateLimit(rpm/rpd)` before dispatching to each provider, skipping exceeded providers.
- **Claude 4.6 models** — `claude-opus-4-6-20250618`, `claude-sonnet-4-6-20250618` added to Anthropic provider.
- **Claude Haiku 4.5** — `claude-haiku-4-5-20251001` added.
- **Claude 3.7 Sonnet** — `claude-3-7-sonnet-20250219` added (replaces incorrect `claude-sonnet-3.7` ID).
- **`CostAnalytics`** and **`ProviderHealthEntry`** — typed return values for `getCostAnalytics()` and `getProviderHealth()`.

### Fixed
- **30+ `any` types eliminated** — all provider interfaces, tool call types, Workers AI response shapes, error bodies, cost analytics returns, and decorator signatures fully typed. Three boundary casts for Cloudflare `Ai.run()` retained with explicit eslint-disable.
- **Data leak removed** — `console.log` at `anthropic.ts:492` that dumped full tool call payloads into worker logs.
- **Anthropic JSON mode** — only prepends `{` if the response doesn't already start with one, preventing `{{...}` corruption.
- **OpenAI `supportsBatching`** — set to `false` (was `true` but `processBatch()` is a sequential loop).
- **Default model** — OpenAI default changed from deprecated `gpt-3.5-turbo` to `gpt-4o-mini`.
- **Default fallback chain** — now includes all 5 configured providers (was hardcoded to cloudflare/anthropic/openai, excluding Cerebras and Groq).
- **Anthropic healthCheck** — switched from real API call (burned tokens) to lightweight OPTIONS reachability check.
- **`TokenUsage.cost`** — made required (was optional, causing NaN accumulation in cost tracker).
- **Circuit breaker test isolation** — `defaultCircuitBreakerManager.resetAll()` in all test `beforeEach` blocks to prevent cross-test state leaks.

### Changed
- **Logging default** — library is now silent by default (`noopLogger`). Pass `consoleLogger` or a custom `Logger` to enable output.
- **Model catalog** — updated to current-gen models; removed stale/incorrect model IDs and TBD pricing.

## [1.1.0] — 2026-04-01

### Added
- **ImageProvider** — multi-provider image generation (Cloudflare Workers AI + Google Gemini). Extracted from img-forge production codebase.
- **5 built-in image models**: `sdxl-lightning` (fast/draft), `flux-klein` (balanced), `flux-dev` (high quality), `gemini-flash-image` (text-capable), `gemini-flash-image-preview` (latest).
- **`IMAGE_MODELS`** registry with full config: dimensions, steps, guidance, negative prompt support, seed support.
- **`normalizeAiResponse()`** — handles all Workers AI return types (ArrayBuffer, ReadableStream, objects with `.image`, base64 strings).
- **`getImageModel()`** — lookup helper for model configs.
- Custom model configs via `ImageProviderConfig.models` — extend or override the built-in registry.

## [1.0.0] — 2026-04-01

First stable release. Production-tested in AEGIS cognitive kernel since v1.72.0.

### Added
- **`LLMProviders.fromEnv()`** — auto-discovers available providers from environment variables. One-line setup for multi-provider configurations.
- **`response_format`** — unified structured output support (`{ type: 'json_object' }`) across all providers that support it.
- **CreditLedger** — per-provider monthly budget tracking with threshold alerts (80%/90%/95%), burn rate calculation, and depletion projection.
- **Burn rate analytics** — `burnRate()` returns current spend velocity and projected depletion date per provider.
- **Cerebras provider** — ZAI-GLM 4.7 (355B reasoning) and Qwen 3 235B (MoE) via OpenAI-compatible API with tool calling support.
- **Groq provider** — fast inference via OpenAI-compatible API.
- **Cloudflare provider** — Workers AI integration with GPT-OSS 120B tool calling support.
- **OpenAI provider** — GPT-4o and compatible models.
- **Anthropic provider** — Claude models via Messages API.
- **Graduated circuit breaker** — half-open probe state, configurable failure thresholds, automatic recovery.
- **CostTracker** — per-provider cost aggregation with `breakdown()`, `total()`, and `drain()` for periodic reporting.
- **RetryManager** — exponential backoff with jitter, configurable `shouldRetry` callback, max attempts.
- **Rich error model** — 12 typed error classes (RateLimitError, QuotaExceededError, AuthenticationError, etc.) with `retryable` flag.
- **Model constants** — `MODELS` object with all supported model identifiers.
- **Model recommendations** — `getRecommendedModel()` for use-case-based model selection (cost-effective, high-performance, balanced, tool-calling, long-context).
- **npm provenance** — all published versions include cryptographic provenance attestation linking to the exact GitHub commit.
- **CI workflows** — typecheck + test suite on Node 18/20/22 for every PR.
- **SECURITY.md** — vulnerability reporting policy and supply chain security documentation.

### Security
- **Zero runtime dependencies.** Published tarball contains only compiled code and license.
- **CI-only publishing** with OIDC-based npm authentication and provenance signing.
- Automated `npm audit` on every CI run.
