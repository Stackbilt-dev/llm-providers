# Changelog

All notable changes to `@stackbilt/llm-providers` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/). Versions use [Semantic Versioning](https://semver.org/).

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
