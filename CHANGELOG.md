# Changelog

All notable changes to `@stackbilt/llm-providers` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/). Versions use [Semantic Versioning](https://semver.org/).

## [1.14.2] — 2026-06-09

Workers AI catalog expansion for Cloudflare credit-backed gateway routing.

### Added
- **Cloudflare Kimi K2.6** — adds `@cf/moonshotai/kimi-k2.6` as an active Workers AI catalog entry with long context, tool calling, vision, and structured-agent workload metadata.
- **Cloudflare GLM-4.7-Flash** — adds `@cf/zai-org/glm-4.7-flash` as an active fast/balanced Workers AI catalog entry with long context and tool-calling metadata.
- **Cloudflare DeepSeek V4 Pro** — adds the dashboard model slug `deepseek/deepseek-v4-pro` as an active high-performance Workers AI catalog entry for reasoning and coding routes.

## [1.14.1] — 2026-06-07

Patch compatibility fix for Cerebras OpenAI-compatible tool-call responses.

### Fixed
- **Cerebras omitted tool-call content** — `CerebrasProvider` now accepts assistant tool-call responses that omit `message.content`, normalizing text content to an empty string while preserving valid `toolCalls`. This matches the tolerance already used for Groq and Cloudflare OpenAI-compatible adapters.

## [1.14.0] — 2026-06-07

Gateway route-planning surface from issue #87. Additive only.

### Added
- **Worker gateway route plan export** — `getGatewayRoutePlan()` packages canonical normalization, catalog routing, cache hints, capability checks, degradations, and warnings into one Worker-friendly object for use behind OpenAI-compatible, Ollama-style, or Anthropic-compatible API routers.
- **Route plan types** — `GatewayRoutePlan`, `GatewayRouteRequirements`, `GatewayRouteCapabilityReport`, and `GatewayRouteCachePlan` describe the route-plan shape (storage-agnostic; consumers map `plan.cache` onto their own KV/Cache API/D1/R2 implementation).
- **LoRA degradation reporting** — when a request carries `lora` and routing selects a non-Cloudflare provider, the plan reports a `stripped` degradation and warns that Cloudflare adapter ids are forwarded to Workers AI without validation.
- **Route plan tests** — `src/__tests__/gateway-routing.test.ts` covers canonical→plan mapping, cache-hint handling, LoRA-on/off-Cloudflare paths, and built-in tool capability mismatches.

## [1.13.1] — 2026-06-06

Patch fix for Groq tool-call-only responses from issue #86.

### Fixed
- **Groq omitted tool-call content** — `GroqProvider` now accepts assistant responses that omit `message.content` when `message.tool_calls` is present, normalizing the assistant text to empty string while preserving `finishReason: "tool_calls"` and populated `toolCalls`.

## [1.13.0] — 2026-06-06

Cloudflare Workers AI cache binding support from issue #84. Additive only.

### Added
- **Cloudflare Workers AI cache run options** — `CloudflareProvider` now translates `CacheHints.sessionId` into Workers AI binding `extraHeaders['x-session-affinity']` for `provider-prefix` / `both` cache strategies, including streaming and raw vision calls.
- **Cloudflare AI Gateway binding config** — `CloudflareConfig.gateway` exposes typed Workers AI binding Gateway options and merges request cache metadata (`GatewayMetadata.cacheKey`, `cacheTtl`, and custom metadata) into the third `env.AI.run()` argument.
- **Workers AI cached token usage** — Cloudflare usage parsing now normalizes cached input token counts into `TokenUsage.cachedInputTokens` when Workers AI returns them.

## [1.12.0] — 2026-06-05

Canonical provider contract hardening from issue #81 / PR #82. Additive only.

### Added
- **Canonical request/response contract exports** — new `CanonicalLLMRequest`, `CanonicalLLMResponse`, and related canonical types provide a provider-neutral boundary for gateways and coding-agent orchestrators.
- **Canonical normalizers** — `normalizeLLMRequest()` maps compatibility `LLMRequest` fields such as `response_format`, `toolChoice`, `images`, `lora`, `topP`, `frequencyPenalty`, `reasoning`, and `prediction` into the canonical shape; `canonicalToLLMRequest()` converts canonical requests back into existing adapter input while providers migrate internally.
- **Canonical response routing metadata helper** — `normalizeLLMResponse()` returns a stable response shape with selected provider/model, fallback chain, capability degradation, normalized error, and provider-extra metadata slots.
- **Canonical contract tests** — one canonical fixture now exercises OpenAI-compatible, Anthropic-compatible, Groq/Cerebras, NVIDIA, and Cloudflare adapter preparation paths without live API calls.
- **Gateway boundary documentation** — README now documents `client protocol -> gateway adapter -> CanonicalLLMRequest -> llm-providers -> vendor API`.

## [1.11.0] — 2026-05-31

Reliability and gateway-routing hardening. Additive APIs plus bug fixes from issues #61, #62, #63, #64, #65, and #67.

### Added
- **Generated `VERSION` export** — `VERSION` now comes from `src/version.ts`, generated by `scripts/sync-version.mjs` from `package.json#version`. Build, test, watch, coverage, prepare, and prepublish paths run the sync step, and package smoke verifies installed CJS and ESM consumers see the published version.
- **Streaming usage reconciliation** — streaming responses now attach final token usage metadata and the factory records it to cost tracking, quota hooks, and request-end hooks after the stream drains. OpenAI, Groq, Cerebras, and NVIDIA request `stream_options.include_usage`; Anthropic parses `message_start` / `message_delta`; Cloudflare uses Workers AI usage when present and falls back to non-zero estimates from streamed content.
- **Reliability state persistence** — `CircuitBreaker`, `CircuitBreakerManager`, and `ExhaustionRegistry` now expose snapshot/serialize/restore/deserialize APIs for KV, D1, Redis, Durable Objects, or other external stores. README shows a Cloudflare Workers KV pattern for circuit breaker, exhaustion, and `CreditLedger` state.
- **Workload-aware model defaults** — `ModelWorkloadClass`, `ModelPreferenceMap`, `normalizeModelWorkload()`, `getRecommendedModelForWorkload()`, and `getProviderDefaultModelForWorkload()` let gateways request provider-tuned defaults for workload classes such as `summary`, `planning`, `code_draft`, `long_context`, and `tool_loop`.
- **Provider-agnostic cache write metric** — `TokenUsage.cacheWriteInputTokens` aliases Anthropic `cache_creation_input_tokens` so gateways can report cache writes without provider-specific parsing.

### Fixed
- **Vision requests on non-vision providers** — Cerebras, Groq, and NVIDIA now explicitly advertise `supportsVision = false`; the base provider rejects image inputs when `supportsVision !== true`; the factory skips vision-incapable providers and throws when no configured provider supports images.
- **Anthropic URL images** — Anthropic URL image inputs now throw `ConfigurationError` instead of being converted to lossy placeholder text. Callers should send base64 image bytes or route URL-based vision to an OpenAI-compatible provider.

## [1.10.0] — 2026-05-29

Groq built-in tools (issue #69). Additive only.

### Added
- **`LLMRequest.builtInTools`** — optional `Array<{ type: BuiltInToolType }>` requesting server-side tools (web search, code interpreter, etc.). Normalized identifiers: `web_search`, `visit_website`, `browser_automation`, `code_interpreter`, `wolfram_alpha`. Ignored by providers/models that don't advertise `supportsBuiltInTools`.
- **`BuiltInTool`, `BuiltInToolType`, `BuiltInToolResult` types** — exported from package root. `BuiltInToolResult` mirrors Groq's `executed_tools[]` (`{ type, name?, arguments?, results: [{ title, url, content, score }] }`); `type` is provider-native and open-ended.
- **`ModelCapabilities.supportsBuiltInTools`** — per-model list of built-in tools a (model, provider) pair advertises; drives capability-aware routing and boundary gating.
- **`modelSupportsBuiltInTools(model, provider, tool?)`** — catalog accessor; the single source of truth for built-in-tool gating.
- **`groq/compound` + `groq/compound-mini` catalog entries** — Groq Compound systems (all five built-in tools), tagged `RESEARCH`-only so they stay out of generic recommendation pools (selecting a Compound model can incur per-search surcharges). `MODELS.GROQ_COMPOUND` / `MODELS.GROQ_COMPOUND_MINI` exported.
- **`RESEARCH` `ModelRecommendationUseCase`** — new use case with `scoreUseCase` weights and a `MODEL_RECOMMENDATIONS.RESEARCH` list; honored by `factory.resolveUseCase()` via `metadata.useCase`. Not inferred from request shape (opt-in only).
- **Capability-aware built-in-tools routing** — `openai/gpt-oss-120b` is hosted by both Cerebras and Groq; a `builtInTools` request is steered to Groq (the capable host) while plain requests keep the prior default. Resolves the catalog collision via `getProvidersForCatalogModel`.
- **Groq built-in-tools request fork + boundary gating** — Compound systems send tools on `compound_custom.tools.enabled_tools` (identifiers verbatim); `openai/gpt-oss-120b` sends OpenAI-style `tools: [{ type }]` with `web_search` → `browser_search` translation, merged alongside function tools. Unsupported `(model, tool)` pairs throw `ConfigurationError` naming the capable models.
- **Groq built-in tool result parsing** — `message.executed_tools[]` is parsed into `LLMResponse.metadata.builtInToolResults` (`Array<{ type, name?, arguments?, results: [{ title, url, content, score }] }>`). Only executions carrying a non-empty `search_results.results` surface; non-search runs (e.g. `code_interpreter`) are omitted, and the field is absent when no search ran. The model's internal reasoning surfaces on `metadata.reasoning` when present. `GROQ_RESPONSE_SCHEMA` extended with an optional, shallow `executed_tools` entry (validates `type` only — citation sub-fields are intentionally unguarded to avoid false `SchemaDriftError` fallback on a single sampled shape).

### Notes
- Built-in tool surcharges are billed by the provider and are **not** attributed per-call in `TokenUsage`; use `CreditLedger` for accounting.
- Streaming with `builtInTools` runs the search server-side but emits content deltas only — structured `metadata.builtInToolResults` / `metadata.reasoning` are available on non-streaming `generateResponse` only.

## [1.9.0] — 2026-05-22

### Added
- **`getRoutingInfo(request, providers?, context?)`** — pre-flight routing snapshot for gateways and agent orchestrators. Call once at ingress to get `{ useCase, provider, model, catalogEntry, estimatedInputTokens, requiresTools, requiresVision, requestsStreaming, modelLifecycle, deprecationWarning }` without dispatching the request. Export from package root.
- **`RoutingInfo` type** — interface for the above snapshot.
- **`metadata.useCase` passthrough** — set `request.metadata.useCase` to a `ModelRecommendationUseCase` value (e.g. `'TOOL_CALLING'`) and the factory's `resolveUseCase()` will honor it directly, bypassing inference. Pair with `getRoutingInfo()` to classify once at the gateway layer and let the catalog engine drive dispatch from there.
- **`ToolLoopAbortSignal` type** — `{ abort: true; reason?: string }`. Return this from the `onIteration` callback in `generateResponseWithTools` to stop the tool loop immediately and throw `ToolLoopAbortedError`. Void-returning callbacks are unaffected.
- **Response deprecation annotation** — `generateResponse()` attaches `metadata.llmProvidersDeprecationWarning` to any response whose model is on `compatibility` or `retired` lifecycle, or whose catalog description contains a deprecation date. Cerebras models deprecating 2026-05-27 will surface warnings on every response starting now.

### Fixed
- **`VERSION` constant** — was hardcoded `'0.1.0'` since v1.0.0; corrected to track the actual package version (`'1.9.0'`).

## [1.8.0] — 2026-05-22

### Added
- **NVIDIA NIM provider** — `NvidiaProvider` adds nine NVIDIA-hosted models behind the same `LLMProvider` interface: `meta/llama-3.3-70b-instruct`, `meta/llama-4-maverick-17b-128e-instruct` (1M context), `nvidia/llama-3.1-nemotron-70b-instruct`, `nvidia/llama-3.3-nemotron-super-49b-v1`, `nvidia/llama-3.1-nemotron-ultra-253b-v1`, `meta/llama-3.1-70b-instruct`, `mistralai/mistral-large-2-instruct`, `deepseek-ai/deepseek-v4-flash`, and `deepseek-ai/deepseek-v4-pro`.
- **`NvidiaConfig` type** — provider config interface exported from package root; identical shape to `GroqConfig` / `CerebrasConfig` (just `apiKey` + base `ProviderConfig` fields).
- **NVIDIA in `ProviderFactoryConfig`** — `nvidia?: NvidiaConfig` added; re-init on `updateConfig({ nvidia })` wired.
- **NVIDIA in `LLMProviders.fromEnv()`** — detects `NVIDIA_API_KEY` (`nvapi-…` prefix) automatically; included in the missing-key error message.
- **NVIDIA catalog entries** — nine entries in `MODEL_CATALOG` with lifecycle, use-case tags, and capability metadata. All `inputTokenCost`/`outputTokenCost` values are `0` — NIM dev-tier is credit-based and production pricing varies by deployment; use `CreditLedger` for budget accounting.
- **NVIDIA in `PROVIDER_FALLBACK_ORDER`** — inserted after Groq, before Anthropic.
- **`MODELS.NVIDIA_*` constants** — nine constants covering all catalog models.
- **Tool calling on NVIDIA NIM** — verified live: Meta Llama 3.1/3.3, Llama 4 Maverick, all Nemotron instruct models, and Mistral Large 2 confirmed tool-capable. DeepSeek V4 Flash/Pro marked `supportsTools: false` pending verification (returned 502 during probe; add to `TOOL_CAPABLE_MODELS` once confirmed).
- **NVIDIA schema drift detection** — `NVIDIA_RESPONSE_SCHEMA` validated on every response. Handles NVIDIA-specific response differences: `tool_calls` always present as `[]` when unused (not absent); `prompt_tokens_details` may be `null`.
- **NVIDIA golden fixture** — `src/__tests__/fixtures/response-shapes/nvidia.json` committed.
- **NVIDIA test suite** — 21 tests covering constructor, generateResponse, streaming, tool call extraction, empty `tool_calls: []` handling, null `prompt_tokens_details` handling, health check, and balance reporting.

## [1.7.0] — 2026-05-22

### Added
- **Cerebras reasoning forwarding** — `LLMRequest.reasoning.effort`, `reasoning.format`, and `reasoning.clearThinking` are translated to `reasoning_effort`, `reasoning_format`, and `clear_thinking` on Cerebras requests. `clearThinking: false` preserves GLM-4.7 reasoning state across turns, improving prompt cache hit rates in agentic loops.
- **Cerebras predicted outputs** — `LLMRequest.prediction` forwards to the Cerebras `prediction` field (`gpt-oss-120b`, `zai-glm-4.7`). Combining `prediction` with `tools` throws `ConfigurationError` at the boundary before any network call.
- **Cerebras `json_schema` structured outputs** — `response_format: { type: 'json_schema', json_schema: { name, schema, strict? } }` is forwarded natively to the Cerebras API. The existing `json_object` system-prompt-injection path is unchanged.
- **Cerebras prompt cache key forwarding** — `LLMRequest.cache.key` is forwarded as `prompt_cache_key` to Cerebras (128-token block caching, 5m–1h TTL).
- **`LLMRequest.reasoning` field** — provider-agnostic reasoning controls: `effort` (`low` | `medium` | `high` | `none`), `format` (`parsed` | `raw` | `hidden`), `clearThinking` (boolean). Currently translated only by Cerebras; other providers ignore the field.
- **`LLMRequest.prediction` field** — provider-agnostic predicted output hint for speculative decoding. Currently forwarded only by Cerebras.
- **`response_format` json_schema union** — `LLMRequest.response_format` now accepts `{ type: 'json_schema'; json_schema: { name, schema, strict? } }` in addition to the existing `json_object` / `text` variants.
- **`supportsPromptCache` on Cerebras catalog entries** — flagged on `gpt-oss-120b`, `zai-glm-4.7`, and `qwen-3-235b-a22b-instruct-2507`.

### Deprecated
- **Cerebras `llama-3.1-8b`** — moved to `compatibility` lifecycle. Cerebras end-of-life: 2026-05-27. Migrate to `openai/gpt-oss-120b` (best perf/cost) or `zai-glm-4.7` (reasoning/tools).
- **Cerebras `qwen-3-235b-a22b-instruct-2507`** — moved to `compatibility` lifecycle. Cerebras end-of-life: 2026-05-27. Migrate to `openai/gpt-oss-120b` or `zai-glm-4.7`.

### Retired
- **Cerebras `llama-3.3-70b`** — moved to `retired` lifecycle; model is no longer listed in Cerebras API. The string remains in `CerebrasProvider.models` for backward compatibility (additive-only policy) but the model catalog will not route new traffic to it.

### Fixed
- **Cerebras `zai-glm-4.7` context length** — corrected from `131000` to `131072` in both `getModelCapabilities()` and the model catalog.

## [1.6.5] — 2026-05-15

### Fixed
- **Published package ESM import resolution** — runtime source imports now use explicit `.js` relative specifiers so Node ESM consumers can resolve package internals after install. This fixes `ERR_MODULE_NOT_FOUND` failures when importing `@stackbilt/llm-providers` from installed tarballs.

### Added
- **Tarball consumer smoke test in CI and publish gates** — new `npm run test:package` packs the module, installs it into a clean temp project, and verifies both `require('@stackbilt/llm-providers')` and `import` entrypoints. This prevents regressions where tests pass in-repo but the published artifact is not installable.

## [1.6.2] — 2026-05-07

### Deprecated
- **`ImageProvider`, `ImageProviderConfig`, `IMAGE_MODELS`** — marked `@deprecated`. img-forge is the org's canonical image generation service; `ImageProvider` is a frozen snapshot of img-forge internals and will not track new models or providers. Use the img-forge API or MCP tools for image generation. Exports are retained for backward compatibility and will be removed in the next major version. `normalizeAiResponse` is not yet deprecated (under audit).

## [1.6.1] — 2026-05-06

### Added
- **Cerebras `openai/gpt-oss-120b`** — added to `CerebrasProvider.models`, `TOOL_CAPABLE_MODELS`, and the model catalog with `HIGH_PERFORMANCE | TOOL_CALLING | BALANCED` tiers, 128k context, and tool support

## [1.6.0] — 2026-04-27

### Added
- **SSE streaming schema validation (#41)** — all four providers (`AnthropicProvider`, `OpenAIProvider`, `GroqProvider`, `CerebrasProvider`) now surface malformed SSE frames as `SchemaDriftError` and fire `onSchemaDrift` instead of swallowing silently. Anthropic additionally validates `content_block_delta` event shape and `delta.text` type; future tool-streaming delta types are skipped via forward-compat discriminator. 26 new streaming tests via `describe.each`.
- **Schema drift canary (#39 Part 2)** — `src/utils/schema-canary.ts` ships three exported utilities:
  - `extractShape(obj)` — walks a raw response and returns a flat `path → JSON-type` map
  - `compareShapes(golden, live)` — diffs two shape maps into `{ added, removed, changed }`
  - `runCanaryCheck(provider, golden, liveResponse)` — one-shot canary returning a `CanaryReport`
  - Golden fixtures committed under `src/__tests__/fixtures/response-shapes/` for all five providers
  - All exports available at package root
- **Cache-aware routing (#52)** — provider-agnostic cache hints on `LLMRequest`:
  - New `CacheHints` type with `strategy`, `key`, `ttl`, `sessionId`, `cacheablePrefix` fields
  - `LLMRequest.cache?: CacheHints` — no-op for callers that don't set it
  - Anthropic: `strategy: 'provider-prefix' | 'both'` wraps the system prompt as a content-block array with `cache_control: { type: 'ephemeral' }` and marks the last tool definition as a breakpoint when `cacheablePrefix` is `'auto'` or `'tools'`
  - OpenAI, Groq, Cerebras: automatic caching with no request-side translation needed
  - Cloudflare: platform-level prefix caching via Workers AI binding
- **Cached token usage reporting (#52)** — all providers now extract provider-specific cached token counts into normalized `TokenUsage` fields:
  - `cachedInputTokens` — Groq, Cerebras, OpenAI automatic cache hits (`prompt_tokens_details.cached_tokens`)
  - `cacheReadInputTokens` — Anthropic `cache_read_input_tokens`
  - `cacheCreationInputTokens` — Anthropic `cache_creation_input_tokens`
  - `supportsPromptCache` flag added to `ModelCapabilities` and populated for all Anthropic models
- **Cloudflare LoRA / fine-tune forwarding (#51)** — `LLMRequest.lora?: string` is forwarded to the Workers AI binding, enabling adapter-based fine-tunes without wrapper code.
- **Factory-level streaming with fallback (#26)** — `LLMProviderFactory.generateResponseStream()` and `LLMProviders.generateResponseStream()` use the same provider-selection, circuit-breaker, and exhaustion-registry path as `generateResponse()`. Pre-stream HTTP errors fall over to the next provider before emitting the first chunk.
- **Tool-use loop helper (#28)** — `generateResponseWithTools(request, executor, opts?)` owns the `generateResponse → parse → execute → append → repeat` loop. Ships with `ToolLoopLimitError` (max iterations / cost cap) and `ToolLoopAbortedError` (AbortSignal). `ToolLoopOptions` supports `maxIterations`, `maxCostUSD`, `onIteration`, `abortSignal`.
- **Cloudflare AI Gateway metadata forwarding (#29)** — `GatewayMetadata` on `LLMRequest` (via `BaseProvider.getAIGatewayHeaders`) forwards `cf-aig-cache-key`, `cf-aig-cache-ttl`, and `cf-aig-metadata` headers only when `baseUrl` matches the Cloudflare AI Gateway pattern. Non-Gateway base URLs are unaffected.

### Fixed
- **`stop_sequence` schema drift false positive** — the Anthropic response schema declared `stop_sequence` as `type: 'string'` but the real API always returns `null` when no stop sequence triggers, causing `SchemaDriftError` on every normal response. Changed to `type: 'string-or-null'`. The `AnthropicResponse` internal interface and `formatResponse` null guard updated to match.
- **`AnthropicProvider.getProviderBalance()` invalid endpoint** — was calling `GET /v1/organizations/cost_report`, which is not a valid Anthropic API endpoint, returning HTTP errors in production. Now returns `status: 'unavailable', source: 'not_supported'` with a message directing users to the Admin API (like `GroqProvider` already did). (#25)
- **Inline `import('../types').TokenUsage` annotations** — cleaned up in `groq.ts`, `cerebras.ts`, `anthropic.ts`, and `openai.ts`; `TokenUsage` now lives in the existing `import type` block in each file.

## [1.5.1] — 2026-04-27

### Fixed
- **`analyzeImage()` silent empty response on Cloudflare** — `@cf/meta/llama-3.2-11b-vision-instruct` via the Workers AI binding requires a raw `{ image: number[], prompt, max_tokens }` input shape, not the OpenAI-compatible `messages/image_url` format. The chat path returns `choices[0].message.content === null` via the binding, causing `extractText()` to silently return `""`. The provider now detects this model and dispatches to the raw binding format, mapping the result's `{ response: string }` back through the existing normalisation path. Other vision models (`@cf/google/gemma-4-26b-a4b-it`, `@cf/meta/llama-4-scout-17b-16e-instruct`) continue using the chat format unchanged. Fixes #53.

## [1.5.0] — 2026-04-23

Bundles the unreleased 1.4.0 scope (model retirements, drift test) with envelope validation, env auto-discovery, and the declarative catalog into a single minor release. 1.4.0 was tagged in `package.json` but never published to npm; consumers upgrading from 1.3.0 receive all of the following.

### Added
- **Declarative model catalog** — new `src/model-catalog.ts` introduces a semantic catalog for provider/model metadata, recommendation use cases, lifecycle status, and runtime scoring.
- **Catalog tests** — coverage for retired-model exclusion, provider-health-aware ranking, and request-shape use-case inference.
- **Runtime recommendation API** — `LLMProviders#getRecommendedModel(request, useCase?)` exposes the same routing logic the factory uses internally.
- **Schema drift envelope validation** — `OpenAIProvider`, `GroqProvider`, and `CerebrasProvider` now validate `/chat/completions` response envelopes at the provider boundary, throwing `SchemaDriftError` on mismatch to route through the factory fallback chain and fire `onSchemaDrift` instead of corrupting downstream consumers silently. Anthropic envelope validation added in the same scope. Per-provider schema constants (not shared) — correlated drift across providers is a signal worth detecting.
- **`LLMProviders.fromEnv()` static factory** — auto-discovers providers from Cloudflare Workers `env` bindings (`AI`, `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) without manual wiring.
- **Model drift test** (`src/__tests__/model-drift.test.ts`) — asserts every provider's `models[]` array is symmetrically covered by its capabilities map. Prevents future retirement drift where a model is removed from one list but not the other. Runs across all 5 providers.

### Changed
- **Factory routing** — `LLMProviderFactory` now selects provider/model pairs from the model catalog instead of relying only on hardcoded provider ordering.
- **Health-aware dispatch** — recommendation and auto-routing now consider circuit-breaker state, including degraded and recovering providers, not just fully open breakers.
- **Budget-aware dispatch** — when a `CreditLedger` is attached, selection can demote providers under high utilization or near projected depletion.
- **Provider defaults** — OpenAI, Anthropic, Cloudflare, Cerebras, and Groq now resolve default models through the shared catalog instead of separate hardcoded fallbacks.
- **Cloudflare model recommendation** — `CloudflareProvider.getRecommendedModel()` now prefers modern active baselines such as Gemma 4 and GPT-OSS instead of legacy TinyLlama/Qwen heuristics.
- **Public recommendation exports** — `MODEL_RECOMMENDATIONS` and `getRecommendedModel()` now exclude retired recommendation targets such as `gpt-4o`, while preserving deprecated constants for compatibility.

### Deprecated
- **`MODELS.CLAUDE_3_HAIKU`** (`claude-3-haiku-20240307`) — Anthropic retired 2026-04-19. Migrate to `MODELS.CLAUDE_HAIKU_4_5` or `MODELS.CLAUDE_3_5_HAIKU`. Export retained; callers get a compile-time `@deprecated` warning.
- **`MODELS.GPT_4O`** (`gpt-4o`) — retired by OpenAI on 2026-04-03. Migrate to `MODELS.GPT_4O_MINI` or a current GPT-4 successor. Export retained; callers get a compile-time `@deprecated` warning.

### Removed
- `claude-3-haiku-20240307` — dropped from `AnthropicProvider.models[]` and its capabilities/pricing table. Calls to this ID will fail at Anthropic's cutoff; keeping it advertised would mislead consumers. Arbitrary-string passthrough on request inputs is unchanged.
- `gpt-4o` — dropped from `OpenAIProvider.models[]` and its capabilities/pricing table.
- `gpt-4-turbo-preview` — dead alias dropped from `OpenAIProvider.models[]` (no corresponding capabilities entry; caught by the new drift test).

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
