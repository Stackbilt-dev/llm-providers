# Open Issues Corpus: Stackbilt-dev/llm-providers
Exported: 2026-06-13T11:26:20.466Z  |  Total: 11

## #93: Bug: glm-4.7-flash tagged COST_EFFECTIVE but is a thinking model — outputs reasoning traces instead of direct responses
URL: https://github.com/Stackbilt-dev/llm-providers/issues/93  |  Labels: none  |  Updated: 2026-06-12T19:18:24Z  |  Comments: 0

## Bug

\`@cf/zai-org/glm-4.7-flash\` is currently tagged \`active, [COST_EFFECTIVE, BALANCED, TOOL_CALLING, LONG_CONTEXT]\` in \`model-catalog\`. This causes it to be selected as the **first-choice model for summary and cost-effective routing**.

The model outputs chain-of-thought reasoning traces instead of direct responses, making it unsuitable for any route class where the caller expects a clean answer. Example of what users get when glm-4.7-flash is selected for a summary turn:

\`\`\`
1. **Analyze the user's request:**
 * **Topic:** Local LLM gateway proxy.
 * **Constraint:** "In one sentence."
 * Why do people use it? ...
 * *Draft 1:* A local LLM gateway proxy is software that...
 * *Draft 2 (More technical
\`\`\`

Response truncates mid-reasoning at \`max_tokens\` because the model never gets to the actual answer.

## Confirmed affected models (same pattern)

Two additional models expected to have the same issue if added to the catalog:
- \`@cf/deepseek-ai/deepseek-r1-distill-qwen-32b\` — DeepSeek R1 distill, reasoning model
- \`@cf/qwen/qwq-32b\` — QwQ is explicitly a reasoning model

## Fix

Either:
1. Remove \`COST_EFFECTIVE\` and \`BALANCED\` use cases from \`glm-4.7-flash\` — keep it only under an explicit \`REASONING\` or \`ANALYTICAL\` use case
2. Add a \`thinkingModel: true\` flag to the catalog entry so routers can filter these out for direct-response routing

When \`@cf/deepseek-ai/deepseek-r1-distill-qwen-32b\` and \`@cf/qwen/qwq-32b\` are added (see companion issue), apply the same handling.

## Discovered via

bildy end-to-end smoke test: live summary routing selected \`glm-4.7-flash\` and returned reasoning trace to Claude Code instead of a summary response.

## #92: Add missing CF Workers AI models: llama-3.3-70b-fp8-fast, qwen2.5-coder-32b, gpt-oss-20b, mistral-small-3.1, qwen3-30b, llama-3.2-1b/3b, kimi-k2.7-code
URL: https://github.com/Stackbilt-dev/llm-providers/issues/92  |  Labels: none  |  Updated: 2026-06-12T19:18:12Z  |  Comments: 0

## Summary

Running bildy (https://github.com/Stackbilt-dev/bildy) against the live CF Workers AI model catalog via \`GET /accounts/{id}/ai/models/search?task=Text+Generation\` reveals 7 models missing from \`model-catalog\` that are material to routing quality.

## Missing models (CF API confirms all active)

| Model | Suggested use cases | Notes |
|---|---|---|
| \`@cf/meta/llama-3.3-70b-instruct-fp8-fast\` | \`BALANCED\`, \`HIGH_PERFORMANCE\`, \`TOOL_CALLING\` | Best value model on Workers AI right now. Fast FP8 quant. Primary gap for \`planning\` route class. |
| \`@cf/qwen/qwen2.5-coder-32b-instruct\` | \`COST_EFFECTIVE\` (code) | Purpose-built for code generation. Critical for \`code_draft\` routing. |
| \`@cf/openai/gpt-oss-20b\` | \`COST_EFFECTIVE\`, \`TOOL_CALLING\` | Smaller sibling of \`gpt-oss-120b\` (already active). Good cheap tool-calling option. |
| \`@cf/mistralai/mistral-small-3.1-24b-instruct\` | \`BALANCED\`, \`TOOL_CALLING\` | Strong quality/cost ratio, vision capable. |
| \`@cf/qwen/qwen3-30b-a3b-fp8\` | \`BALANCED\`, \`HIGH_PERFORMANCE\` | Qwen3 generation, state-of-the-art for cost tier. |
| \`@cf/meta/llama-3.2-1b-instruct\` | \`COST_EFFECTIVE\` | Ultra-cheap for dead-simple summary/classify turns. |
| \`@cf/meta/llama-3.2-3b-instruct\` | \`COST_EFFECTIVE\` | Pair with 1b above as step-up model. |
| \`@cf/moonshotai/kimi-k2.7-code\` | \`TOOL_CALLING\`, \`BALANCED\` | Code-focused variant of kimi-k2.6 (already active). |

## Impact on bildy routing

Without \`llama-3.3-70b-fp8-fast\`, bildy's \`planning\` and \`code_draft\` route classes fall back to \`gpt-oss-120b\` which is the right capability tier but not the best cost model. Without \`qwen2.5-coder-32b\`, code generation routing has no purpose-built option.

## Intentionally excluded

- LoRA variants (\`-lora\` suffix) — fine-tune infrastructure, not general routing
- \`@cf/meta/llama-guard-3-8b\` — safety classifier, not chat
- \`@cf/aisingapore/gemma-sea-lion-v4-27b-it\` — SEA-localized, niche

## #91: CF provider: malformed messages for multi-turn tool conversations via /ai/v1/chat/completions
URL: https://github.com/Stackbilt-dev/llm-providers/issues/91  |  Labels: none  |  Updated: 2026-06-10T12:17:03Z  |  Comments: 2

## Summary

When `AI_GATEWAY_ID` is set (enabling the `/ai/v1/chat/completions` path in `cloudflare-ai-binding.ts`), the Cloudflare provider fails on multi-turn tool conversations with:

\`\`\`
AiError: Bad input: Error: oneOf at '/' not met, 0 matches:
required properties at '/' are 'prompt',
Type mismatch of '/messages/0/content', 'array' not in 'string',
Type mismatch of '/messages/1/content', 'string' not in 'null',
required properties at '/messages/1' are 'role,content'
\`\`\`

Single-turn tool requests succeed. The bug only surfaces when `messages` contains prior `function_call` + `function_call_output` history.

## Root cause (localized)

`cloudflare.ts` — `formatRequest()` message builder (around line 588–623):

When the request includes messages with `toolCalls` and `toolResults`, the resulting `messages` array contains:
- A `user` message whose `content` ends up as an array (not a string) in some paths — CF's `/ai/v1/chat/completions` requires string content
- An `assistant` message with `content: null` and `tool_calls` — the endpoint schema may require this to be absent rather than null

The `/ai/run/{model}` path (used without `AI_GATEWAY_ID`) does not exhibit this error — its schema appears more permissive.

## Impact

- Any multi-turn tool-loop request routed through cloudflare provider via `AI_GATEWAY_ID` fails
- Affects `@cf/openai/gpt-oss-120b`, `@cf/moonshotai/kimi-k2.6`, and all other CF tool-capable models
- In llm-gateway: the error is a plain `Error` (not a typed `LLMProviderError`), which additionally blocks fallback (separate gateway-side workaround already applied)

## Suggested fix

In `cloudflare.ts` `formatRequest()`:
1. Ensure all `message.content` values are coerced to `string | null`, never `string[]`, before building the CF payload
2. For assistant messages with `tool_calls`, verify whether the `/ai/v1/chat/completions` endpoint requires `content` to be omitted vs `null`
3. Consider throwing `InvalidRequestError` (or another typed error) instead of propagating the raw `AiError`, so callers can distinguish bad-input from transient failures

## Reproduction

```typescript
// Construct an LLMRequest with prior tool loop history:
const request = {
 messages: [
 { role: 'user', content: 'list files' },
 { role: 'assistant', content: '', toolCalls: [{ id: 'c1', type: 'function', function: { name: 'bash', arguments: '{"command":"ls"}' } }] },
 { role: 'user', content: 'file1\nfile2', toolResults: [{ id: 'c1', output: 'file1\nfile2'

## #90: Add Cloudflare-managed Anthropic Claude Opus 4.8 catalog lane
URL: https://github.com/Stackbilt-dev/llm-providers/issues/90  |  Labels: none  |  Updated: 2026-06-10T10:37:45Z  |  Comments: 2

## Context

Cloudflare now documents a model page for Claude Opus 4.8 under the AI models catalog:

- Docs: https://developers.cloudflare.com/ai/models/anthropic/claude-opus-4.8/
- Model ID: `anthropic/claude-opus-4.8`
- Worker binding example: `await env.AI.run("anthropic/claude-opus-4.8", { max_tokens, messages })`
- REST example: `POST /client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/ai/v1/messages` with a Cloudflare API token
- Documented context window: 1,000,000 tokens
- Example response includes `gatewayMetadata.keySource = "Unified"`, which indicates Cloudflare-managed/unified provider access rather than direct Anthropic API-key usage.

This is relevant to the StackBilt Cloudflare-credit routing policy because it gives us a Cloudflare-native frontier Claude lane alongside Workers AI open models.

## Current repo state

`@stackbilt/llm-providers@1.14.2` includes Cloudflare catalog entries for:

- `@cf/moonshotai/kimi-k2.6`
- `@cf/zai-org/glm-4.7-flash`
- `deepseek/deepseek-v4-pro`

It does not appear to include Cloudflare provider support/catalog metadata for:

- `anthropic/claude-opus-4.8`

The direct Anthropic provider has older direct Anthropic model IDs such as `claude-opus-4-6-20250618`, but this new route is materially different because billing/auth/transport is Cloudflare-managed.

## Desired behavior

Add a Cloudflare provider/catalog entry for the Cloudflare-managed Anthropic route, exposed downstream as a gateway-selectable alias such as:

```text
cloudflare/anthropic/claude-opus-4.8
```

Provider-native model string should remain:

```text
anthropic/claude-opus-4.8
```

## Suggested catalog metadata

Initial conservative metadata, subject to live smoke and pricing review:

- provider: `cloudflare`
- model: `anthropic/claude-opus-4.8`
- lifecycle: `active`
- use cases: `HIGH_PERFORMANCE`, `LONG_CONTEXT`, `TOOL_CALLING`, possibly `BALANCED` only if pricing/routing policy allows
- max context: `1_000_000`
- streaming: likely true, but verify through Cloudflare endpoint/binding
- tool calling: likely true given Anthropic message semantics, but verify before marking Codex/Claude tool-safe
- vision: unknown from the Cloudflare page; do not mark unless documented or verified
- cost: do not hardcode zero; Cloudflare docs link pricing to dashboard, so either leave unknown/placeholder or add explicit Cloudflare-dashboard-derived rates when available
- description: Cloudflare-managed Anthropic Claude Opus 4.8 frontier model with 1M context

## Implementat

## #87: Harden provider contracts for Worker-backed agent routing
URL: https://github.com/Stackbilt-dev/llm-providers/issues/87  |  Labels: none  |  Updated: 2026-06-10T09:36:45Z  |  Comments: 8

## Intent

Harden `@stackbilt/llm-providers` as the reusable provider/capability layer behind a Cloudflare Worker API router for coding-agent clients.

The Worker should be able to delegate model choice, structured-output handling, tool capability checks, failover, cache hints, and provider request/response normalization through public library contracts.

## Use Case

The downstream gateway wants to expose Ollama/OpenAI/Anthropic-compatible endpoints while routing each request to the best available model:

- cheap JSON/structured-output model
- tool-safe model
- code model
- Workers AI LoRA reviewer/critic model
- frontier fallback
- local/remote provider bridge

## Contract Surfaces To Review

- canonical request/response shape
- route preflight/introspection
- cache hint semantics
- provider/model capability metadata for JSON, tools, streaming, multimodal, LoRA, and code tasks
- fallback and degradation explanations
- response normalization for compatibility endpoints

## Acceptance Criteria

- A Worker/router can ask for route selection using a canonical request without provider-specific branching.
- Capability metadata is expressive enough to choose models for JSON, tools, code, structured output, and LoRA-backed reviewer paths.
- Cache hints are concrete enough for a Worker implementation using KV/Cache API/D1/R2 without baking Cloudflare storage into the library.
- Route inspection returns enough explanation for a gateway debug endpoint.
- Public docs include a gateway/router integration example that avoids private Stackbilt-only assumptions.

## #85: Clarify AI Gateway response-cache contract and metadata
URL: https://github.com/Stackbilt-dev/llm-providers/issues/85  |  Labels: enhancement, P2-medium  |  Updated: 2026-06-09T21:54:15Z  |  Comments: 2

## Summary

AI Gateway response caching is different from provider prompt/prefix caching. Cloudflare's current AI Gateway caching docs make the distinction concrete:

- Gateway cache only applies to identical requests unless callers provide `cf-aig-cache-key`.
- Per-request controls are `cf-aig-cache-ttl`, `cf-aig-skip-cache`, and `cf-aig-cache-key`.
- Responses expose cache status with `cf-aig-cache-status` as `HIT` or `MISS`.

`llm-providers` already has pieces of this surface (`GatewayMetadata.cacheKey`, `GatewayMetadata.cacheTtl`, `ResponseCacheAdapter`, and `CacheHints.strategy: 'response' | 'both'`), but downstream consumers still need to know too much about which fields affect Cloudflare AI Gateway versus local response-cache adapters.

Docs:

- https://developers.cloudflare.com/ai-gateway/features/caching/

## Current code surface

- `BaseProvider.getAIGatewayHeaders()` forwards `cf-aig-cache-key` and `cf-aig-cache-ttl` when the provider base URL is AI Gateway.
- `GatewayMetadata` does not expose `skipCache`.
- `CacheHints.strategy: 'response'` is documented, but does not itself create Gateway response-cache headers or an explicit local adapter policy.
- Provider response metadata does not normalize `cf-aig-cache-status` when Gateway returns it.
- The factory-level `ResponseCacheAdapter` key is internal and separate from AI Gateway's cache key behavior.

## Proposed work

1. Add explicit `GatewayMetadata.skipCache?: boolean` mapped to `cf-aig-skip-cache: true` for HTTP-provider Gateway calls.
2. Add response metadata normalization for `cf-aig-cache-status` where provider adapters can access response headers.
3. Document how `CacheHints.strategy` relates to `GatewayMetadata` and `ResponseCacheAdapter`:
 - `provider-prefix`: provider/native prompt cache only.
 - `response`: response-cache policy only, but caller must provide Gateway metadata or a response cache adapter.
 - `both`: both layers, still separately observable.
4. Consider a small helper for deterministic cache key generation that consumers can call before dispatch, so gateway/custom cache keys are stable without copying factory internals.
5. Add tests that prove old Cloudflare headers still work and the new skip-cache/status surfaces are backward-compatible.

## Acceptance criteria

- `GatewayMetadata.skipCache` is exported, documented, and mapped to the current Cloudflare header name.
- Provider responses routed through AI Gateway can expose normalized cache status, at least as `response

## #83: Add cache observability and cold/warm canary guidance
URL: https://github.com/Stackbilt-dev/llm-providers/issues/83  |  Labels: documentation, enhancement, P2-medium  |  Updated: 2026-06-06T09:26:14Z  |  Comments: 0

## Summary

After the provider and Gateway cache controls are wired, downstream orchestrators need a cheap way to prove the cache layer is actually being used in edge-native deployments.

For coding-agent workloads, the useful metrics differ by cache layer:

- Workers AI prompt/prefix cache: cached input tokens and lower time-to-first-token on repeated prefixes with the same `x-session-affinity`.
- AI Gateway response cache: `cf-aig-cache-status` (`HIT` / `MISS`) and lower end-to-end latency on exact repeated requests.
- Factory/local response cache: adapter hit/miss and zero provider spend on hits.

Docs:

- https://developers.cloudflare.com/workers-ai/features/prompt-caching/
- https://developers.cloudflare.com/ai-gateway/features/caching/

## Proposed work

1. Extend observability hook payloads or response metadata so cache fields are easy to log without provider-specific parsing:
 - provider prompt cache: `cachedInputTokens`, `cacheReadInputTokens`, `cacheWriteInputTokens`
 - AI Gateway response cache: cache status (`HIT`, `MISS`, or absent)
 - factory response cache: local adapter hit/miss
2. Add a small test/canary recipe that sends a cold request and a warm repeated request using:
 - stable system prompt and tool definitions at the front
 - dynamic user content at the end
 - a stable `cache.sessionId`
 - optional Gateway `cacheKey`/`cacheTtl` for exact-response-cache cases
3. Update README guidance so downstream agent orchestrators know how to structure prompts for prefix caching:
 - static content first
 - avoid timestamps in system prompts
 - keep tool definitions stable
 - append dynamic/user-specific content later

## Acceptance criteria

- Consumers can log cache hit/miss behavior from the normalized response/hook surface without inspecting raw provider responses.
- A documented canary can verify cold/warm behavior for Workers AI prefix caching and AI Gateway response caching separately.
- README examples distinguish prefix-cache optimization from response-cache optimization.
- Tests cover metadata/hook behavior for cached responses and provider responses with cached token fields.

## Notes

This should be a proof/observability layer, not a production deploy task. Live Workers AI / AI Gateway verification can happen downstream in `aegis-daemon` or another Cloudflare Worker once the library exposes the needed surfaces.

## #80: provider proposal: Cohere integration
URL: https://github.com/Stackbilt-dev/llm-providers/issues/80  |  Labels: none  |  Updated: 2026-05-31T11:14:44Z  |  Comments: 0

## Summary

Preliminary provider proposal: evaluate adding Cohere as a first-class provider to `@stackbilt/llm-providers`.

## Why this came up

During OSS imagery generation for this repo, the text-to-image model hallucinated Cohere as an integrated provider. That is not currently true. The package currently supports OpenAI, Anthropic, Cloudflare Workers AI, Cerebras, Groq, and NVIDIA NIM.

## Initial scope questions

- Should the provider target Cohere Chat/Command models only, or include embeddings/rerank APIs in a separate surface?
- Which capabilities map cleanly to the existing `LLMProvider` contract: streaming, tool/function calling, structured output, citations/search, usage accounting, cache metadata?
- Which models should enter the catalog initially, and what use cases should they be tagged for?
- Should `LLMProviders.fromEnv()` discover `COHERE_API_KEY`?

## Preliminary acceptance criteria

- Verify current Cohere request/response/stream shapes against primary documentation or live fixtures before implementation.
- Add provider config type, factory wiring, `fromEnv()` detection, catalog entries, capability metadata, and model drift tests.
- Add schema drift fixtures/tests for non-streaming and streaming responses.
- Document capability boundaries clearly, especially if embeddings/rerank are out of scope for this text-generation package.
- Keep imagery/marketing copy accurate: do not list Cohere as integrated until this ships.

## Notes

This is not a bug in current runtime behavior. It is a product/package roadmap issue created to convert a hallucinated visual claim into an explicit tracking primitive.

## #79: provider proposal: Google Vertex AI integration
URL: https://github.com/Stackbilt-dev/llm-providers/issues/79  |  Labels: none  |  Updated: 2026-05-31T11:14:39Z  |  Comments: 0

## Summary

Preliminary provider proposal: evaluate adding Google Vertex AI / Gemini-on-Vertex as a first-class provider to `@stackbilt/llm-providers`.

## Why this came up

During OSS imagery generation for this repo, the text-to-image model hallucinated Vertex AI as an integrated provider. That is not currently true. The package currently supports OpenAI, Anthropic, Cloudflare Workers AI, Cerebras, Groq, and NVIDIA NIM.

## Initial scope questions

- Should this be a generic `VertexAIProvider`, a Gemini-specific provider, or separate Google AI Studio vs Vertex AI integrations?
- What auth surface should be supported in OSS: service account credentials, ADC, workload identity, API key where applicable, or consumer-provided fetch/token hooks?
- Which model families should enter the catalog initially?
- Which capabilities are native and stable: streaming, tool/function calling, JSON/structured output, vision, usage accounting, safety settings, cache semantics?
- Should `LLMProviders.fromEnv()` auto-discover this provider, or should auth complexity require explicit config?

## Preliminary acceptance criteria

- Verify current Vertex AI request/response/stream shapes against primary Google documentation or live fixtures before implementation.
- Add provider config type, factory wiring, catalog entries, capability metadata, and model drift tests.
- Add schema drift fixtures/tests for non-streaming and streaming responses.
- Add clear docs for auth setup and any unsupported surfaces.
- Keep imagery/marketing copy accurate: do not list Vertex AI as integrated until this ships.

## Notes

This is not a bug in current runtime behavior. It is a product/package roadmap issue created to convert a hallucinated visual claim into an explicit tracking primitive.

## #78: provider proposal: native Mistral AI integration
URL: https://github.com/Stackbilt-dev/llm-providers/issues/78  |  Labels: none  |  Updated: 2026-05-31T11:14:33Z  |  Comments: 0

## Summary

Preliminary provider proposal: evaluate adding a native Mistral AI provider to `@stackbilt/llm-providers`.

## Why this came up

During OSS imagery generation for this repo, the text-to-image model hallucinated Mistral as an integrated provider. That is not currently true: the package has Mistral model IDs through other providers, but no first-class `MistralProvider`.

Current related support:

- Cloudflare Workers AI includes `@cf/mistral/mistral-7b-instruct-v0.1`.
- NVIDIA NIM includes `mistralai/mistral-large-2-instruct`.

Those are provider-hosted models, not native Mistral API integration.

## Initial scope questions

- Should the repo support the native Mistral API as a first-class provider?
- Which models should enter the catalog initially?
- Does native Mistral support the same package surfaces we expect from providers: streaming, tool/function calling, structured output, usage accounting, prompt/cache metadata, and balance/health semantics?
- Should `LLMProviders.fromEnv()` discover `MISTRAL_API_KEY`?

## Preliminary acceptance criteria

- Add `MistralProvider` only after verifying current native API request/response/stream shapes against primary documentation or live fixtures.
- Add provider config type, factory wiring, `fromEnv()` detection, model catalog entries, and model drift tests.
- Add schema drift fixtures/tests for non-streaming and streaming responses.
- Document capabilities accurately in README and CHANGELOG.
- Keep imagery/marketing copy accurate: do not list Mistral as integrated until this ships.

## Notes

This is not a bug in current runtime behavior. It is a product/package roadmap issue created to convert a hallucinated visual claim into an explicit tracking primitive.

## #77: ci: supply-chain workflow fails before creating jobs
URL: https://github.com/Stackbilt-dev/llm-providers/issues/77  |  Labels: none  |  Updated: 2026-05-31T11:02:30Z  |  Comments: 0

## Summary

The package release path is green, but the push-triggered supply-chain workflow is failing before it creates jobs or logs.

Observed during the v1.11.0 release pass:

- CI run for faacc42 succeeded.
- Publish to npm release workflow for v1.11.0 succeeded and published @stackbilt/llm-providers@1.11.0 with provenance.
- `.github/workflows/supply-chain.yml` failed on push with no jobs and no downloadable logs.

Run evidence:

- Failed supply-chain run: https://github.com/Stackbilt-dev/llm-providers/actions/runs/26710627004
- Successful CI run: https://github.com/Stackbilt-dev/llm-providers/actions/runs/26710626027
- Successful publish run: https://github.com/Stackbilt-dev/llm-providers/actions/runs/26710714484

This failure also appears on earlier pushes, so it is not introduced by the v1.11.0 package changes.

## Likely area

`.github/workflows/supply-chain.yml` calls reusable workflows from `Stackbilt-dev/stackbilt_llc`:

- `.github/workflows/supply-chain-sbom.yml`
- `.github/workflows/supply-chain-dep-review.yml`

The referenced reusable workflow files exist at the pinned SHA, but GitHub marks the caller run as failed before materializing any jobs. That usually points to a workflow-reference, permissions, or reusable-workflow call validation problem.

## Acceptance criteria

- Pushes to `main` create supply-chain jobs instead of failing at workflow setup.
- SBOM artifact generation still runs for pushes.
- Dependency review still runs for PRs only.
- The workflow remains suitable for a public OSS package.

## Current release status

Not a v1.11.0 release blocker: `@stackbilt/llm-providers@1.11.0` is published and the publish workflow passed all package gates.
