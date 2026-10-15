# Issue sweep checkpoint

Checkpoint: 2026-10-12. Repository: `Stackbilt-dev/llm-providers`.

## Resume here

The implementation is **local, uncommitted, unpushed and unreleased** on `main`, based on `8a9eda6c058d63f0ec08332f6a408aaf409232cf` (package version `1.21.0`). GitHub issue comments describe local progress, not merged fixes. Keep the issues open until the corresponding changes are reviewed and integrated. No live or billable vendor requests were made.

1. Inspect `git status --short` and this checkpoint. Preserve the pre-existing `src/__tests__/factory.test.ts` modification; its diff is line-ending-only when checked with `--ignore-space-at-eol`. This sweep did not edit that file.
2. Review the local changes, particularly streaming terminal races, accounting policy and routing compatibility. Do not discard or reimplement the work already present.
3. Next implementation targets: #112 vendor IDs/pricing, then #109 atomic spend reservations. #116 remains a serious non-streaming telemetry replay defect and is a useful smaller intervening fix.
4. Run the validation commands below after substantive code changes, then prepare reviewable commits/PRs. The new public APIs require a minor release; `package.json`, the lockfile and `VERSION` have not been bumped.

`issue-corpus/Stackbilt-dev__llm-providers/` is a June 2026 historical export. `.agent-memory/status.json` belongs to the archive/upload machinery. Neither is the current task checkpoint; preserve their historical/managed contents.

## Implementation and issue state

| Issue | Local work completed | Remaining before closing the full issue |
|---|---|---|
| [#94](https://github.com/Stackbilt-dev/llm-providers/issues/94) | `inlineSources` embeds TypeScript in JavaScript maps; package smoke adds a map-content check. Final packed artifact contains 33 verified `.js.map` files. | Review, CI and integration. Declaration maps are a separate surface; the check targets the JavaScript maps described in this issue. |
| [#108](https://github.com/Stackbilt-dev/llm-providers/issues/108) | Versioned SHA-256 keys cover semantic inputs, tenant, application namespace, configured provider transport/account options and routing policy. Explicit keys stay scoped. `off`/`provider-prefix` bypass local caching; malformed entries/non-JSON identities bypass safely. Hits check local quota at zero new estimated spend and mark usage historical. | Review and integration; an actual Workers runtime crypto/cache test remains. Version 2 deliberately ignores old keys. Use a namespace to distinguish AI bindings/policy revisions that cannot be serialized. |
| [#109](https://github.com/Stackbilt-dev/llm-providers/issues/109) | `QuotaDeniedError` terminates response, image and streaming requests without fallback or provider exhaustion. Throwing quota observers cannot fail open. Vendor account exhaustion remains separate. Streaming settlement is awaited. | Atomic reserve/settle coordinator, in-flight money/request/token admission, retry-attempt accounting, leases/rollover/idempotency, conservative unknown usage and a durable implementation/example. Non-streaming quota writes still float. Ledger budgets remain soft analytics/routing signals. |
| [#110](https://github.com/Stackbilt-dev/llm-providers/issues/110) | Shared incremental SSE decoder across six adapters: split UTF-8, CR/LF/CRLF, multiline data, comments, terminal markers and protocol errors. Demand-driven reads, upstream cancellation, HTTP fetch aborts, stream body deadline, settled partial usage on error/cancel, stream health recording. Factory awaits accounting on terminal paths and exposes `generateManagedResponseStream()`. Reported pre-output failure usage is recorded; known unbilled admission/HTTP rejections are not charged. | Provider-level pre-output retries, an absolute deadline/signal spanning retries/fallbacks, recorded live-shaped fixtures and a Workers runtime lifecycle test. Workers AI binding work cannot be aborted before the binding returns; its eventual reader is cancelled. |
| [#111](https://github.com/Stackbilt-dev/llm-providers/issues/111) | `stateScope: 'client'`, injectable/shared `ResilienceContext`, adapter breaker disablement, rejected-call decay fix, neutral caller/account/policy errors, idempotent streaming breaker permits and completion-order tests. Explicit provider `maxRetries` retains precedence over legacy `enableRetries: false`. | Isolated defaults belong in the next major. Strict retry disablement would need a deliberate compatible policy/migration. Unsupported latency rules and cost-rule semantics are unchanged and need resolution. |
| [#113](https://github.com/Stackbilt-dev/llm-providers/issues/113) | Explicit canonical/legacy normalization preserves metadata-only identity and defines system-message precedence. Client `getRoutingInfo()` uses the execution planner/admission state. Catalog introspection chooses configured hosts for shared IDs and checks built-in tools; `NoRouteError` covers missing routes. Diagnostic gateway plans expose `executable: false` for failed capabilities. | Canonical execution API; hard requirements/output/context/streaming/tool capabilities across every fallback; distinct native JSON capability metadata; custom-model/pinned-model policy; actual attempt/degradation metadata; next-major convergence. The old union helper remains ambiguous by design. |
| [#85](https://github.com/Stackbilt-dev/llm-providers/issues/85) | Local adapter policy and the three cache layers are documented; #108 hardens factory caching. | Review existing HTTP AI Gateway skip/status surfaces against this issue independently; no claim that factory caching implements Gateway caching or its remote key policy. |
| [#87](https://github.com/Stackbilt-dev/llm-providers/issues/87) | Public normalization boundaries, client preflight, managed stream completion, diagnostic executability and cache semantics improve router contracts. | Broad canonical execution/capability enforcement and complete gateway integration remain. |
| [#115](https://github.com/Stackbilt-dev/llm-providers/issues/115) | #94's map fix and a final packed-public-export check are covered. Local Windows build tools now run. | Explicit ESM/CJS builds, subpaths/provider registration, Node minimum-version/NodeNext/workerd matrix, measured bundle budgets/startup. Current Node 24 `require` success is not proof of Node 20.0 compatibility; package still emits the module-type warning. |
| [#116](https://github.com/Stackbilt-dev/llm-providers/issues/116) | Quota observers and streaming terminal/schema observers are guarded; stream body deadlines, cancellation, fault/race tests and awaited stream settlement were added. | Raw non-streaming lifecycle hooks can still replay a completed request; normalize hook invocation centrally. Non-streaming body deadlines, retry sleep aborts, Retry-After/vendor error fixtures, unproduced hook events, floating non-streaming cache/quota writes, request/tool-loop signals, workerd and coverage baselines remain. |
| [#118](https://github.com/Stackbilt-dev/llm-providers/issues/118) | Coordinating roadmap receives this checkpoint and priorities. | Track reviewed/merged implementation phases separately from local progress. |

Other open items are unchanged: #112 (vendor audit, next), #114 (failure demo/README restructuring/repo cleanup), #117 (optional AI SDK adapter), #77 (supply-chain workflow), #78/#79/#80 (new provider proposals). #90/#95 have existing code surfaces but were not re-audited for closure during this sweep.

## Public behavior to retain

- New exports: `QuotaDeniedError`, `NoRouteError`, `ResilienceContext`, `normalizeCanonicalRequest`, `normalizeLegacyRequest`, `ManagedTextStream`, `StreamCompletion`.
- Client/factory methods: `generateManagedResponseStream(request)` and `getRoutingInfo(request)`. Legacy `generateResponseStream(request)` returns only the managed result's string stream.
- Managed completion: `{ outcome: 'success' | 'error' | 'cancelled', emittedChars, usage }`; resolves after required accounting, rejects on settlement failure. Consume/cancel the stream before awaiting completion. Attach the promise to `ctx.waitUntil` where appropriate.
- Reported partial usage is retained; when unavailable, factory accounting conservatively uses the preflight cost estimate with estimated provenance. This is not an actual provider charge or an enforced USD cap.
- Streaming `onRequestEnd` receives `outcome`; partial errors can also emit `onRequestError`. Cancellation is neutral for breaker health. No post-output failover.
- Global resilience state remains the compatibility default; explicit client state isolates health/exhaustion/cost analytics/latency. An explicit context wins over `stateScope`.
- Metadata-only canonical input must use `normalizeCanonicalRequest`; the compatibility union helper defaults structurally ambiguous input to legacy. Legacy system messages join with blank lines unless `systemPrompt` overrides them.
- `GatewayRoutePlan.executable` reflects the existing diagnostic failed-capability checks; it is **not** a guarantee of native schema support or full execution-time hard-requirement enforcement.

## Files and tests

Main changes: [factory](../../src/factory.ts), [public exports](../../src/index.ts), [types](../../src/types.ts), [errors](../../src/errors.ts), [canonical conversion](../../src/canonical.ts), [catalog](../../src/model-catalog.ts), [gateway diagnostics](../../src/gateway-routing.ts), all six provider streaming methods, [base provider](../../src/providers/base.ts), [breaker](../../src/utils/circuit-breaker.ts), [hook types](../../src/utils/hooks.ts), `tsconfig.json`, and `scripts/package-smoke.sh`.

New utilities: [response cache identity/validation](../../src/utils/response-cache.ts), [provider SSE lifecycle](../../src/utils/provider-stream.ts), [resilience context](../../src/utils/resilience-context.ts).

New regression suites: [tenant safety](../../src/__tests__/tenant-safety.test.ts) (45 tests), [stream lifecycle](../../src/__tests__/stream-lifecycle.test.ts) (35), [context ownership/recovery](../../src/__tests__/resilience-context.test.ts) (12), [canonical identity/routing](../../src/__tests__/canonical-identity-routing.test.ts) (9). Existing schema-drift tests now reset breaker history between cases; gateway diagnostic tests check failed-plan executability. The pre-existing factory test file was preserved.

## Validation evidence

Host: Windows/PowerShell, Node `v24.19.0`, npm `11.17.0`.

- `npm test`: **605 passing tests across 27 files**.
- `npm run typecheck`: passed.
- `npm run test:worker-import`: build and browser/Worker-targeted bundle smoke passed.
- Packed artifact: unpacked in a temporary directory, verified 33 JavaScript maps have complete nonempty `sourcesContent`, and required the packed entrypoint to verify all new public exports and client methods. The Linux-oriented `npm run test:package` shell script itself was not run on this Windows host; its artifact checks were exercised manually.
- Whitespace check passed for the sweep's modified tracked files with CR-at-EOL allowed, excluding the preserved factory-test diff.
- No live providers, workerd execution, minimum Node version, NodeNext matrix, measured bundle size, coverage baseline or published-release validation was performed.

The initial Linux-oriented local install lacked Windows Rollup/esbuild binaries. Windows packages were installed with `--ignore-scripts --no-save --package-lock=false`. No dependency-manifest or lockfile change belongs to this sweep. Existing npm audit findings were not remediated here.

Repeat after code changes:

```powershell
git -c safe.directory=C:/Users/kover/Documents/stackbilt_eco/llm-providers status --short
npm.cmd test
npm.cmd run typecheck
npm.cmd run test:worker-import
```

Git's ownership guard applies on this host. Use the per-command `safe.directory` override; no global Git setting was changed. `npm pack` can require approved npm-cache access outside the workspace; use a temporary pack destination to avoid overwriting existing archives.

## GitHub checkpoint comments

Posted on 2026-10-12. Every comment states that the work is local and pending review/integration; none closes an issue or marks an unreleased fix as shipped. The original audit/acceptance text was preserved.

| Issue | Posted checkpoint |
|---|---|
| #85 | [Cache-layer contract](https://github.com/Stackbilt-dev/llm-providers/issues/85#issuecomment-6136617619) |
| #87 | [Router contract](https://github.com/Stackbilt-dev/llm-providers/issues/87#issuecomment-6136618031) |
| #94 | [Embedded sourcemaps](https://github.com/Stackbilt-dev/llm-providers/issues/94#issuecomment-6136618468) |
| #108 | [Tenant-safe caching](https://github.com/Stackbilt-dev/llm-providers/issues/108#issuecomment-6136618918) |
| #109 | [Local denial and remaining reservations](https://github.com/Stackbilt-dev/llm-providers/issues/109#issuecomment-6136619380) |
| #110 | [Streaming lifecycle](https://github.com/Stackbilt-dev/llm-providers/issues/110#issuecomment-6136619898) |
| #111 | [State isolation and breaker policy](https://github.com/Stackbilt-dev/llm-providers/issues/111#issuecomment-6136620389) |
| #113 | [Canonical identity and route planning](https://github.com/Stackbilt-dev/llm-providers/issues/113#issuecomment-6136620841) |
| #114 | [Documentation and remaining demo work](https://github.com/Stackbilt-dev/llm-providers/issues/114#issuecomment-6136621403) |
| #115 | [Artifact evidence and remaining packaging](https://github.com/Stackbilt-dev/llm-providers/issues/115#issuecomment-6136621809) |
| #116 | [Fault tests and remaining telemetry replay](https://github.com/Stackbilt-dev/llm-providers/issues/116#issuecomment-6136622246) |
| #118 | [Coordinating roadmap and next priorities](https://github.com/Stackbilt-dev/llm-providers/issues/118#issuecomment-6136622703) |
