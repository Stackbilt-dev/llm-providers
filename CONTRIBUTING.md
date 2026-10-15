# Contributing to @stackbilt/llm-providers

Contributions are welcome. Here is how to get started.

## Setup

```bash
git clone https://github.com/Stackbilt-dev/llm-providers.git
cd llm-providers
npm install --ignore-scripts
```

## Development

For the current uncommitted implementation checkpoint, issue status, validation evidence and remaining work, read [the issue sweep checkpoint](docs/maintenance/issue-sweep.md). Historical issue exports and archive-upload status are not the active work queue.

```bash
npm run typecheck    # Type-check without emitting
npm test             # Run tests once
npm run test:watch   # Re-run on file changes
npm run build        # Compile to dist/
npm run test:worker-import # Build and check the Worker import graph
```

## Guidelines

1. **Keep the core logic intact.** This library was extracted from a production system. Changes to the provider implementations, circuit breaker, retry, or cost tracking logic should be well-motivated and tested.

2. **No new runtime dependencies.** The package has zero production dependencies by design. Dev dependencies (TypeScript, vitest, Cloudflare types) are fine.

3. **Type safety.** All public APIs must be fully typed. No `any` in exported signatures.

4. **Tests.** New features or bug fixes should include tests. Run `npm test` before submitting.

5. **Commits.** Use conventional commit messages (`feat:`, `fix:`, `docs:`, `chore:`).

## Adding a New Provider

1. Create `src/providers/<name>.ts` extending `BaseProvider`.
2. Implement all abstract methods: `generateResponse`, `validateConfig`, `getModels`, `estimateCost`, `healthCheck`.
3. Add the provider to `LLMProviderFactory.initializeProviders()` in `src/factory.ts`.
4. Export from `src/index.ts`.
5. Add tests in `src/__tests__/`.

For streaming adapters, use `BaseProvider.createStream()` and the shared incremental decoder in `src/utils/provider-stream.ts` so usage, cancellation, deadlines and breaker outcomes settle consistently. Do not drain the response in an eager `start` loop or accumulate its full text solely to estimate tokens. Add fault/cancellation tests and check the managed factory completion path. Record fixture provenance and distinguish synthetic Node tests from live provider or workerd validation.

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 license.
