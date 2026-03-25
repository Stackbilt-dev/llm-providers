# Contributing to @stackbilt/llm-providers

Contributions are welcome. Here is how to get started.

## Setup

```bash
git clone https://github.com/Stackbilt-dev/llm-providers.git
cd llm-providers
npm install
```

## Development

```bash
npm run typecheck    # Type-check without emitting
npm test             # Run tests once
npm run test:watch   # Re-run on file changes
npm run build        # Compile to dist/
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

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 license.
