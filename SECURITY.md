# Security Policy

## Supply Chain Security

`@stackbilt/llm-providers` is designed with supply chain security as a first-class concern:

- **Zero runtime dependencies.** This package has no `dependencies` in `package.json`. The published npm tarball contains only our compiled code and the Apache-2.0 license. No transitive dependency tree to audit, no hidden packages to compromise.
- **npm provenance attestation.** Every published version includes a [provenance attestation](https://docs.npmjs.com/generating-provenance-statements) cryptographically linking the npm package to the exact GitHub commit and CI workflow that built it. Verify this on the npm registry page under "Provenance."
- **CI-only publishing.** Releases are published exclusively through GitHub Actions with OIDC-based npm authentication. No human has `npm publish` credentials — the publish token is scoped to the CI workflow and cannot be used outside of it.
- **Signed commits.** All maintainer commits to `main` are signed.
- **Branch protection.** The `main` branch requires passing CI (typecheck + full test suite) before merge.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |
| < 1.0   | No        |

We support the latest major version with security patches. Older major versions receive critical security fixes for 6 months after the next major release.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email **security@stackbilt.dev** with:

1. Description of the vulnerability
2. Steps to reproduce
3. Impact assessment (what can an attacker do?)
4. Any suggested fix

We will acknowledge receipt within 48 hours and provide an initial assessment within 5 business days. Critical vulnerabilities affecting user credentials or API keys will be patched within 24 hours of confirmation.

## Security Design

### What this package handles

- **API key transport.** Keys are passed via constructor config and sent only to the configured provider endpoints over HTTPS. Keys are never logged, serialized to disk, or sent to any Stackbilt service.
- **Circuit breakers.** Prevent cascading failures from propagating across providers. A failing provider is isolated, not retried indefinitely.
- **Rate limiting.** Configurable per-provider rate limits prevent accidental quota exhaustion or abuse amplification.
- **Cost controls.** CreditLedger tracks spend per provider with configurable monthly budgets and threshold alerts.

### What this package does NOT do

- Store or cache API keys beyond the in-memory provider instance.
- Make network requests to any endpoint other than the configured LLM provider APIs.
- Phone home, collect telemetry, or transmit usage data.
- Execute arbitrary code from provider responses.

### Recommendations for users

- **Rotate API keys** if you suspect any compromise of your environment.
- **Set budget limits** via CreditLedger to cap spend in case of unexpected usage spikes.
- **Use environment variables** for API keys, never hardcode them in source.
- **Pin versions** in production (`@stackbilt/llm-providers@1.0.0`, not `^1.0.0`) for maximum reproducibility.
- **Verify provenance** on the npm registry page before adopting a new version.

## Audit Trail

This package undergoes the following automated checks on every commit:

- TypeScript strict mode compilation
- Full test suite (unit + integration)
- `npm audit` for known vulnerabilities in dev dependencies
- Provenance attestation on publish

For questions about our security practices, email security@stackbilt.dev.
