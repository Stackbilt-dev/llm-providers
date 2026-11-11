/**
 * LLM Provider Error Classes
 * Structured error handling for all LLM providers
 */

import type { LLMError } from './types.js';

export class LLMProviderError extends Error implements LLMError {
  code: string;
  provider: string;
  retryable: boolean;
  statusCode?: number;
  rateLimited?: boolean;
  quotaExceeded?: boolean;

  constructor(
    message: string,
    code: string,
    provider: string,
    retryable: boolean = false,
    statusCode?: number
  ) {
    super(message);
    this.name = 'LLMProviderError';
    this.code = code;
    this.provider = provider;
    this.retryable = retryable;
    this.statusCode = statusCode;
  }
}

export class RateLimitError extends LLMProviderError {
  constructor(provider: string, message: string = 'Rate limit exceeded') {
    super(message, 'RATE_LIMIT', provider, true, 429);
    this.rateLimited = true;
  }
}

export class QuotaExceededError extends LLMProviderError {
  constructor(provider: string, message: string = 'Quota exceeded') {
    super(message, 'QUOTA_EXCEEDED', provider, false, 429);
    this.quotaExceeded = true;
  }
}

/**
 * Local policy denial; never retry, fail over, or quarantine a provider.
 * Extends QuotaExceededError (same code/status) so existing quota-hook denial
 * checks keep matching; use `instanceof QuotaDeniedError` or `localPolicy` to
 * distinguish it from vendor account exhaustion.
 */
export class QuotaDeniedError extends QuotaExceededError {
  readonly localPolicy = true;

  constructor(provider: string, message: string = 'Local quota policy denied request') {
    super(provider, message);
  }
}

export class AuthenticationError extends LLMProviderError {
  constructor(provider: string, message: string = 'Authentication failed') {
    super(message, 'AUTHENTICATION_ERROR', provider, false, 401);
  }
}

export class InvalidRequestError extends LLMProviderError {
  constructor(provider: string, message: string = 'Invalid request') {
    super(message, 'INVALID_REQUEST', provider, false, 400);
  }
}

/**
 * `reasoning.strict` could not be honoured: the model can't take the reasoning
 * effort exactly (thrown by the Cloudflare provider), or no provider that enforces
 * strict can serve the request (thrown by the factory, provider `'factory'`: a pin
 * on another provider, a pin that can't be served, or no Cloudflare provider).
 * The factory never falls back on this error: another provider could not honour
 * the level either, so failing over would drop the guarantee.
 */
export class ReasoningEffortUnsupportedError extends InvalidRequestError {
  constructor(provider: string, message: string) {
    super(provider, message);
  }
}

export class ModelNotFoundError extends LLMProviderError {
  constructor(provider: string, model: string) {
    super(`Model '${model}' not found`, 'MODEL_NOT_FOUND', provider, false, 404);
  }
}

export class TimeoutError extends LLMProviderError {
  constructor(provider: string, message: string = 'Request timeout') {
    super(message, 'TIMEOUT', provider, true, 408);
  }
}

export class NetworkError extends LLMProviderError {
  constructor(provider: string, message: string = 'Network error') {
    super(message, 'NETWORK_ERROR', provider, true, 503);
  }
}

export class ServerError extends LLMProviderError {
  constructor(provider: string, message: string = 'Internal server error', statusCode: number = 500) {
    super(message, 'SERVER_ERROR', provider, true, statusCode);
  }
}

export class ContentFilterError extends LLMProviderError {
  constructor(provider: string, message: string = 'Content filtered') {
    super(message, 'CONTENT_FILTER', provider, false, 400);
  }
}

export class TokenLimitError extends LLMProviderError {
  constructor(provider: string, message: string = 'Token limit exceeded') {
    super(message, 'TOKEN_LIMIT', provider, false, 400);
  }
}

export class ConfigurationError extends LLMProviderError {
  constructor(provider: string, message: string) {
    super(message, 'CONFIGURATION_ERROR', provider, false);
  }
}

export class StructuredOutputUnsupportedError extends ConfigurationError {
  constructor(provider: string, message: string) {
    super(provider, message);
    this.code = 'STRUCTURED_OUTPUT_UNSUPPORTED';
  }
}

export class NoRouteError extends ConfigurationError {
  constructor(message: string = 'No configured provider can serve this request') {
    super('factory', message);
    this.code = 'NO_ROUTE';
  }
}

/**
 * A pinned model whose catalog context window can't hold the request's input, with
 * `contextWindowPinCheck` on, when nothing else may serve the request: `allowFallback`
 * is false, or no other provider can be called. Thrown before any call. A
 * `NoRouteError` (code `NO_ROUTE`), so existing handling still applies.
 */
export class ContextWindowExceededError extends NoRouteError {
  /** The provider that serves the pinned model. */
  modelProvider: string;
  /** The pinned model, as sent: after any correction of an outdated model ID. */
  model: string;
  /** The model's context window, from the catalog. */
  contextWindow: number;
  /** The request's input tokens: the caller's `inputTokenCount`, else the estimate. */
  inputTokens: number;
  /**
   * Where `inputTokens` came from: `'caller'` is your `inputTokenCount`; `'mixed'` is
   * your count plus an estimate of what the package added (messages appended by
   * `generateResponseWithTools`, or a system prompt `classify` supplies);
   * `'estimate'` is the package's estimate alone.
   */
  inputTokensSource: 'caller' | 'mixed' | 'estimate';
  /**
   * Why nothing else was tried: `'allowFallback-false'`, or `'no-other-provider'` when
   * no other provider with a planned model that holds the input can be called (none
   * configured, the rest removed by `reasoning.strict` or image support, their planned
   * models too small, or every provider that fits unavailable or unable to stream).
   */
  reason: 'allowFallback-false' | 'no-other-provider';

  constructor(details: {
    modelProvider: string;
    model: string;
    contextWindow: number;
    inputTokens: number;
    inputTokensSource: 'caller' | 'mixed' | 'estimate';
    reason: 'allowFallback-false' | 'no-other-provider';
  }) {
    const estimated = details.inputTokensSource === 'estimate'
      ? ' (estimated)'
      : details.inputTokensSource === 'mixed' ? ' (partly estimated)' : '';
    const why = details.reason === 'allowFallback-false'
      ? 'and allowFallback is false'
      : 'and no other provider can serve the request';
    super(
      `Pinned model '${details.model}' (${details.modelProvider}) has a ${details.contextWindow}-token context window, `
      + `which can't hold the request's ${details.inputTokens} input tokens${estimated}, ${why}`
    );
    this.name = 'ContextWindowExceededError';
    this.modelProvider = details.modelProvider;
    this.model = details.model;
    this.contextWindow = details.contextWindow;
    this.inputTokens = details.inputTokens;
    this.inputTokensSource = details.inputTokensSource;
    this.reason = details.reason;
  }
}

export class CfGatewayInvalidConfigError extends LLMProviderError {
  /** The specific cfGateway field that was empty or missing. */
  field: 'accountId' | 'gatewayId';

  constructor(provider: string, field: 'accountId' | 'gatewayId') {
    super(
      `cfGateway.${field} must be a non-empty string for the ${provider} provider`,
      'CF_GATEWAY_INVALID_CONFIG',
      provider,
      false
    );
    this.field = field;
  }
}

export class CircuitBreakerOpenError extends LLMProviderError {
  retryAfterSec: number;
  consecutiveFailures: number;

  constructor(provider: string, retryAfterSec: number = 0, consecutiveFailures: number = 0) {
    const retryHint = retryAfterSec > 0 ? ` Retry after ${retryAfterSec}s.` : '';
    super(
      `Circuit breaker rejected request for provider: ${provider}.${retryHint}`,
      'CIRCUIT_BREAKER_OPEN',
      provider,
      true
    );
    this.retryAfterSec = retryAfterSec;
    this.consecutiveFailures = consecutiveFailures;
  }
}

export class ToolLoopLimitError extends LLMProviderError {
  constructor(provider: string, message: string = 'Tool loop limit exceeded') {
    super(message, 'TOOL_LOOP_LIMIT', provider, false, 400);
  }
}

export class ToolLoopAbortedError extends LLMProviderError {
  constructor(provider: string, message: string = 'Tool loop aborted') {
    super(message, 'TOOL_LOOP_ABORTED', provider, false, 400);
  }
}

/**
 * Thrown when a provider's response envelope fails runtime schema validation.
 *
 * Indicates the upstream API silently changed shape - a field was renamed,
 * removed, or had its type changed. Non-retryable: retrying hits the same
 * broken shape. The factory treats this as fallback-eligible so traffic
 * routes to a healthy provider while the drift is investigated.
 */
export class SchemaDriftError extends LLMProviderError {
  path: string;
  expected: string;
  actual: string;

  constructor(provider: string, path: string, expected: string, actual: string) {
    super(
      `Response schema drift at ${path}: expected ${expected}, got ${actual}`,
      'SCHEMA_DRIFT',
      provider,
      false,
      422
    );
    this.path = path;
    this.expected = expected;
    this.actual = actual;
  }
}

/**
 * Error factory for creating provider-specific errors from HTTP responses
 */
/** Shape of error response bodies across all LLM providers. */
interface ErrorResponseBody {
  message?: string;
  error?: {
    message?: string;
    code?: string;
    type?: string;
    param?: string;
  };
}

export class LLMErrorFactory {
  static fromHttpResponse(
    provider: string,
    statusCode: number,
    responseBody?: ErrorResponseBody,
    message?: string
  ): LLMProviderError {
    const defaultMessage = message || responseBody?.message || responseBody?.error?.message || 'Unknown error';

    switch (statusCode) {
      case 400:
        if (responseBody?.error?.code === 'content_filter') {
          return new ContentFilterError(provider, defaultMessage);
        }
        if (responseBody?.error?.code === 'token_limit_exceeded') {
          return new TokenLimitError(provider, defaultMessage);
        }
        return new InvalidRequestError(provider, defaultMessage);

      case 401:
        return new AuthenticationError(provider, defaultMessage);

      case 404: {
        const model = responseBody?.error?.param || 'unknown';
        return new ModelNotFoundError(provider, model);
      }

      case 408:
        return new TimeoutError(provider, defaultMessage);

      case 429:
        if (responseBody?.error?.type === 'quota_exceeded') {
          return new QuotaExceededError(provider, defaultMessage);
        }
        return new RateLimitError(provider, defaultMessage);

      case 500:
      case 502:
      case 503:
      case 504:
        return new ServerError(provider, defaultMessage, statusCode);

      default:
        return new LLMProviderError(
          defaultMessage,
          'UNKNOWN_ERROR',
          provider,
          statusCode >= 500,
          statusCode
        );
    }
  }

  /**
   * Create error from fetch response
   */
  static async fromFetchResponse(
    provider: string,
    response: Response
  ): Promise<LLMProviderError> {
    let responseBody: ErrorResponseBody;

    try {
      const contentType = response.headers.get('content-type');
      if (contentType?.includes('application/json')) {
        responseBody = await response.json() as ErrorResponseBody;
      } else {
        responseBody = { message: await response.text() };
      }
    } catch {
      responseBody = { message: 'Failed to parse error response' };
    }

    return this.fromHttpResponse(provider, response.status, responseBody);
  }

  /**
   * Check if error is retryable
   */
  static isRetryable(error: Error): boolean {
    if (error instanceof LLMProviderError) {
      return error.retryable;
    }

    // Network errors are generally retryable
    if (error.message.includes('fetch')) {
      return true;
    }

    return false;
  }

  /**
   * Check if error indicates rate limiting
   */
  static isRateLimited(error: Error): boolean {
    return error instanceof RateLimitError;
  }

  /**
   * Check if error indicates quota exceeded
   */
  static isQuotaExceeded(error: Error): boolean {
    return error instanceof QuotaExceededError;
  }

  /**
   * Get retry delay from rate limit error
   */
  static getRetryDelay(error: Error): number {
    if (error instanceof RateLimitError) {
      // Default exponential backoff starting at 1 second
      return 1000;
    }
    if (error instanceof ServerError) {
      // Server errors get shorter delays
      return 500;
    }
    return 0;
  }
}
