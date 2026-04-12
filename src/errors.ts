/**
 * LLM Provider Error Classes
 * Structured error handling for all LLM providers
 */

import type { LLMError } from './types';

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
      502
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
