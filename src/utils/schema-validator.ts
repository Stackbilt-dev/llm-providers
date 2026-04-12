/**
 * Response Envelope Schema Validator
 *
 * Zero-dependency runtime validator for provider response shapes. Used at the
 * provider boundary to detect when an upstream API silently changes its
 * response envelope — the classic "field renamed at 2am, parser throws at
 * 3am" failure mode.
 *
 * Philosophy: validate the *minimum* fields each provider's parser actually
 * reads. Don't re-type the full upstream schema — that creates brittle
 * over-specification that churns every time the provider adds an optional
 * field. Only the fields we touch matter.
 *
 * Usage:
 *   validateSchema('anthropic', data, [
 *     { path: 'content', type: 'array' },
 *     { path: 'usage.input_tokens', type: 'number' },
 *     { path: 'usage.output_tokens', type: 'number' },
 *     { path: 'model', type: 'string' },
 *   ]);
 *
 * On first failure, throws SchemaDriftError. The caller (typically the
 * factory) catches it, fires the onSchemaDrift hook, and falls over to
 * another provider.
 */

import { SchemaDriftError } from '../errors';

export type SchemaFieldType =
  | 'string'
  | 'number'
  | 'boolean'
  | 'array'
  | 'object'
  | 'string-or-null'; // anthropic/openai sometimes null content fields

export interface SchemaField {
  /**
   * Dot-separated path into the response object. Array indices not supported
   * because we validate shape, not contents — if you care about element
   * shape, validate the array type here and the elements at their own
   * parse site.
   */
  path: string;
  type: SchemaFieldType;
  /**
   * If true, missing paths are allowed and skipped. Useful for fields that
   * are genuinely optional (e.g. stop_sequence on Anthropic).
   */
  optional?: boolean;
}

/**
 * Walk a dot-path into an object. Returns undefined if any segment is missing.
 * Does NOT distinguish "path missing" from "path present but undefined" —
 * callers should treat both as missing.
 */
function getPath(obj: unknown, path: string): unknown {
  const segments = path.split('.');
  let current: unknown = obj;

  for (const segment of segments) {
    if (current == null || typeof current !== 'object') {
      return undefined;
    }
    current = (current as Record<string, unknown>)[segment];
  }

  return current;
}

/**
 * Describe the actual runtime type of a value for error messages.
 * Separates null / undefined / array from generic object.
 */
function describeType(value: unknown): string {
  if (value === null) return 'null';
  if (value === undefined) return 'undefined';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}

function matchesType(value: unknown, type: SchemaFieldType): boolean {
  switch (type) {
    case 'string':
      return typeof value === 'string';
    case 'number':
      return typeof value === 'number' && Number.isFinite(value);
    case 'boolean':
      return typeof value === 'boolean';
    case 'array':
      return Array.isArray(value);
    case 'object':
      return value !== null && typeof value === 'object' && !Array.isArray(value);
    case 'string-or-null':
      return value === null || typeof value === 'string';
  }
}

/**
 * Validate a response envelope against a minimal field schema.
 * Throws SchemaDriftError on the first mismatch, with provider + path +
 * expected/actual types surfaced for observability.
 *
 * We fail fast rather than collecting all errors: the first drift is enough
 * to trigger fallback, and walking the whole schema when we're already
 * broken wastes budget.
 */
export function validateSchema(
  provider: string,
  data: unknown,
  fields: SchemaField[]
): void {
  if (data == null || typeof data !== 'object') {
    throw new SchemaDriftError(
      provider,
      '$root',
      'object',
      describeType(data)
    );
  }

  for (const field of fields) {
    const value = getPath(data, field.path);

    if (value === undefined) {
      if (field.optional) continue;
      throw new SchemaDriftError(
        provider,
        field.path,
        field.type,
        'undefined'
      );
    }

    if (!matchesType(value, field.type)) {
      throw new SchemaDriftError(
        provider,
        field.path,
        field.type,
        describeType(value)
      );
    }
  }
}
