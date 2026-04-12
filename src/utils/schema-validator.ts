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
   * in the path itself — use `items` to validate array element shapes.
   */
  path: string;
  type: SchemaFieldType;
  /**
   * If true, missing paths are allowed and skipped. Useful for fields that
   * are genuinely optional (e.g. stop_sequence on Anthropic).
   */
  optional?: boolean;
  /**
   * For type: 'array' — validate each element against a nested schema.
   *
   * - `shape`: a flat SchemaField[] applied to every element (all elements
   *   the same shape).
   * - `variants` + `discriminator`: discriminated union. Each element is
   *   routed by the value of `discriminator` (a field name on the element)
   *   to the matching variant schema. **Unknown discriminator values are
   *   allowed and skipped** — we want forward-compat for additive API
   *   changes (e.g. Anthropic adds a new content block type). Only *missing*
   *   discriminators or *wrong-typed* known variants trigger drift.
   */
  items?: {
    shape?: SchemaField[];
    discriminator?: string;
    variants?: Record<string, SchemaField[]>;
  };
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
 * Validate a single object against a flat SchemaField list, prefixing any
 * error path with `pathPrefix` so nested element errors read like
 * `content[2].id` instead of bare `id`.
 */
function validateFields(
  provider: string,
  data: unknown,
  fields: SchemaField[],
  pathPrefix: string
): void {
  if (data == null || typeof data !== 'object') {
    throw new SchemaDriftError(
      provider,
      pathPrefix || '$root',
      'object',
      describeType(data)
    );
  }

  for (const field of fields) {
    const fullPath = pathPrefix ? `${pathPrefix}.${field.path}` : field.path;
    const value = getPath(data, field.path);

    if (value === undefined) {
      if (field.optional) continue;
      throw new SchemaDriftError(provider, fullPath, field.type, 'undefined');
    }

    if (!matchesType(value, field.type)) {
      throw new SchemaDriftError(provider, fullPath, field.type, describeType(value));
    }

    if (field.type === 'array' && field.items) {
      validateArrayItems(provider, value as unknown[], field.items, fullPath);
    }
  }
}

/**
 * Validate the elements of an array against either a flat `shape` or a
 * discriminated `variants` map. Unknown discriminator values are
 * forward-compatible and skipped.
 */
function validateArrayItems(
  provider: string,
  items: unknown[],
  spec: NonNullable<SchemaField['items']>,
  arrayPath: string
): void {
  for (let i = 0; i < items.length; i++) {
    const element = items[i];
    const elementPath = `${arrayPath}[${i}]`;

    if (element == null || typeof element !== 'object') {
      throw new SchemaDriftError(provider, elementPath, 'object', describeType(element));
    }

    if (spec.discriminator && spec.variants) {
      const disc = (element as Record<string, unknown>)[spec.discriminator];
      if (typeof disc !== 'string') {
        throw new SchemaDriftError(
          provider,
          `${elementPath}.${spec.discriminator}`,
          'string',
          describeType(disc)
        );
      }
      const variantFields = spec.variants[disc];
      // Unknown discriminator value — forward-compat. Skip validation
      // rather than reject, so adding a new block type upstream doesn't
      // break every consumer on the next deploy.
      if (!variantFields) continue;
      validateFields(provider, element, variantFields, elementPath);
      continue;
    }

    if (spec.shape) {
      validateFields(provider, element, spec.shape, elementPath);
    }
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
  validateFields(provider, data, fields, '');
}
