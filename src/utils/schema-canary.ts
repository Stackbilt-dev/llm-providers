/**
 * Schema Drift Canary (#39 Part 2)
 *
 * Utilities for detecting API response shape drift against committed golden fixtures.
 *
 * Usage pattern:
 *   1. Fetch a raw provider response (integration test or cron Worker).
 *   2. Load the golden fixture for that provider.
 *   3. Call runCanaryCheck(provider, golden, liveResponse) to get a CanaryReport.
 *   4. Report / alert on drift.status === 'drift'.
 *
 * Scheduling and transport are left to the consumer. This module contains no I/O.
 */

/** Flat map of dot-notation path → JSON type name. */
export type ShapeMap = Record<string, string>;

export interface CanaryDiff {
  /** Paths present in the live response but absent from the golden fixture. */
  added: string[];
  /** Paths present in the golden fixture but absent from the live response. */
  removed: string[];
  /** Paths present in both, but with differing JSON types. */
  changed: string[];
}

export interface CanaryReport {
  provider: string;
  status: 'ok' | 'drift';
  diff: CanaryDiff;
}

/**
 * Walk an arbitrary value and return a flat map of dot-notation paths → JSON type names.
 * Arrays are represented by their element type at index 0 (path suffix `[0]`).
 * Recursion stops at depth 6 to keep fixtures stable against deep nesting.
 */
export function extractShape(obj: unknown, prefix = '', depth = 0): ShapeMap {
  const result: ShapeMap = {};

  if (depth > 6) return result;

  const typeName = jsonTypeName(obj);
  if (prefix) result[prefix] = typeName;

  if (typeName === 'object' && obj !== null) {
    for (const [key, value] of Object.entries(obj as Record<string, unknown>)) {
      const path = prefix ? `${prefix}.${key}` : key;
      Object.assign(result, extractShape(value, path, depth + 1));
    }
  } else if (typeName === 'array' && Array.isArray(obj) && obj.length > 0) {
    Object.assign(result, extractShape(obj[0], `${prefix}[0]`, depth + 1));
  }

  return result;
}

/**
 * Compare a live shape map against a golden fixture.
 * Returns the set of added, removed, and type-changed paths.
 */
export function compareShapes(golden: ShapeMap, live: ShapeMap): CanaryDiff {
  const goldenKeys = new Set(Object.keys(golden));
  const liveKeys = new Set(Object.keys(live));

  const added = [...liveKeys].filter(k => !goldenKeys.has(k));
  const removed = [...goldenKeys].filter(k => !liveKeys.has(k));
  const changed = [...goldenKeys]
    .filter(k => liveKeys.has(k) && golden[k] !== live[k]);

  return { added, removed, changed };
}

/**
 * Run a single canary check: extract the live shape, compare against golden,
 * and return a CanaryReport.
 */
export function runCanaryCheck(
  provider: string,
  golden: ShapeMap,
  liveResponse: unknown
): CanaryReport {
  const live = extractShape(liveResponse);
  const diff = compareShapes(golden, live);
  const hasDrift = diff.added.length > 0 || diff.removed.length > 0 || diff.changed.length > 0;

  return {
    provider,
    status: hasDrift ? 'drift' : 'ok',
    diff,
  };
}

function jsonTypeName(value: unknown): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}
