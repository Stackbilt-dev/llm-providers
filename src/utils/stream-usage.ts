import type { TokenUsage } from '../types.js';

const STREAM_USAGE = Symbol.for('@stackbilt/llm-providers.streamUsage');

export type UsageTrackedStream = ReadableStream<string> & {
  [STREAM_USAGE]?: Promise<TokenUsage | undefined>;
};

export function createStreamUsageTracker(): {
  promise: Promise<TokenUsage | undefined>;
  resolve: (usage: TokenUsage | undefined) => void;
} {
  let settled = false;
  let resolvePromise: (usage: TokenUsage | undefined) => void = () => {};
  const promise = new Promise<TokenUsage | undefined>((resolve) => {
    resolvePromise = resolve;
  });

  return {
    promise,
    resolve: (usage) => {
      if (settled) return;
      settled = true;
      resolvePromise(usage);
    },
  };
}

export function attachStreamUsage(
  stream: ReadableStream<string>,
  usage: Promise<TokenUsage | undefined>
): UsageTrackedStream {
  Object.defineProperty(stream, STREAM_USAGE, {
    value: usage,
  });

  return stream as UsageTrackedStream;
}

export function getStreamUsage(stream: ReadableStream<string>): Promise<TokenUsage | undefined> | undefined {
  return (stream as UsageTrackedStream)[STREAM_USAGE];
}
