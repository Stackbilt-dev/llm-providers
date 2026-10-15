import { LLMErrorFactory, SchemaDriftError, TimeoutError } from '../errors.js';
import type { TokenUsage } from '../types.js';
import { attachStreamUsage, createStreamUsageTracker } from './stream-usage.js';

interface StreamEvent { data: string; event: string }

/** Incremental UTF-8 SSE framing; handles CR, LF, CRLF, comments and multiline data. */
export class SSEDecoder {
  private decoder = new TextDecoder();
  private line = '';
  private data: string[] = [];
  private event = '';
  private afterCR = false;

  push(bytes: Uint8Array, eof = false): StreamEvent[] {
    const events: StreamEvent[] = [];
    const dispatch = () => {
      if (this.data.length) events.push({ data: this.data.join('\n'), event: this.event });
      this.data = [];
      this.event = '';
    };
    const line = () => {
      const text = this.line;
      this.line = '';
      if (!text) { dispatch(); return; }
      if (text.startsWith(':')) return;
      const colon = text.indexOf(':');
      const field = colon < 0 ? text : text.slice(0, colon);
      const value = colon < 0 ? '' : text.slice(colon + 1).replace(/^ /, '');
      if (field === 'data') this.data.push(value);
      if (field === 'event') this.event = value;
    };
    const text = this.decoder.decode(bytes, { stream: !eof });
    for (const char of text) {
      if (this.afterCR && char === '\n') { this.afterCR = false; continue; }
      this.afterCR = char === '\r';
      if (char === '\r' || char === '\n') line();
      else this.line += char;
    }
    if (eof) { if (this.line) line(); dispatch(); }
    return events;
  }
}

export interface ProviderStreamOptions {
  provider: string;
  open(signal: AbortSignal): Promise<ReadableStream<unknown> | unknown>;
  event(value: unknown): { text?: string; done?: boolean };
  usage(): TokenUsage | undefined;
  /** Deadline for opening the stream (headers / binding return); cleared once the source is available. */
  timeoutMs?: number;
  /** Opt-in total deadline for the whole stream, including the body. Unset means no body deadline. */
  bodyTimeoutMs?: number;
  onError?(error: unknown): void;
  onFinish?(outcome: 'success' | 'error' | 'cancelled', error?: unknown): void;
  mapError?(error: unknown): unknown;
}

/** One outstanding read, no output accumulation, and usage settlement on every exit. */
export function createProviderStream(options: ProviderStreamOptions): ReadableStream<string> {
  const usage = createStreamUsageTracker();
  const decoder = new SSEDecoder();
  const abort = new AbortController();
  let reader: ReadableStreamDefaultReader<unknown> | undefined;
  let opening: Promise<void> | undefined;
  let events: StreamEvent[] = [];
  let eof = false;
  let terminal = false;
  let controller: ReadableStreamDefaultController<string>;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let bodyTimer: ReturnType<typeof setTimeout> | undefined;

  const cancelReader = (reason?: unknown) => {
    const active = reader;
    if (!active) return;
    // Settlement must not hang on an upstream cancel callback.
    void active.cancel(reason).catch(() => {}).finally(() => {
      try { active.releaseLock(); } catch { /* A read may still be pending. */ }
    });
  };
  const finish = (error?: unknown, cancelled = false) => {
    if (terminal) return;
    terminal = true;
    abort.abort(error);
    if (timer !== undefined) clearTimeout(timer);
    if (bodyTimer !== undefined) clearTimeout(bodyTimer);
    try { usage.resolve(options.usage()); }
    catch { usage.resolve(undefined); }
    try { options.onFinish?.(cancelled ? 'cancelled' : error === undefined ? 'success' : 'error', error); }
    catch { /* Statistics must not change the delivered stream. */ }
    events = [];
    cancelReader(error);
    if (cancelled) return;
    if (error !== undefined) {
      try { options.onError?.(error); } catch { /* Telemetry cannot change transport behavior. */ }
      controller.error(options.mapError?.(error) ?? error);
    } else controller.close();
  };

  const stream = new ReadableStream<string>({
    start(value) {
      controller = value;
      if (options.timeoutMs && options.timeoutMs > 0) {
        timer = setTimeout(() => finish(new TimeoutError(options.provider,
          `Stream open timeout after ${options.timeoutMs}ms`)), options.timeoutMs);
      }
      if (options.bodyTimeoutMs && options.bodyTimeoutMs > 0) {
        bodyTimer = setTimeout(() => finish(new TimeoutError(options.provider,
          `Stream timeout after ${options.bodyTimeoutMs}ms`)), options.bodyTimeoutMs);
      }
    },
    async pull() {
      try {
        if (!opening) opening = (async () => {
          const source = await options.open(abort.signal);
          // The provider timeout covers opening only; long generations may keep streaming.
          if (timer !== undefined) { clearTimeout(timer); timer = undefined; }
          if (source && typeof source === 'object' && 'getReader' in source && typeof source.getReader === 'function') {
            reader = (source as ReadableStream<unknown>).getReader();
            if (terminal) cancelReader();
          } else if (!terminal) {
            const result = options.event(source);
            if (result.text) controller.enqueue(result.text);
            finish();
          }
        })();
        await opening;
        while (!terminal) {
          if (events.length) {
            const frame = events.shift()!;
            if (!frame.data.trim()) continue;
            if (frame.data.trim() === '[DONE]') { finish(); return; }
            let value: unknown;
            try { value = JSON.parse(frame.data); }
            catch { throw new SchemaDriftError(options.provider, 'sse.chunk', 'valid-json', 'malformed-json'); }
            if (frame.event === 'error') {
              const body = value && typeof value === 'object' ? value : {};
              throw LLMErrorFactory.fromHttpResponse(options.provider, 500, body);
            }
            const result = options.event(value);
            if (result.text) controller.enqueue(result.text);
            if (result.done) finish();
            if (result.text || result.done) return;
            continue;
          }
          if (eof) { finish(); return; }
          const chunk = await reader!.read();
          if (terminal) return;
          if (chunk.done) {
            eof = true;
            events = decoder.push(new Uint8Array(), true);
          } else if (chunk.value instanceof Uint8Array) {
            events = decoder.push(chunk.value);
          } else {
            const result = options.event(chunk.value);
            if (result.text) controller.enqueue(result.text);
            if (result.done) finish();
            if (result.text || result.done) return;
          }
        }
      } catch (error) { finish(error); }
    },
    cancel(reason) { finish(reason, true); },
  }, { highWaterMark: 0 });
  return attachStreamUsage(stream, usage.promise);
}

/** Shared Chat Completions delta contract for OpenAI-compatible transports. */
export function readChatStreamEvent(provider: string, value: unknown): { text?: string } {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
  const chunk = value as Record<string, unknown>;
  if (chunk.error || chunk.type === 'error') {
    throw LLMErrorFactory.fromHttpResponse(provider, 500, chunk);
  }
  const choices = chunk.choices;
  if (!Array.isArray(choices) || choices.length === 0) return {};
  const first = choices[0];
  if (!first || typeof first !== 'object') return {};
  const delta = first.delta;
  if (!delta || typeof delta !== 'object' || Array.isArray(delta)) return {};
  const content = delta.content;
  if (content === undefined || content === null) return {};
  if (typeof content !== 'string') {
    throw new SchemaDriftError(provider, 'sse.choices[0].delta.content', 'string', typeof content);
  }
  return { text: content };
}
