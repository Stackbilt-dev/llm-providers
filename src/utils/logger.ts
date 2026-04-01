/**
 * Logger interface for structured, controllable logging.
 *
 * Libraries should not spray console.* into consuming applications.
 * Default is noopLogger (silent). Users opt in by passing consoleLogger
 * or their own implementation via config.
 */

export interface Logger {
  debug(msg: string, ...args: unknown[]): void;
  info(msg: string, ...args: unknown[]): void;
  warn(msg: string, ...args: unknown[]): void;
  error(msg: string, ...args: unknown[]): void;
}

/** Silent logger — default for all components. */
export const noopLogger: Logger = {
  debug() {},
  info() {},
  warn() {},
  error() {},
};

/** Forwards to console.* — opt-in for development/debugging. */
export const consoleLogger: Logger = {
  debug: (...args: unknown[]) => console.debug(...args),
  info: (...args: unknown[]) => console.log(...args),
  warn: (...args: unknown[]) => console.warn(...args),
  error: (...args: unknown[]) => console.error(...args),
};
