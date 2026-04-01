/**
 * Credit Ledger — single source of truth for LLM spend across the ecosystem.
 *
 * Tracks per-provider, per-model spend with budgets, rate limits, and
 * threshold events. Persistence-agnostic: holds state in memory, serializes
 * via snapshot()/restore() for any storage backend (D1, KV, R2, etc.).
 */

import type { Logger } from './logger';
import { noopLogger } from './logger';

// ─── Types ──────────────────────────────────────────────────

export type ThresholdTier = 'warning' | 'critical' | 'emergency';
export type DepletionTier = 'depletion_warning' | 'depletion_critical' | 'depletion_emergency';

/** Timestamped spend entry for burn rate calculation. */
export interface SpendEntry {
  timestamp: number; // epoch ms
  provider: string;
  cost: number;
}

/** Burn rate over a rolling window. */
export interface BurnRate {
  costPerHour: number;
  costPerDay: number;
  windowMs: number;
  sampleCount: number;
}

/** Projected depletion estimate for a provider. */
export interface DepletionEstimate {
  remainingBudget: number;
  burnRate: BurnRate;
  projectedDepletionDate: Date | null; // null if burn rate is zero
  daysRemaining: number | null;        // null if burn rate is zero or no budget
}

/** Windowed spend summary for a provider. */
export interface SpendSummary {
  provider: string;
  spend: number;
  requestCount: number;
  inputTokens: number;
  outputTokens: number;
  windowMs: number;
}

export interface RateLimitWindow {
  used: number;
  limit: number;
  windowStart: number; // epoch ms
}

export type RateLimitDimension = 'rpm' | 'rpd' | 'tpm' | 'tpd';

export interface ModelAccumulator {
  spend: number;
  inputTokens: number;
  outputTokens: number;
  requestCount: number;
  lastRecordedAt: number;
}

export interface ProviderAccumulator extends ModelAccumulator {
  models: Record<string, ModelAccumulator>;
  budget: number | null; // null = unlimited
  rateLimits: Partial<Record<RateLimitDimension, RateLimitWindow>>;
}

export interface BudgetConfig {
  provider: string;
  model?: string;
  monthlyBudget?: number;
  rateLimits?: Partial<Record<RateLimitDimension, number>>;
}

export interface ThresholdConfig {
  warning: number;
  critical: number;
  emergency: number;
}

export interface ThresholdEvent {
  type: 'threshold_crossed';
  provider: string;
  model?: string;
  tier: ThresholdTier;
  spend: number;
  budget: number;
  utilizationPct: number;
}

export interface DepletionEvent {
  type: 'depletion_projected';
  provider: string;
  tier: DepletionTier;
  daysRemaining: number;
  projectedDepletionDate: Date;
  burnRate: BurnRate;
}

export type LedgerEvent = ThresholdEvent | DepletionEvent;

export type LedgerListener = (event: LedgerEvent) => void;

export interface RateLimitCheck {
  allowed: boolean;
  used: number;
  limit: number;
}

// ─── Snapshot (versioned, JSON-serializable) ─────────────────

export interface CreditLedgerSnapshot {
  version: 1;
  periodStart: number;
  providers: Record<string, {
    spend: number;
    inputTokens: number;
    outputTokens: number;
    requestCount: number;
    lastRecordedAt: number;
    budget: number | null;
    models: Record<string, ModelAccumulator>;
    rateLimits: Partial<Record<RateLimitDimension, RateLimitWindow>>;
  }>;
  thresholds: ThresholdConfig;
  budgets: BudgetConfig[];
  exportedAt: number;
  /** Timestamped spend entries for burn rate calculation. Optional for backward compat. */
  spendHistory?: SpendEntry[];
}

// ─── Constants ──────────────────────────────────────────────

const DEFAULT_THRESHOLDS: ThresholdConfig = {
  warning: 0.80,
  critical: 0.90,
  emergency: 0.95,
};

const MINUTE_MS = 60_000;
const HOUR_MS = 3_600_000;
const DAY_MS = 86_400_000;

/** Default ring buffer capacity: 2000 entries ≈ a few days of heavy usage. */
const DEFAULT_RING_BUFFER_SIZE = 2000;

/** Default burn rate window: 24 hours. */
const DEFAULT_BURN_RATE_WINDOW_MS = 24 * HOUR_MS;

/** Depletion projection thresholds (days remaining). */
const DEPLETION_THRESHOLDS = {
  depletion_warning: 7,
  depletion_critical: 3,
  depletion_emergency: 1,
} as const;

const DEPLETION_SEVERITY: Record<DepletionTier, number> = {
  depletion_warning: 1,
  depletion_critical: 2,
  depletion_emergency: 3,
};

const WINDOW_DURATIONS: Record<RateLimitDimension, number> = {
  rpm: MINUTE_MS,
  rpd: DAY_MS,
  tpm: MINUTE_MS,
  tpd: DAY_MS,
};

// ─── Helpers ────────────────────────────────────────────────

function createModelAccumulator(): ModelAccumulator {
  return { spend: 0, inputTokens: 0, outputTokens: 0, requestCount: 0, lastRecordedAt: 0 };
}

function createProviderAccumulator(budget: number | null = null): ProviderAccumulator {
  return { ...createModelAccumulator(), models: {}, budget, rateLimits: {} };
}

function evaluateTier(spend: number, budget: number, thresholds: ThresholdConfig): ThresholdTier | null {
  if (budget <= 0) return null;
  const ratio = spend / budget;
  if (ratio >= thresholds.emergency) return 'emergency';
  if (ratio >= thresholds.critical) return 'critical';
  if (ratio >= thresholds.warning) return 'warning';
  return null;
}

const TIER_SEVERITY: Record<ThresholdTier, number> = { warning: 1, critical: 2, emergency: 3 };

// ─── CreditLedger ───────────────────────────────────────────

export class CreditLedger {
  private providers = new Map<string, ProviderAccumulator>();
  private thresholds: ThresholdConfig;
  private budgets: BudgetConfig[] = [];
  private listeners: LedgerListener[] = [];
  private periodStart: number;
  private lastFiredTier = new Map<string, ThresholdTier>(); // provider → last tier fired
  private lastFiredDepletionTier = new Map<string, DepletionTier>();
  private spendHistory: SpendEntry[] = [];
  private ringBufferSize: number;
  private logger: Logger;

  constructor(config?: {
    thresholds?: Partial<ThresholdConfig>;
    budgets?: BudgetConfig[];
    ringBufferSize?: number;
    logger?: Logger;
  }) {
    this.thresholds = { ...DEFAULT_THRESHOLDS, ...config?.thresholds };
    this.periodStart = Date.now();
    this.ringBufferSize = config?.ringBufferSize ?? DEFAULT_RING_BUFFER_SIZE;
    this.logger = config?.logger ?? noopLogger;
    if (config?.budgets) {
      for (const b of config.budgets) this.setBudget(b);
    }
  }

  // ─── Core recording ─────────────────────────────────────

  record(provider: string, model: string, cost: number, inputTokens: number, outputTokens: number): void {
    const now = Date.now();
    const acc = this.getOrCreateProvider(provider);
    const totalTokens = inputTokens + outputTokens;

    // Update provider-level accumulators
    acc.spend += cost;
    acc.inputTokens += inputTokens;
    acc.outputTokens += outputTokens;
    acc.requestCount++;
    acc.lastRecordedAt = now;

    // Update model-level accumulators
    if (!acc.models[model]) acc.models[model] = createModelAccumulator();
    const ma = acc.models[model];
    ma.spend += cost;
    ma.inputTokens += inputTokens;
    ma.outputTokens += outputTokens;
    ma.requestCount++;
    ma.lastRecordedAt = now;

    // Update rate limits (auto-reset stale windows)
    for (const [dim, window] of Object.entries(acc.rateLimits) as Array<[RateLimitDimension, RateLimitWindow]>) {
      const duration = WINDOW_DURATIONS[dim];
      if (now - window.windowStart >= duration) {
        window.used = 0;
        window.windowStart = now;
      }
      if (dim === 'rpm' || dim === 'rpd') {
        window.used++;
      } else {
        window.used += totalTokens;
      }
    }

    // Record spend entry for burn rate calculation
    this.spendHistory.push({ timestamp: now, provider, cost });
    if (this.spendHistory.length > this.ringBufferSize) {
      this.spendHistory.shift();
    }

    // Check budget thresholds
    if (acc.budget !== null && acc.budget > 0) {
      const tier = evaluateTier(acc.spend, acc.budget, this.thresholds);
      if (tier) {
        const lastTier = this.lastFiredTier.get(provider);
        // Only fire if we crossed into a new (higher) tier
        if (!lastTier || TIER_SEVERITY[tier] > TIER_SEVERITY[lastTier]) {
          this.lastFiredTier.set(provider, tier);
          this.emit({
            type: 'threshold_crossed',
            provider,
            model,
            tier,
            spend: acc.spend,
            budget: acc.budget,
            utilizationPct: acc.spend / acc.budget,
          });
        }
      }

      // Check depletion projections
      const estimate = this.getDepletionEstimate(provider);
      if (estimate?.daysRemaining !== null && estimate?.daysRemaining !== undefined) {
        let depletionTier: DepletionTier | null = null;
        if (estimate.daysRemaining <= DEPLETION_THRESHOLDS.depletion_emergency) {
          depletionTier = 'depletion_emergency';
        } else if (estimate.daysRemaining <= DEPLETION_THRESHOLDS.depletion_critical) {
          depletionTier = 'depletion_critical';
        } else if (estimate.daysRemaining <= DEPLETION_THRESHOLDS.depletion_warning) {
          depletionTier = 'depletion_warning';
        }

        if (depletionTier) {
          const lastDep = this.lastFiredDepletionTier.get(provider);
          if (!lastDep || DEPLETION_SEVERITY[depletionTier] > DEPLETION_SEVERITY[lastDep]) {
            this.lastFiredDepletionTier.set(provider, depletionTier);
            this.emit({
              type: 'depletion_projected',
              provider,
              tier: depletionTier,
              daysRemaining: estimate.daysRemaining,
              projectedDepletionDate: estimate.projectedDepletionDate!,
              burnRate: estimate.burnRate,
            });
          }
        }
      }
    }
  }

  // ─── Budget management ──────────────────────────────────

  setBudget(config: BudgetConfig): void {
    // Remove existing budget for this provider/model combo
    this.budgets = this.budgets.filter(b =>
      !(b.provider === config.provider && b.model === config.model)
    );
    this.budgets.push(config);

    const acc = this.getOrCreateProvider(config.provider);

    // Provider-level budget
    if (config.monthlyBudget !== undefined && !config.model) {
      acc.budget = config.monthlyBudget;
    }

    // Rate limits
    if (config.rateLimits) {
      for (const [dim, limit] of Object.entries(config.rateLimits) as Array<[RateLimitDimension, number]>) {
        if (limit === undefined) continue;
        if (!acc.rateLimits[dim]) {
          acc.rateLimits[dim] = { used: 0, limit, windowStart: Date.now() };
        } else {
          acc.rateLimits[dim]!.limit = limit;
        }
      }
    }
  }

  removeBudget(provider: string, model?: string): void {
    this.budgets = this.budgets.filter(b =>
      !(b.provider === provider && b.model === model)
    );
    const acc = this.providers.get(provider);
    if (acc && !model) {
      acc.budget = null;
    }
  }

  // ─── Queries ────────────────────────────────────────────

  remainingBalance(provider: string, model?: string): number | null {
    const acc = this.providers.get(provider);
    if (!acc) return null;

    if (model) {
      // Per-model budget from budgets array
      const modelBudget = this.budgets.find(b => b.provider === provider && b.model === model);
      if (!modelBudget?.monthlyBudget) return null;
      const modelAcc = acc.models[model];
      return modelBudget.monthlyBudget - (modelAcc?.spend ?? 0);
    }

    if (acc.budget === null) return null;
    return acc.budget - acc.spend;
  }

  utilizationPct(provider: string, model?: string): number {
    const acc = this.providers.get(provider);
    if (!acc) return 0;

    if (model) {
      const modelBudget = this.budgets.find(b => b.provider === provider && b.model === model);
      if (!modelBudget?.monthlyBudget || modelBudget.monthlyBudget <= 0) return 0;
      const modelAcc = acc.models[model];
      return (modelAcc?.spend ?? 0) / modelBudget.monthlyBudget;
    }

    if (acc.budget === null || acc.budget <= 0) return 0;
    return acc.spend / acc.budget;
  }

  getProviderAccumulator(provider: string): ProviderAccumulator | undefined {
    return this.providers.get(provider);
  }

  getModelAccumulator(provider: string, model: string): ModelAccumulator | undefined {
    return this.providers.get(provider)?.models[model];
  }

  totalSpend(): number {
    let sum = 0;
    for (const acc of this.providers.values()) sum += acc.spend;
    return sum;
  }

  breakdown(): Record<string, ProviderAccumulator> {
    const result: Record<string, ProviderAccumulator> = {};
    for (const [name, acc] of this.providers) {
      result[name] = { ...acc, models: { ...acc.models }, rateLimits: { ...acc.rateLimits } };
    }
    return result;
  }

  // ─── Burn rate & depletion projection ───────────────────

  /**
   * Calculate burn rate for a provider over a rolling window.
   * Default window: 24 hours.
   */
  getBurnRate(provider: string, windowMs = DEFAULT_BURN_RATE_WINDOW_MS): BurnRate {
    const now = Date.now();
    const cutoff = now - windowMs;
    const entries = this.spendHistory.filter(e => e.provider === provider && e.timestamp >= cutoff);
    const totalCost = entries.reduce((sum, e) => sum + e.cost, 0);
    const windowHours = windowMs / HOUR_MS;
    const windowDays = windowMs / DAY_MS;

    return {
      costPerHour: windowHours > 0 ? totalCost / windowHours : 0,
      costPerDay: windowDays > 0 ? totalCost / windowDays : 0,
      windowMs,
      sampleCount: entries.length,
    };
  }

  /**
   * Project when a provider's budget will be depleted at the current burn rate.
   * Returns null if the provider has no budget or no spend history.
   */
  getDepletionEstimate(provider: string, windowMs?: number): DepletionEstimate | null {
    const acc = this.providers.get(provider);
    if (!acc || acc.budget === null || acc.budget <= 0) return null;

    const remaining = acc.budget - acc.spend;
    if (remaining <= 0) {
      return {
        remainingBudget: 0,
        burnRate: this.getBurnRate(provider, windowMs),
        projectedDepletionDate: new Date(),
        daysRemaining: 0,
      };
    }

    const burnRate = this.getBurnRate(provider, windowMs);
    if (burnRate.costPerDay <= 0) {
      return {
        remainingBudget: remaining,
        burnRate,
        projectedDepletionDate: null,
        daysRemaining: null,
      };
    }

    const daysRemaining = remaining / burnRate.costPerDay;
    const depletionMs = daysRemaining * DAY_MS;
    return {
      remainingBudget: remaining,
      burnRate,
      projectedDepletionDate: new Date(Date.now() + depletionMs),
      daysRemaining,
    };
  }

  /**
   * Get windowed spend summary for a provider.
   */
  getSpendSummary(provider: string, windowMs: number): SpendSummary {
    const now = Date.now();
    const cutoff = now - windowMs;
    const entries = this.spendHistory.filter(e => e.provider === provider && e.timestamp >= cutoff);

    return {
      provider,
      spend: entries.reduce((sum, e) => sum + e.cost, 0),
      requestCount: entries.length,
      inputTokens: 0,  // ring buffer tracks cost, not tokens — use accumulator for totals
      outputTokens: 0,
      windowMs,
    };
  }

  /**
   * Get spend breakdown for all providers over a window.
   */
  getSpendBreakdown(windowMs: number): SpendSummary[] {
    const providers = new Set(this.spendHistory.map(e => e.provider));
    return Array.from(providers).map(p => this.getSpendSummary(p, windowMs));
  }

  // ─── Rate limit tracking ────────────────────────────────

  checkRateLimit(provider: string, dimension: RateLimitDimension): RateLimitCheck {
    const acc = this.providers.get(provider);
    if (!acc) return { allowed: true, used: 0, limit: 0 };

    const window = acc.rateLimits[dimension];
    if (!window) return { allowed: true, used: 0, limit: 0 };

    // Auto-reset stale windows
    const now = Date.now();
    const duration = WINDOW_DURATIONS[dimension];
    if (now - window.windowStart >= duration) {
      window.used = 0;
      window.windowStart = now;
    }

    return {
      allowed: window.used < window.limit,
      used: window.used,
      limit: window.limit,
    };
  }

  // ─── Persistence ────────────────────────────────────────

  snapshot(): CreditLedgerSnapshot {
    const providers: CreditLedgerSnapshot['providers'] = {};
    for (const [name, acc] of this.providers) {
      providers[name] = {
        spend: acc.spend,
        inputTokens: acc.inputTokens,
        outputTokens: acc.outputTokens,
        requestCount: acc.requestCount,
        lastRecordedAt: acc.lastRecordedAt,
        budget: acc.budget,
        models: { ...acc.models },
        rateLimits: { ...acc.rateLimits },
      };
    }

    return {
      version: 1,
      periodStart: this.periodStart,
      providers,
      thresholds: { ...this.thresholds },
      budgets: this.budgets.map(b => ({ ...b })),
      exportedAt: Date.now(),
      spendHistory: this.spendHistory.map(e => ({ ...e })),
    };
  }

  restore(snapshot: CreditLedgerSnapshot): void {
    if (snapshot.version !== 1) {
      this.logger.warn(`[CreditLedger] Unknown snapshot version ${snapshot.version}, skipping restore`);
      return;
    }

    this.periodStart = snapshot.periodStart;
    this.thresholds = { ...DEFAULT_THRESHOLDS, ...snapshot.thresholds };
    this.budgets = snapshot.budgets.map(b => ({ ...b }));
    this.providers.clear();
    this.lastFiredTier.clear();
    this.lastFiredDepletionTier.clear();
    this.spendHistory = (snapshot.spendHistory ?? []).map(e => ({ ...e }));

    for (const [name, data] of Object.entries(snapshot.providers)) {
      const acc = createProviderAccumulator(data.budget);
      acc.spend = data.spend;
      acc.inputTokens = data.inputTokens;
      acc.outputTokens = data.outputTokens;
      acc.requestCount = data.requestCount;
      acc.lastRecordedAt = data.lastRecordedAt;
      acc.models = { ...data.models };
      acc.rateLimits = { ...data.rateLimits };
      this.providers.set(name, acc);

      // Re-evaluate last fired tier so we don't re-fire on next record
      if (acc.budget !== null && acc.budget > 0) {
        const tier = evaluateTier(acc.spend, acc.budget, this.thresholds);
        if (tier) this.lastFiredTier.set(name, tier);
      }
    }
  }

  // ─── Period management ──────────────────────────────────

  resetPeriod(): void {
    this.periodStart = Date.now();
    this.lastFiredTier.clear();
    this.lastFiredDepletionTier.clear();
    this.spendHistory = [];
    for (const acc of this.providers.values()) {
      acc.spend = 0;
      acc.inputTokens = 0;
      acc.outputTokens = 0;
      acc.requestCount = 0;
      acc.lastRecordedAt = 0;
      for (const model of Object.keys(acc.models)) {
        acc.models[model] = createModelAccumulator();
      }
      for (const window of Object.values(acc.rateLimits)) {
        if (window) {
          window.used = 0;
          window.windowStart = Date.now();
        }
      }
    }
  }

  // ─── Event system ───────────────────────────────────────

  on(listener: LedgerListener): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  // ─── Internals ──────────────────────────────────────────

  private getOrCreateProvider(name: string): ProviderAccumulator {
    let acc = this.providers.get(name);
    if (!acc) {
      const budgetConfig = this.budgets.find(b => b.provider === name && !b.model);
      acc = createProviderAccumulator(budgetConfig?.monthlyBudget ?? null);

      // Initialize rate limit windows from budget config
      const rlConfigs = this.budgets.filter(b => b.provider === name && b.rateLimits);
      for (const rl of rlConfigs) {
        if (!rl.rateLimits) continue;
        for (const [dim, limit] of Object.entries(rl.rateLimits) as Array<[RateLimitDimension, number]>) {
          if (limit === undefined) continue;
          acc.rateLimits[dim] = { used: 0, limit, windowStart: Date.now() };
        }
      }

      this.providers.set(name, acc);
    }
    return acc;
  }

  private emit(event: LedgerEvent): void {
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch (err) {
        this.logger.warn('[CreditLedger] Listener error:', err instanceof Error ? err.message : String(err));
      }
    }
  }
}
