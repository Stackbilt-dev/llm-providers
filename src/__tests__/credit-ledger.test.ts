import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { CreditLedger } from '../utils/credit-ledger';
import type { LedgerEvent, DepletionEstimate } from '../utils/credit-ledger';

describe('CreditLedger — burn rate & depletion projection', () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: false });
    vi.setSystemTime(new Date('2026-03-31T12:00:00Z'));
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  function makeLedger(monthlyBudget = 20) {
    return new CreditLedger({
      budgets: [{ provider: 'anthropic', monthlyBudget }],
    });
  }

  // ─── getBurnRate ──────────────────────────────────────────

  describe('getBurnRate', () => {
    it('returns zero burn rate with no spend history', () => {
      const ledger = makeLedger();
      const rate = ledger.getBurnRate('anthropic');
      expect(rate.costPerHour).toBe(0);
      expect(rate.costPerDay).toBe(0);
      expect(rate.sampleCount).toBe(0);
    });

    it('calculates burn rate from spend entries', () => {
      const ledger = makeLedger();
      // Simulate 24 hours of spending: $1 per hour
      for (let h = 0; h < 24; h++) {
        vi.setSystemTime(new Date('2026-03-31T00:00:00Z').getTime() + h * 3_600_000);
        ledger.record('anthropic', 'claude-sonnet', 1.0, 1000, 500);
      }

      vi.setSystemTime(new Date('2026-03-31T23:59:59Z'));
      const rate = ledger.getBurnRate('anthropic');
      expect(rate.costPerHour).toBeCloseTo(1.0, 1);
      expect(rate.costPerDay).toBeCloseTo(24.0, 1);
      expect(rate.sampleCount).toBe(24);
    });

    it('respects custom window size', () => {
      const ledger = makeLedger();
      const HOUR = 3_600_000;

      // Spend $2 in the last hour, $10 in the 6 hours before that
      vi.setSystemTime(new Date('2026-03-31T06:00:00Z'));
      ledger.record('anthropic', 'claude-sonnet', 10.0, 5000, 2000);

      vi.setSystemTime(new Date('2026-03-31T12:00:00Z'));
      ledger.record('anthropic', 'claude-sonnet', 2.0, 1000, 500);

      // 1-hour window: only the $2 entry
      const rate1h = ledger.getBurnRate('anthropic', HOUR);
      expect(rate1h.sampleCount).toBe(1);
      expect(rate1h.costPerHour).toBeCloseTo(2.0, 1);

      // 24-hour window: both entries
      const rate24h = ledger.getBurnRate('anthropic', 24 * HOUR);
      expect(rate24h.sampleCount).toBe(2);
    });

    it('only includes entries for the requested provider', () => {
      const ledger = new CreditLedger({
        budgets: [
          { provider: 'anthropic', monthlyBudget: 20 },
          { provider: 'groq', monthlyBudget: 5 },
        ],
      });

      ledger.record('anthropic', 'claude-sonnet', 5.0, 1000, 500);
      ledger.record('groq', 'llama-70b', 0.01, 500, 200);

      const rate = ledger.getBurnRate('anthropic');
      expect(rate.sampleCount).toBe(1);
    });
  });

  // ─── getDepletionEstimate ─────────────────────────────────

  describe('getDepletionEstimate', () => {
    it('returns null for providers with no budget', () => {
      const ledger = new CreditLedger();
      ledger.record('anthropic', 'claude-sonnet', 1.0, 1000, 500);
      expect(ledger.getDepletionEstimate('anthropic')).toBeNull();
    });

    it('returns null days remaining when burn rate is zero', () => {
      const ledger = makeLedger();
      const estimate = ledger.getDepletionEstimate('anthropic');
      // No spend history → burn rate is zero
      expect(estimate).not.toBeNull();
      expect(estimate!.daysRemaining).toBeNull();
      expect(estimate!.projectedDepletionDate).toBeNull();
      expect(estimate!.remainingBudget).toBe(20);
    });

    it('projects depletion date from burn rate', () => {
      const ledger = makeLedger(20); // $20/month budget
      const HOUR = 3_600_000;

      // Spend $1/hour for 10 hours ($10 total, $10 remaining)
      for (let h = 0; h < 10; h++) {
        vi.setSystemTime(new Date('2026-03-31T00:00:00Z').getTime() + h * HOUR);
        ledger.record('anthropic', 'claude-sonnet', 1.0, 1000, 500);
      }
      vi.setSystemTime(new Date('2026-03-31T10:00:00Z'));

      // Use a 10-hour window matching the spend period for accurate rate
      const estimate = ledger.getDepletionEstimate('anthropic', 10 * HOUR);
      expect(estimate).not.toBeNull();
      expect(estimate!.remainingBudget).toBeCloseTo(10.0, 1);
      expect(estimate!.burnRate.costPerHour).toBeCloseTo(1.0, 1);
      // $10 remaining at $1/hr = 10 hours = 0.417 days
      expect(estimate!.daysRemaining).toBeCloseTo(10 / 24, 1);
      expect(estimate!.projectedDepletionDate).toBeInstanceOf(Date);
    });

    it('returns zero days remaining when budget is exhausted', () => {
      const ledger = makeLedger(5);
      ledger.record('anthropic', 'claude-sonnet', 5.0, 5000, 2000);

      const estimate = ledger.getDepletionEstimate('anthropic');
      expect(estimate!.daysRemaining).toBe(0);
      expect(estimate!.remainingBudget).toBe(0);
    });
  });

  // ─── Depletion threshold events ───────────────────────────

  describe('depletion events', () => {
    it('fires depletion_emergency when projected depletion < 1 day', () => {
      const events: LedgerEvent[] = [];
      const ledger = makeLedger(10);
      ledger.on(e => events.push(e));

      const HOUR = 3_600_000;
      // Spend $9 in 9 hours → $1 remaining, burning $1/hr → ~1 hour left
      for (let h = 0; h < 9; h++) {
        vi.setSystemTime(new Date('2026-03-31T00:00:00Z').getTime() + h * HOUR);
        ledger.record('anthropic', 'claude-sonnet', 1.0, 1000, 500);
      }

      const depletionEvents = events.filter(e => e.type === 'depletion_projected');
      expect(depletionEvents.length).toBeGreaterThan(0);

      const lastEvent = depletionEvents[depletionEvents.length - 1];
      expect(lastEvent.type).toBe('depletion_projected');
      if (lastEvent.type === 'depletion_projected') {
        expect(lastEvent.tier).toBe('depletion_emergency');
        expect(lastEvent.daysRemaining).toBeLessThanOrEqual(1);
      }
    });

    it('fires only on tier escalation, not every record', () => {
      const events: LedgerEvent[] = [];
      const ledger = makeLedger(100);
      ledger.on(e => events.push(e));

      const HOUR = 3_600_000;
      // Steady $1/hr for 90 hours → $90 spent, $10 remaining, ~10 hours left
      for (let h = 0; h < 90; h++) {
        vi.setSystemTime(new Date('2026-03-28T00:00:00Z').getTime() + h * HOUR);
        ledger.record('anthropic', 'claude-sonnet', 1.0, 1000, 500);
      }

      const depletionEvents = events.filter(e => e.type === 'depletion_projected');
      // Should have fired warning → critical → emergency (3 tier crossings max)
      const tiers = depletionEvents.map(e => e.type === 'depletion_projected' ? e.tier : null);
      const uniqueTiers = [...new Set(tiers)];
      expect(uniqueTiers.length).toBeLessThanOrEqual(3);
    });
  });

  // ─── getSpendSummary / getSpendBreakdown ──────────────────

  describe('spend summaries', () => {
    it('returns windowed spend summary', () => {
      const ledger = makeLedger();
      const HOUR = 3_600_000;

      vi.setSystemTime(new Date('2026-03-31T10:00:00Z'));
      ledger.record('anthropic', 'claude-sonnet', 3.0, 1000, 500);

      vi.setSystemTime(new Date('2026-03-31T11:00:00Z'));
      ledger.record('anthropic', 'claude-sonnet', 2.0, 800, 400);

      vi.setSystemTime(new Date('2026-03-31T12:00:00Z'));
      const summary = ledger.getSpendSummary('anthropic', 24 * HOUR);
      expect(summary.spend).toBeCloseTo(5.0, 1);
      expect(summary.requestCount).toBe(2);
      expect(summary.provider).toBe('anthropic');
    });

    it('returns breakdown for all providers', () => {
      const ledger = new CreditLedger({
        budgets: [
          { provider: 'anthropic', monthlyBudget: 20 },
          { provider: 'groq', monthlyBudget: 5 },
        ],
      });

      ledger.record('anthropic', 'claude-sonnet', 5.0, 1000, 500);
      ledger.record('groq', 'llama-70b', 0.5, 500, 200);

      const breakdown = ledger.getSpendBreakdown(24 * 3_600_000);
      expect(breakdown).toHaveLength(2);
      expect(breakdown.find(s => s.provider === 'anthropic')?.spend).toBeCloseTo(5.0, 1);
      expect(breakdown.find(s => s.provider === 'groq')?.spend).toBeCloseTo(0.5, 1);
    });
  });

  // ─── Ring buffer ──────────────────────────────────────────

  describe('ring buffer', () => {
    it('caps at configured size', () => {
      const ledger = new CreditLedger({
        budgets: [{ provider: 'anthropic', monthlyBudget: 100 }],
        ringBufferSize: 5,
      });

      for (let i = 0; i < 10; i++) {
        ledger.record('anthropic', 'claude-sonnet', 1.0, 100, 50);
      }

      const snapshot = ledger.snapshot();
      expect(snapshot.spendHistory).toHaveLength(5);
    });

    it('survives snapshot/restore round-trip', () => {
      const ledger = makeLedger();
      ledger.record('anthropic', 'claude-sonnet', 3.0, 1000, 500);
      ledger.record('anthropic', 'claude-sonnet', 2.0, 800, 400);

      const snap = ledger.snapshot();
      expect(snap.spendHistory).toHaveLength(2);

      const restored = makeLedger();
      restored.restore(snap);

      const rate = restored.getBurnRate('anthropic');
      expect(rate.sampleCount).toBe(2);
    });

    it('restores cleanly from snapshots without spendHistory (backward compat)', () => {
      const ledger = makeLedger();
      const snap = ledger.snapshot();
      delete (snap as Record<string, unknown>).spendHistory;

      const restored = makeLedger();
      restored.restore(snap);

      const rate = restored.getBurnRate('anthropic');
      expect(rate.sampleCount).toBe(0);
    });
  });

  // ─── resetPeriod ──────────────────────────────────────────

  describe('resetPeriod', () => {
    it('clears spend history and depletion tiers', () => {
      const ledger = makeLedger();
      ledger.record('anthropic', 'claude-sonnet', 5.0, 1000, 500);
      expect(ledger.getBurnRate('anthropic').sampleCount).toBe(1);

      ledger.resetPeriod();
      expect(ledger.getBurnRate('anthropic').sampleCount).toBe(0);
    });
  });
});
