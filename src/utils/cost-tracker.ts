/**
 * Cost Tracker
 * Tracks and optimizes LLM usage costs across providers
 */

import type { LLMRequest, LLMResponse, CostConfig, ModelCapabilities } from '../types';
import type { Logger } from './logger';
import { noopLogger } from './logger';
import type { CreditLedger } from './credit-ledger';

export interface ProviderCostEntry {
  totalCost: number;
  requestCount: number;
  inputTokens: number;
  outputTokens: number;
  lastRecordedAt: number;
}

export interface ProviderCostBreakdownEntry extends ProviderCostEntry {
  cost: number;
  requests: number;
  tokens: { input: number; output: number };
  averageCostPerRequest: number;
}

export class CostTracker {
  private providers: Map<string, ProviderCostEntry> = new Map();
  private config: CostConfig;
  private ledger?: CreditLedger;
  private logger: Logger;

  constructor(config: Partial<CostConfig> = {}, ledger?: CreditLedger, logger?: Logger) {
    this.config = {
      inputTokenCost: config.inputTokenCost ?? 0.001, // $0.001 per 1k tokens default
      outputTokenCost: config.outputTokenCost ?? 0.002, // $0.002 per 1k tokens default
      maxMonthlyCost: config.maxMonthlyCost,
      alertThreshold: config.alertThreshold ?? 0.8 // 80% of max cost
    };
    this.ledger = ledger;
    this.logger = logger ?? noopLogger;
  }

  /**
   * Calculate cost for a request
   */
  calculateRequestCost(
    request: LLMRequest,
    capabilities: ModelCapabilities
  ): number {
    const estimatedInputTokens = this.estimateInputTokens(request);
    const estimatedOutputTokens = request.maxTokens || 1000;

    const inputCost = (estimatedInputTokens / 1000) * capabilities.inputTokenCost;
    const outputCost = (estimatedOutputTokens / 1000) * capabilities.outputTokenCost;

    return inputCost + outputCost;
  }

  /**
   * Track actual cost from response
   */
  trackCost(provider: string, response: LLMResponse): void {
    if (!response.usage) return;
    this.record(
      provider,
      response.usage.cost || 0,
      response.usage.inputTokens,
      response.usage.outputTokens,
      response.model || 'unknown',
    );
  }

  /**
   * Record cost directly without requiring a full LLMResponse.
   * This is the simplified API — callers pass provider + cost + tokens.
   */
  record(
    provider: string,
    cost: number,
    inputTokens: number = 0,
    outputTokens: number = 0,
    model: string = 'unknown',
  ): void {
    // Delegate to CreditLedger when present
    if (this.ledger) {
      this.ledger.record(provider, model, cost, inputTokens, outputTokens);
    }

    const entry = this.providers.get(provider) || this.createProviderEntry();
    entry.totalCost += cost;
    entry.requestCount++;
    entry.inputTokens += inputTokens;
    entry.outputTokens += outputTokens;
    entry.lastRecordedAt = Date.now();
    this.providers.set(provider, entry);

    // Check cost alerts
    this.checkCostAlerts(provider);
  }

  /**
   * Get the attached CreditLedger (if any)
   */
  getLedger(): CreditLedger | undefined {
    return this.ledger;
  }

  /**
   * Get total cost across all providers
   */
  getTotalCost(): number {
    return this.total();
  }

  /**
   * Get cost for specific provider
   */
  getProviderCost(provider: string): number {
    return this.providers.get(provider)?.totalCost || 0;
  }

  /**
   * Get cost breakdown by provider
   */
  getCostBreakdown(): Record<string, ProviderCostBreakdownEntry> {
    const breakdown: Record<string, ProviderCostBreakdownEntry> = {};

    for (const [provider, entry] of this.providers) {
      breakdown[provider] = {
        ...entry,
        cost: entry.totalCost,
        requests: entry.requestCount,
        tokens: {
          input: entry.inputTokens,
          output: entry.outputTokens
        },
        averageCostPerRequest: entry.requestCount > 0 ? entry.totalCost / entry.requestCount : 0
      };
    }

    return breakdown;
  }

  /**
   * Get a provider snapshot aligned with Workers usage reporting.
   */
  breakdown(): Record<string, ProviderCostEntry> {
    const snapshot: Record<string, ProviderCostEntry> = {};

    for (const [provider, entry] of this.providers) {
      snapshot[provider] = { ...entry };
    }

    return snapshot;
  }

  /**
   * Get total cost across all providers.
   */
  total(): number {
    let sum = 0;

    for (const entry of this.providers.values()) {
      sum += entry.totalCost;
    }

    return sum;
  }

  /**
   * Get most cost-effective provider for a request
   */
  getMostCostEffectiveProvider(
    providers: Record<string, ModelCapabilities>,
    request: LLMRequest
  ): string {
    let minCost = Infinity;
    let bestProvider = '';

    for (const [provider, capabilities] of Object.entries(providers)) {
      const cost = this.calculateRequestCost(request, capabilities);
      if (cost < minCost) {
        minCost = cost;
        bestProvider = provider;
      }
    }

    return bestProvider;
  }

  /**
   * Check if we're approaching cost limits
   */
  shouldRouteBasedOnCost(provider: string): boolean {
    if (!this.config.maxMonthlyCost) return false;

    const currentCost = this.getProviderCost(provider);
    const threshold = this.config.maxMonthlyCost * (this.config.alertThreshold || 0.8);

    return currentCost >= threshold;
  }

  /**
   * Reset cost tracking (e.g., monthly reset)
   */
  reset(): void {
    this.providers.clear();
  }

  /**
   * Reset tracking for specific provider
   */
  resetProvider(provider: string): void {
    this.providers.delete(provider);
  }

  /**
   * Drain and reset provider tracking for periodic reporting.
   */
  drain(): Record<string, ProviderCostEntry> {
    const snapshot = this.breakdown();
    this.providers.clear();
    return snapshot;
  }

  /**
   * Export cost data for analytics
   */
  exportData(): {
    totalCost: number;
    breakdown: Record<string, ProviderCostBreakdownEntry>;
    period: { start: number; end: number };
  } {
    return {
      totalCost: this.getTotalCost(),
      breakdown: this.getCostBreakdown(),
      period: {
        start: Date.now() - 30 * 24 * 60 * 60 * 1000, // 30 days ago
        end: Date.now()
      }
    };
  }

  /**
   * Estimate input tokens from request
   */
  private estimateInputTokens(request: LLMRequest): number {
    // Rough estimation: 1 token ≈ 0.75 words ≈ 4 characters
    let totalChars = 0;

    // Count characters in messages
    for (const message of request.messages) {
      totalChars += message.content.length;
    }

    // Add system prompt if present
    if (request.systemPrompt) {
      totalChars += request.systemPrompt.length;
    }

    // Convert to tokens (rough approximation)
    return Math.ceil(totalChars / 4);
  }

  /**
   * Check for cost alerts
   */
  private checkCostAlerts(provider: string): void {
    if (!this.config.maxMonthlyCost || !this.config.alertThreshold) return;

    const currentCost = this.getProviderCost(provider);
    const alertThreshold = this.config.maxMonthlyCost * this.config.alertThreshold;

    if (currentCost >= alertThreshold) {
      this.logger.warn(
        `[CostTracker] Cost alert for ${provider}: $${currentCost.toFixed(4)} (${((currentCost / this.config.maxMonthlyCost) * 100).toFixed(1)}% of limit)`
      );
    }
  }

  /**
   * Update cost configuration
   */
  updateConfig(config: Partial<CostConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfig(): CostConfig {
    return { ...this.config };
  }

  private createProviderEntry(): ProviderCostEntry {
    return {
      totalCost: 0,
      requestCount: 0,
      inputTokens: 0,
      outputTokens: 0,
      lastRecordedAt: 0
    };
  }
}

/**
 * Default cost tracker instance
 */
export const defaultCostTracker = new CostTracker();

/**
 * Cost optimization utilities
 */
export class CostOptimizer {
  /**
   * Choose optimal provider based on cost and performance
   */
  static chooseOptimalProvider(
    providers: Record<string, ModelCapabilities>,
    request: LLMRequest,
    costTracker: CostTracker,
    performanceWeights: { cost: number; latency: number; quality: number } = { cost: 0.5, latency: 0.3, quality: 0.2 }
  ): string {
    let bestScore = -Infinity;
    let bestProvider = '';

    for (const [provider, capabilities] of Object.entries(providers)) {
      const cost = costTracker.calculateRequestCost(request, capabilities);
      
      // Normalize cost (lower is better)
      const costScore = 1 / (1 + cost * 1000); // Scale cost for scoring
      
      // Simple quality score based on model capabilities
      const qualityScore = capabilities.maxContextLength / 100000; // Normalize context length
      
      // Assume latency score (would be tracked from actual usage)
      const latencyScore = 0.5; // Placeholder
      
      const totalScore = 
        costScore * performanceWeights.cost +
        latencyScore * performanceWeights.latency +
        qualityScore * performanceWeights.quality;

      if (totalScore > bestScore) {
        bestScore = totalScore;
        bestProvider = provider;
      }
    }

    return bestProvider;
  }

  /**
   * Suggest cost reduction strategies
   */
  static suggestOptimizations(
    costBreakdown: Record<string, ProviderCostBreakdownEntry>
  ): string[] {
    const suggestions: string[] = [];
    const entries = Object.values(costBreakdown);
    const totalCost = entries.reduce((sum, p) => sum + p.cost, 0);

    // Find most expensive provider
    const mostExpensive = Object.entries(costBreakdown)
      .sort(([, a], [, b]) => b.cost - a.cost)[0];

    if (mostExpensive && mostExpensive[1].cost > totalCost * 0.5) {
      suggestions.push(`Consider reducing usage of ${mostExpensive[0]} (${((mostExpensive[1].cost / totalCost) * 100).toFixed(1)}% of total cost)`);
    }

    // Check for high per-request costs
    for (const [provider, data] of Object.entries(costBreakdown)) {
      if (data.averageCostPerRequest > 0.01) { // $0.01 per request
        suggestions.push(`${provider} has high per-request cost ($${data.averageCostPerRequest.toFixed(4)}). Consider using smaller models or reducing max_tokens.`);
      }
    }

    // Suggest batch processing if many small requests
    const totalRequests = entries.reduce((sum, p) => sum + p.requests, 0);
    if (totalRequests > 1000 && totalCost / totalRequests < 0.001) {
      suggestions.push('Consider batch processing for small requests to reduce overhead costs.');
    }

    return suggestions;
  }
}
