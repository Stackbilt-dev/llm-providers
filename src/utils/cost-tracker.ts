/**
 * Cost Tracker
 * Tracks and optimizes LLM usage costs across providers
 */

import type { LLMRequest, LLMResponse, TokenUsage, CostConfig, ModelCapabilities } from '../types';

export class CostTracker {
  private costs: Map<string, number> = new Map(); // provider -> total cost
  private requests: Map<string, number> = new Map(); // provider -> request count
  private tokens: Map<string, { input: number; output: number }> = new Map();
  private config: CostConfig;

  constructor(config: Partial<CostConfig> = {}) {
    this.config = {
      inputTokenCost: config.inputTokenCost ?? 0.001, // $0.001 per 1k tokens default
      outputTokenCost: config.outputTokenCost ?? 0.002, // $0.002 per 1k tokens default
      maxMonthlyCost: config.maxMonthlyCost,
      alertThreshold: config.alertThreshold ?? 0.8 // 80% of max cost
    };
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
    if (!response.usage?.totalTokens) return;

    const providerCosts = this.costs.get(provider) || 0;
    const cost = response.usage.cost || 0;
    
    this.costs.set(provider, providerCosts + cost);
    
    // Track requests
    const requestCount = this.requests.get(provider) || 0;
    this.requests.set(provider, requestCount + 1);

    // Track tokens
    const tokenUsage = this.tokens.get(provider) || { input: 0, output: 0 };
    tokenUsage.input += response.usage.inputTokens;
    tokenUsage.output += response.usage.outputTokens;
    this.tokens.set(provider, tokenUsage);

    // Check cost alerts
    this.checkCostAlerts(provider);
  }

  /**
   * Get total cost across all providers
   */
  getTotalCost(): number {
    return Array.from(this.costs.values()).reduce((sum, cost) => sum + cost, 0);
  }

  /**
   * Get cost for specific provider
   */
  getProviderCost(provider: string): number {
    return this.costs.get(provider) || 0;
  }

  /**
   * Get cost breakdown by provider
   */
  getCostBreakdown(): Record<string, {
    cost: number;
    requests: number;
    tokens: { input: number; output: number };
    averageCostPerRequest: number;
  }> {
    const breakdown: Record<string, any> = {};

    for (const [provider, cost] of this.costs) {
      const requests = this.requests.get(provider) || 0;
      const tokens = this.tokens.get(provider) || { input: 0, output: 0 };

      breakdown[provider] = {
        cost,
        requests,
        tokens,
        averageCostPerRequest: requests > 0 ? cost / requests : 0
      };
    }

    return breakdown;
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
    this.costs.clear();
    this.requests.clear();
    this.tokens.clear();
  }

  /**
   * Reset tracking for specific provider
   */
  resetProvider(provider: string): void {
    this.costs.delete(provider);
    this.requests.delete(provider);
    this.tokens.delete(provider);
  }

  /**
   * Export cost data for analytics
   */
  exportData(): {
    totalCost: number;
    breakdown: Record<string, any>;
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
      console.warn(
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
    costBreakdown: Record<string, any>
  ): string[] {
    const suggestions: string[] = [];
    const totalCost = Object.values(costBreakdown).reduce((sum: number, p: any) => sum + p.cost, 0);

    // Find most expensive provider
    const mostExpensive = Object.entries(costBreakdown)
      .sort(([,a]: [string, any], [,b]: [string, any]) => b.cost - a.cost)[0];

    if (mostExpensive && mostExpensive[1].cost > totalCost * 0.5) {
      suggestions.push(`Consider reducing usage of ${mostExpensive[0]} (${((mostExpensive[1].cost / totalCost) * 100).toFixed(1)}% of total cost)`);
    }

    // Check for high per-request costs
    for (const [provider, data] of Object.entries(costBreakdown)) {
      if ((data as any).averageCostPerRequest > 0.01) { // $0.01 per request
        suggestions.push(`${provider} has high per-request cost ($${((data as any).averageCostPerRequest).toFixed(4)}). Consider using smaller models or reducing max_tokens.`);
      }
    }

    // Suggest batch processing if many small requests
    const totalRequests = Object.values(costBreakdown).reduce((sum: number, p: any) => sum + p.requests, 0);
    if (totalRequests > 1000 && totalCost / totalRequests < 0.001) {
      suggestions.push('Consider batch processing for small requests to reduce overhead costs.');
    }

    return suggestions;
  }
}