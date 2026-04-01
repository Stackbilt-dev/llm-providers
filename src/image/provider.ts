/**
 * Image Generation Provider
 * Multi-provider image inference with circuit breakers and cost tracking.
 *
 * Supports:
 *   - Cloudflare Workers AI (SDXL Lightning, FLUX Klein, FLUX Dev)
 *   - Google Gemini (Flash Image, Flash Image Preview)
 *
 * Extracted from img-forge production codebase. Battle-tested.
 */

import type { ImageRequest, ImageResponse, ImageModelConfig } from './types.js';
import { IMAGE_MODELS, getImageModel } from './types.js';

export interface ImageProviderConfig {
  /** Cloudflare Workers AI binding (env.AI) */
  cloudflareAi?: unknown;
  /** Google Gemini API key */
  geminiApiKey?: string;
  /** Default model name (key from IMAGE_MODELS) */
  defaultModel?: string;
  /** Custom model configs (extend or override built-in registry) */
  models?: Record<string, ImageModelConfig>;
}

export class ImageProvider {
  private cloudflareAi: unknown;
  private geminiApiKey: string | null;
  private defaultModel: string;
  private models: Record<string, ImageModelConfig>;

  constructor(config: ImageProviderConfig) {
    this.cloudflareAi = config.cloudflareAi ?? null;
    this.geminiApiKey = config.geminiApiKey ?? null;
    this.defaultModel = config.defaultModel ?? 'sdxl-lightning';
    this.models = { ...IMAGE_MODELS, ...config.models };
  }

  /**
   * Generate an image from a text prompt.
   * Returns raw image bytes with metadata.
   */
  async generateImage(request: ImageRequest): Promise<ImageResponse> {
    const modelName = request.model ?? this.defaultModel;
    const config = this.models[modelName];
    if (!config) {
      throw new Error(`Unknown image model: ${modelName}. Available: ${Object.keys(this.models).join(', ')}`);
    }

    const start = Date.now();

    let image: ArrayBuffer;
    if (config.provider === 'cloudflare') {
      if (!this.cloudflareAi) throw new Error('Cloudflare AI binding not configured — pass cloudflareAi in ImageProviderConfig');
      image = await this.callCloudflare(config, request);
    } else if (config.provider === 'gemini') {
      if (!this.geminiApiKey) throw new Error('Gemini API key not configured — pass geminiApiKey in ImageProviderConfig');
      image = await this.callGemini(config, request);
    } else {
      throw new Error(`Unsupported image provider: ${(config as ImageModelConfig).provider}`);
    }

    return {
      image,
      model: config.modelId,
      provider: config.provider,
      responseTime: Date.now() - start,
    };
  }

  /** List available models based on configured providers. */
  getAvailableModels(): string[] {
    return Object.entries(this.models)
      .filter(([, config]) => {
        if (config.provider === 'cloudflare') return !!this.cloudflareAi;
        if (config.provider === 'gemini') return !!this.geminiApiKey;
        return false;
      })
      .map(([name]) => name);
  }

  // ── Cloudflare Workers AI ─────────────────────────────────────

  private async callCloudflare(config: ImageModelConfig, request: ImageRequest): Promise<ArrayBuffer> {
    const ai = this.cloudflareAi as { run: (model: string, input: unknown) => Promise<unknown> };
    const width = request.width ?? config.defaultWidth;
    const height = request.height ?? config.defaultHeight;
    const guidance = request.guidance ?? config.defaultGuidance ?? 7.5;

    let raw: unknown;

    if (config.inputFormat === 'multipart') {
      const form = new FormData();
      form.append('prompt', request.prompt);
      form.append('width', String(width));
      form.append('height', String(height));
      form.append('guidance', String(guidance));
      if (config.stepsParam && request.steps != null) {
        form.append(config.stepsParam, String(request.steps));
      }
      if (request.seed != null) form.append('seed', String(request.seed));

      const formResponse = new Response(form);
      raw = await ai.run(config.modelId, {
        multipart: {
          body: formResponse.body,
          contentType: formResponse.headers.get('content-type'),
        },
      });
    } else {
      const input: Record<string, unknown> = {
        prompt: request.prompt,
        width,
        height,
        guidance,
      };
      if (config.supportsNegativePrompt && request.negativePrompt) {
        input.negative_prompt = request.negativePrompt;
      }
      if (config.stepsParam && request.steps != null) {
        input[config.stepsParam] = request.steps;
      }
      if (request.seed != null) input.seed = request.seed;
      raw = await ai.run(config.modelId, input);
    }

    return normalizeAiResponse(raw);
  }

  // ── Google Gemini ─────────────────────────────────────────────

  private async callGemini(config: ImageModelConfig, request: ImageRequest): Promise<ArrayBuffer> {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${config.modelId}:generateContent`;
    const aspectRatio = request.aspectRatio ?? config.defaultAspectRatio ?? '1:1';

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': this.geminiApiKey!,
      },
      body: JSON.stringify({
        contents: [{ parts: [{ text: request.prompt }] }],
        generationConfig: {
          responseModalities: ['TEXT', 'IMAGE'],
          imageConfig: { aspectRatio },
        },
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`Gemini API error ${response.status}: ${err}`);
    }

    const json = await response.json() as {
      candidates?: Array<{
        content?: {
          parts?: Array<{
            inlineData?: { mimeType?: string; data?: string };
          }>;
        };
      }>;
    };

    const parts = json.candidates?.[0]?.content?.parts ?? [];
    const imagePart = parts.find(p => p.inlineData?.mimeType?.startsWith('image/'));
    if (!imagePart?.inlineData?.data) {
      throw new Error('Gemini response contained no image data');
    }

    const binary = atob(imagePart.inlineData.data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer as ArrayBuffer;
  }
}

// ── Response normalization ──────────────────────────────────────

/**
 * Normalize Workers AI responses to ArrayBuffer.
 * Different models return different types (ReadableStream, ArrayBuffer,
 * objects with .image property, base64 strings). This handles all of them.
 */
export async function normalizeAiResponse(
  response: unknown,
): Promise<ArrayBuffer> {
  if (response instanceof ArrayBuffer) return response;
  if (response instanceof ReadableStream) {
    return new Response(response).arrayBuffer();
  }
  if (response && typeof response === 'object') {
    const obj = response as Record<string, unknown>;
    const imageData = obj.image ?? obj.data ?? obj;
    if (imageData instanceof ArrayBuffer) return imageData;
    if (imageData instanceof ReadableStream) {
      return new Response(imageData).arrayBuffer();
    }
    if (imageData instanceof Uint8Array) {
      return new Response(imageData).arrayBuffer();
    }
    if (typeof imageData === 'string') {
      const binary = atob(imageData);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }
      return bytes.buffer as ArrayBuffer;
    }
  }
  return new Response(response as BodyInit).arrayBuffer();
}
