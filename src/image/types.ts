/**
 * Image Generation Types
 * Unified types for multi-provider image generation.
 */

export type ImageModelInputFormat = 'json' | 'multipart';

export interface ImageModelConfig {
  provider: 'cloudflare' | 'gemini';
  modelId: string;
  inputFormat?: ImageModelInputFormat;  // cloudflare only
  supportsNegativePrompt: boolean;
  supportsSeed: boolean;
  stepsParam?: 'num_steps' | 'steps' | null;
  defaultSteps?: number | null;
  defaultGuidance?: number;
  defaultAspectRatio?: string;          // gemini only
  defaultWidth: number;
  defaultHeight: number;
  maxWidth: number;
  maxHeight: number;
}

export interface ImageRequest {
  prompt: string;
  negativePrompt?: string | null;
  model?: string;
  width?: number;
  height?: number;
  steps?: number | null;
  guidance?: number;
  seed?: number | null;
  aspectRatio?: string;
}

export interface ImageResponse {
  image: ArrayBuffer;
  model: string;
  provider: string;
  responseTime: number;
  cost?: number;
}

/**
 * Built-in image model registry.
 * Quality tiers map to specific provider + model combinations.
 */
export const IMAGE_MODELS: Record<string, ImageModelConfig> = {
  // Cloudflare Workers AI — fast, included in free tier
  'sdxl-lightning': {
    provider: 'cloudflare',
    modelId: '@cf/bytedance/stable-diffusion-xl-lightning',
    inputFormat: 'json',
    supportsNegativePrompt: true,
    supportsSeed: true,
    stepsParam: 'num_steps',
    defaultSteps: 4,
    defaultGuidance: 7.5,
    defaultWidth: 1024,
    defaultHeight: 1024,
    maxWidth: 2048,
    maxHeight: 2048,
  },
  // Cloudflare Workers AI — balanced quality/speed
  'flux-klein': {
    provider: 'cloudflare',
    modelId: '@cf/black-forest-labs/flux-2-klein-4b',
    inputFormat: 'multipart',
    supportsNegativePrompt: false,
    supportsSeed: true,
    stepsParam: null,
    defaultSteps: null,
    defaultGuidance: 7.5,
    defaultWidth: 1024,
    defaultHeight: 768,
    maxWidth: 1920,
    maxHeight: 1920,
  },
  // Cloudflare Workers AI — highest CF quality
  'flux-dev': {
    provider: 'cloudflare',
    modelId: '@cf/black-forest-labs/flux-2-dev',
    inputFormat: 'multipart',
    supportsNegativePrompt: false,
    supportsSeed: true,
    stepsParam: 'steps',
    defaultSteps: 25,
    defaultGuidance: 7.5,
    defaultWidth: 1024,
    defaultHeight: 768,
    maxWidth: 1920,
    maxHeight: 1920,
  },
  // Google Gemini — text rendering capable
  'gemini-flash-image': {
    provider: 'gemini',
    modelId: 'gemini-2.5-flash-image',
    supportsNegativePrompt: false,
    supportsSeed: false,
    defaultAspectRatio: '1:1',
    defaultWidth: 1024,
    defaultHeight: 1024,
    maxWidth: 2048,
    maxHeight: 2048,
  },
  // Google Gemini — latest preview, best text rendering
  'gemini-flash-image-preview': {
    provider: 'gemini',
    modelId: 'gemini-3.1-flash-image-preview',
    supportsNegativePrompt: false,
    supportsSeed: false,
    defaultAspectRatio: '1:1',
    defaultWidth: 1024,
    defaultHeight: 1024,
    maxWidth: 2048,
    maxHeight: 2048,
  },
};

export function getImageModel(name: string): ImageModelConfig | undefined {
  return IMAGE_MODELS[name];
}
