// https://developers.cloudflare.com/workers-ai/features/json-mode/#supported-models
// GLM schemas: https://developers.cloudflare.com/workers-ai/models/glm-5.3/
// and https://developers.cloudflare.com/workers-ai/models/glm-5.3-flash/
const formats: Record<string, 'raw' | 'wrapped'> = {
  '@cf/meta/llama-3.3-70b-instruct-fp8-fast': 'raw',
  '@cf/meta/llama-3-8b-instruct': 'raw',
  '@cf/meta/llama-3.1-8b-instruct': 'raw',
  '@hf/nousresearch/hermes-2-pro-mistral-7b': 'raw',
  '@hf/thebloke/deepseek-coder-6.7b-instruct-awq': 'raw',
  '@cf/deepseek-ai/deepseek-r1-distill-qwen-32b': 'raw',
  '@cf/zai-org/glm-5.3': 'wrapped',
  '@cf/zai-org/glm-5.3-flash': 'wrapped',
};

export function cloudflareStructuredOutputFormat(model: string): 'raw' | 'wrapped' | undefined {
  return formats[model];
}
