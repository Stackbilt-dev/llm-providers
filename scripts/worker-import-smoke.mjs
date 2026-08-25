import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const result = await build({
  stdin: {
    contents: "import { LLMProviders } from './dist/index.js'; if (!LLMProviders) throw new Error('missing LLMProviders');",
    resolveDir: root,
    sourcefile: 'worker-import-smoke.mjs',
  },
  bundle: true,
  format: 'esm',
  platform: 'browser',
  conditions: ['worker', 'browser', 'import'],
  target: 'es2022',
  write: false,
  metafile: true,
});

const inputs = Object.keys(result.metafile.inputs);
const unsafe = inputs.filter(path => path.endsWith('.wasm') || path.includes('@stackbilt/wasm-core'));
if (unsafe.length > 0) {
  throw new Error(`Worker import graph contains eager WASM dependencies: ${unsafe.join(', ')}`);
}
if (result.outputFiles.length !== 1 || result.outputFiles[0].contents.length === 0) {
  throw new Error('Worker bundle was not produced');
}

console.log('worker import smoke test passed');
