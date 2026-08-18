/**
 * Node-half bundle: the SD package runs in the host process only.
 * @deepseek-ai/* peers resolve from the DSH profile tree at runtime.
 */
import { defineConfig } from 'tsdown'

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm'],
  platform: 'node',
  outDir: 'lib',
  clean: true,
  dts: false,
  external: [/^@deepseek-ai\//, /^node:/],
})