/**
 * Node-half bundle: providers run in the host process only.
 * @deepseek-ai/* peers resolve from the DSH profile tree at runtime;
 * @zhijunlstudio/dsh-video resolves from the profile's own node_modules.
 */
import { defineConfig } from 'tsdown'

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm'],
  platform: 'node',
  outDir: 'lib',
  clean: true,
  dts: false,
  external: [/^@deepseek-ai\//, /^@zhijunlstudio\//, /^node:/, /^schemastery$/],
})
