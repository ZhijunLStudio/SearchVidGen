/**
 * Node-half bundle: tools run in the host process; the preset mounts this row
 * on the agent plane. @deepseek-ai/* 与 @zhijunlstudio/* 由 DSH profile 树解析。
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
