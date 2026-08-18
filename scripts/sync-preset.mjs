#!/usr/bin/env node
/**
 * 把 presets/video「视频生成模式」同步进 harness-home agent-presets 发现根
 * （~/.dsh/.agent-presets/video），GUI 新建会话的预设选择器即可选。
 * 幂等：内容变化才写入（升级插件时自动更新，与 dsh-liangshen 的 sync 语义一致）。
 *
 * 用法: node scripts/sync-preset.mjs [--home <harness-home>]
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..')
const sourceDir = join(repoRoot, 'presets', 'video')

const args = process.argv.slice(2)
const homeFlag = args.indexOf('--home')
const home = homeFlag >= 0 ? args[homeFlag + 1] : join(homedir(), '.dsh')
const targetDir = join(home, '.agent-presets', 'video')

const PRESET_FILES = ['agent.cordis.yml', 'preset.yml']

function sync() {
  mkdirSync(targetDir, { recursive: true })
  let changed = 0
  for (const name of PRESET_FILES) {
    const source = join(sourceDir, name)
    const target = join(targetDir, name)
    const content = readFileSync(source, 'utf-8')
    if (existsSync(target) && readFileSync(target, 'utf-8') === content) continue
    writeFileSync(target, content, 'utf-8')
    changed++
  }
  console.log(
    changed > 0
      ? '已同步视频生成模式 preset 到 ' + targetDir + '（更新 ' + changed + ' 个文件）'
      : 'preset 已是最新: ' + targetDir,
  )
  console.log('新建会话时在预设选择器里选「视频生成模式」即可使用。')
}

sync()
