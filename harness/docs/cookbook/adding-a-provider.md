# Cookbook：接入新模型（新模型 = 新文件）

VidHarness 的核心承诺："换下一个全模态模型 = 新适配器文件，核心零改动"。
本指南给出完整清单，每一步都有对应的验证手段。

## 1. 生成骨架

```bash
vh scaffold generator my-model          # 或 judge / script / transcribe
# → vidharness/providers/generator_my-model.py
```

骨架包含：@register 注册、按 seam 能力 schema 生成的 capabilities 占位、
param_schema 示例、协议方法 docstring（含本缝的契约说明）。

## 2. 实现协议方法（各缝契约）

| 缝 | 协议方法 | 关键契约 |
|---|---|---|
| generator | generate(req, workdir) | 返回 Artifact(kind='video')；meta 记 adapter/model/elapsed_s/cost_usd；backend 能力声明决定成本口径 |
| judge | judge(media, criteria, workdir) | payload 只含原始 {scores, feedback}——加权/阈值由消费者结算；modalities 诚实声明 |
| script | generate(query, template, workdir) | 提示契约用 build_script_prompt；输出经 parse_script_json |
| transcribe | transcribe(media, workdir) | payload 至少含 text |

## 3. 注册与能力

- capabilities 只允许本缝 schema 中的键（core/registry.py）；新能力键
  = 协议演进，先登记 schema（注册点校验会拦住自由键）。
- param_schema 声明参数类型/choices/必需/help——`vh adapters --verbose`
  即自助文档；元测试锁定"声明 == 构造签名"。

## 4. 装配（加载即注册）

providers/__init__.py 加 `import`（幂等）；`vh adapters` 确认出现。
任务 YAML 引用：`generator: {adapter: generator.my-model, params: {...}}`。
多后端可用 `generator.fallback`（降级链）或 `generator.route`（按能力路由）。

## 5. 验证清单（全部可机械执行）

1. `python -m pytest tests/ -q`——seam 一致性元测试会检查协议成员
   （name/方法/capabilities/backend 声明）；
2. `vh run tasks/story_smoke.yaml`（或最小变体）——真实端到端；
3. 评测型提供者：跑 `scripts/calibrate_judges.py` 取得与主裁判的
   口径偏移（混用裁判的 leaderboard 会自动标注警告）；
4. 生成型提供者：加一个 check 任务进 `tasks/regression.yaml`（配置
   漂移检测会让"已回归过"不再变假象）。

## 6. 纪律提醒（本仓库的血泪教训）

- **fail loud**：无 keyframe 的 fl2va 在最早点报错（E20）；GPU 不足
  预检报错（E29）；参数拼错在 instantiate 报错（E11）。
- **不静默降级**：能力不满足 = 报错，不是悄悄换行为（E11 Bug#3）。
- **模型可见 ⟺ 日志**：把完整输入记进 meta.params（E16 的可重建性）。
- **孪生思维**：如果你的实现无法用现有 seam 词汇表达，先怀疑 seam 缺口
  （E15/E20 都是第二个实现暴露的）。
