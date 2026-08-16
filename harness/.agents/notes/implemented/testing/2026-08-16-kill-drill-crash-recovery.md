# Agent Note: kill -9 故障演练——崩溃恢复实证与僵尸进程预检

Status: implemented

## Problem

崩溃恢复只有单元测试（manifest 删除重放），从未对真实 SIGKILL 验证过。
E29 演练：对真实生成中的 run 执行 kill -9，验证事件流完整性 → 续跑恢复。

演练发现两个事实：
1. **事件溯源按设计工作**：kill 后 events.jsonl 12 条完好（最后一条 =
   未闭合的 stage.started segments），投影一致（0 产物、无 finished_at）。
2. **kill -9 父进程遗留僵尸子进程占住显存**：diffusers 的子进程在父进程
   被杀后幸存，占住 62.7GB；第一次续跑在 torch 深处 OOM，报错完全不可读
   （"GPU 0 has 79.25 GiB…Process 3665889 has 62.69 GiB"）。

## Decision

1. **加载前 GPU 显存预检**（`check_gpu_free`，minimax_h3 适配器内）：
   按 gpu_spec 逐卡检查 nvidia-smi 空闲显存，低于 40GB 阈值即响亮失败，
   指引"pkill -f vidharness 清理僵尸进程"。OOM 从 torch 深处的不可读
   报错变成加载前的可操作报错。nvidia-smi 不可用时跳过（非 GPU 环境）。
2. **RUNBOOK 运维指引**：中断运行的规范姿势 = `pkill -f vidharness`
   （杀进程组含子进程），恢复前 nvidia-smi 确认显存释放。
3. 演练收尾：清僵尸 → 二次续跑 → 剧本/段1 缓存命中、段2 重生成 →
   finalize → 不变量通过（43 条事件、finalized=1、阶段配对集合闭合）。

## Alternatives considered

- **在 harness 里接管 SIGKILL**：否决。SIGKILL 不可捕获；子进程归属是
  diffusers 多进程行为的产物，harness 只能做预检与指引。
- **阈值写死 40GB**：否决。做成参数（min_free_gb）并默认 40GB——
  H3 双卡单侧需 ~38GB 可用，40GB 是部署常量（允许配置覆盖）。
- **恢复前自动 pkill**：否决。harness 不得擅自杀进程（可能是别人的
  任务）；报错 + 指引是对的边界。

## Consequences

- 崩溃恢复承诺获得真实 SIGKILL 实证（E29）；"断点续跑"在进程级崩溃后
  确实成立。
- 僵尸进程占卡从此有响亮且可操作的报错；RUNBOOK 记录规范终止姿势。
- 演练暴露的管道掩码问题（`| tail` 吞掉 exit code）成为脚本纪律：
  后台运行命令用 `> log 2>&1; echo EXIT=$?` 而非管道 tail。
