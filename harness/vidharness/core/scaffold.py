"""提供者脚手架 —— 把"新模型 = 新文件"做成可执行的承诺。

vh scaffold <seam> <name> 生成一个带注册/能力声明/参数目录/协议方法骨架的
提供者模板文件。能力骨架从 registry.SEAM_CAPABILITY_SCHEMAS 生成（新能力
键必须登记 schema 的纪律自动体现在模板里）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .registry import SEAM_CAPABILITY_SCHEMAS

_SEAM_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "generator": {
        "imports": "from ..seams import Artifact, ArtifactMeta, GenRequest",
        "method": (
            "    def generate(self, req: GenRequest, workdir: Path, **kw) -> Artifact:\n"
            '        """一次生成：req.text + 可选 refs/首尾帧 → 视频产物。"""\n'
            "        workdir.mkdir(parents=True, exist_ok=True)\n"
            "        # TODO: 调用模型，把产物写到 workdir/；产物 meta 记录\n"
            "        #   adapter/model/elapsed_s/cost_usd（可重建 + 成本口径）。\n"
            '        raise NotImplementedError("接入新模型：实现 generate")\n'),
    },
    "judge": {
        "imports": "from ..seams import Artifact, ArtifactMeta, spec_to_criteria\n"
                   "from ..consumers.judge_loop import parse_scores",
        "extra": (
            "    modalities = [\"text\"]   # 按实现声明：text / image / video\n"
        ),
        "method": (
            "    def judge(self, media, criteria, workdir, **kw) -> Artifact:\n"
            '        """评分：media(路径列表，可为空=文本评测) + criteria 规格 → 原始分。\n'
            "        协议：返回 payload={\"scores\": {维度: 分数}, \"feedback\": str}，\n"
            "        加权/阈值判定由消费者 run_judge 结算，勿在此计算总分。\"\"\"\n"
            "        workdir.mkdir(parents=True, exist_ok=True)\n"
            "        crits = spec_to_criteria(criteria)\n"
            "        # TODO: 调用模型；输出按 JSON 解析 → parse_scores(raw, crits)\n"
            '        raise NotImplementedError("接入新裁判：实现 judge")\n'),
    },
    "script": {
        "imports": "from ..seams import (Artifact, ArtifactMeta,\n"
                   "                     build_script_prompt, parse_script_json)",
        "method": (
            "    def generate(self, query: str, template: dict, workdir: Path, **kw) -> Artifact:\n"
            '        """剧本生成：query + template(brief/segments/experience) → 分镜计划。\n'
            "        提示契约用 build_script_prompt(query, template)（seam 拥有）。\"\"\"\n"
            "        workdir.mkdir(parents=True, exist_ok=True)\n"
            "        # TODO: 调用模型；输出经 parse_script_json 解析\n"
            '        raise NotImplementedError("接入新剧本生成器：实现 generate")\n'),
    },
    "transcribe": {
        "imports": "from ..seams import Artifact, ArtifactMeta",
        "method": (
            "    def transcribe(self, media: Path, workdir: Path, **kw) -> Artifact:\n"
            '        """转写：媒体文件 → {text, ...}。"""\n'
            "        workdir.mkdir(parents=True, exist_ok=True)\n"
            "        # TODO: 调用 ASR；payload 至少含 text\n"
            '        raise NotImplementedError("接入新转写器：实现 transcribe")\n'),
    },
}


def _class_name(name: str) -> str:
    return "".join(w.capitalize() for w in name.replace("-", "_").split("_")) or "Provider"


def scaffold_provider(seam: str, name: str, out_dir: Path) -> Path:
    """生成提供者骨架文件；返回文件路径。"""
    if seam not in SEAM_CAPABILITY_SCHEMAS:
        raise RuntimeError(
            f"未知 seam '{seam}'（已知: {sorted(SEAM_CAPABILITY_SCHEMAS)}）——"
            f"新 seam 需先登记 registry.SEAM_CAPABILITY_SCHEMAS")
    if not name:
        raise RuntimeError("提供者名不能为空")
    tpl = _SEAM_TEMPLATES.get(seam)
    if tpl is None:
        raise RuntimeError(f"seam '{seam}' 暂无脚手架模板（支持: {sorted(_SEAM_TEMPLATES)}）")
    # 能力骨架（按 schema 生成占位，类型驱动的诚实模板）
    caps = {k: "..." for k in SEAM_CAPABILITY_SCHEMAS[seam]}
    extra = tpl.get("extra", "")
    class_name = _class_name(name)
    body = f'''"""<描述> —— {seam} 缝的新提供者（vh scaffold 生成）。

接入清单（cookbook：docs/cookbook/adding-a-provider.md）：
1. 实现协议方法（下方 TODO）；
2. capabilities/param_schema 按真实能力填写（schema 在 core/registry.py）；
3. providers/__init__.py import 本模块（加载即注册）；
4. 元测试锁定 seam 一致性；真实任务验证；必要时入 vh regress 套件。
"""
from __future__ import annotations

from pathlib import Path

{tpl["imports"]}
from ..core.registry import register


@register("{seam}.{name}")
class {class_name}:
    name = "{seam}.{name}"
    capabilities = {caps!r}
    param_schema = {{
        "model": {{"type": "str", "required": True, "help": "模型名/路径"}},
    }}
{extra}
    def __init__(self, model: str):
        self.model = model

{tpl["method"]}
'''
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{seam}_{name}.py"
    if path.exists():
        raise RuntimeError(f"{path} 已存在（不覆盖，手动处理）")
    path.write_text(body, encoding="utf-8")
    return path
