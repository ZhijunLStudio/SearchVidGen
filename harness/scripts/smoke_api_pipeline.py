"""API 后端的整条流水线冒烟（mock MiniMax API + 真实剧本/裁判）。

验证"API 对比实验一键可跑"承诺的最后一块：director 全流程在 API 后端上
走通（生成→评测→跨段→总装→finalize 不变量），仅真实端点未接入。

用法（需 vLLM 裁判在 :8030、DEEPSEEK_API_KEY）：
  python scripts/smoke_api_pipeline.py --query "雨夜，一只小猫在旧书店的橱窗前躲雨"
输出：实验目录 + 关键分 + doctor 结论。
"""
import argparse
import http.server
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidharness.core.registry import load_builtin_adapters, instantiate  # noqa: E402
from vidharness.core.experiment import Experiment  # noqa: E402
from vidharness.consumers.segment_director import SegmentDirector  # noqa: E402
from vidharness.core.invariants import check_experiment  # noqa: E402


def _mock_server(tmp: Path):
    tmp.mkdir(parents=True, exist_ok=True)
    vid = tmp / "mock.mp4"
    try:
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                        "color=c=blue:s=128x128:d=2", "-c:v", "libx264",
                        str(vid)], capture_output=True, check=True)
    except Exception:
        vid.write_bytes(b"fake-mp4")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            if self.path.startswith("/v1/files/upload"):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"file": {"url": "http://127.0.0.1:%d/vid.mp4"}}'
                                 % self.server.server_port)
            elif self.path == "/v2/video_generation":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"task_id": "mock-task-1"}')
            else:
                self.send_response(404)
                self.end_headers()

        def do_GET(self):
            if self.path == "/v2/query/video_generation?task_id=mock-task-1":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    b'{"task": {"status": "succeeded", "content": {"url": '
                    b'"http://127.0.0.1:%d/vid.mp4"}}}' % self.server.server_port)
            elif self.path == "/vid.mp4":
                data = vid.read_bytes()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="雨夜，一只小猫在旧书店的橱窗前躲雨")
    ap.add_argument("--workdir", default="/tmp/vh-api-pipeline")
    args = ap.parse_args()

    load_builtin_adapters()
    base = Path(args.workdir)
    shutil.rmtree(base, ignore_errors=True)
    server = _mock_server(base)
    try:
        cfg = {
            "task_name": "api_pipeline_smoke",
            "segments": 2,
            "pipeline": {
                "script": {"adapter": "script.deepseek-v4-flash",
                           "params": {"model": "deepseek-chat", "temperature": 0.9}},
                "generator": {"adapter": "generator.minimax-h3-api",
                              "params": {"api_key": "k",
                                         "base_url": f"http://127.0.0.1:{server.server_port}",
                                         "resolution": "768P", "duration": 5}},
                "context": {"chain_mode": "none", "anchor_refs": []},
            },
            "judge": {
                "adapter": "judge.openai-compat",
                "params": {"base_url": "http://127.0.0.1:8030/v1",
                           "model": "judge-qwen3.5-27b",
                           "frame_samples": 2, "disable_thinking": True},
                "stages": {"script_judge": {"adapter": "judge.deepseek-text",
                                            "params": {"model": "deepseek-chat"}}},
            },
            "script_judge": [
                {"name": "叙事完整", "question": "分镜是否构成完整有起伏的故事？",
                 "weight": 1.0, "min_score": 6},
            ],
            "segment_judge": [
                {"name": "与指令一致性", "question": "视频是否准确呈现指令主体？",
                 "weight": 1.0, "min_score": 6},
            ],
            "segment_retry": {"max_attempts": 1, "inject_feedback": False},
            "cross_judge": [
                {"name": "跨段一致性", "question": "两帧是否自然衔接？",
                 "weight": 1.0, "min_score": 6},
            ],
            "memory": {"path": "_memory.jsonl", "promote_threshold": 2},
        }
        exp = Experiment(task="api_pipeline_smoke", base_dir=base)
        exp.bind_query(args.query)
        exp.snapshot_config(cfg)
        director = SegmentDirector(exp, cfg)
        final = director.run(args.query)
        print("✓ 成片:", final)
        print("✓ doctor:", check_experiment(exp.root))
        for f in sorted((exp.root / "eval").glob("*.json")):
            recs = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(recs, list):
                print(" ", f.name,
                      [(r.get("scores"), r.get("passed")) for r in recs
                       if isinstance(r, dict) and "scores" in r])
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
