"""审计 HMtoumanfen666 下载包：统计文件类型并判断是否包含可运行源码。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else "HMtoumanfen666-main")
if not root.exists():
    raise SystemExit(f"目录不存在：{root.resolve()}")

files = [p for p in root.rglob("*") if p.is_file()]
counts = Counter(p.suffix.lower() or "<无扩展名>" for p in files)
print("文件总数:", len(files))
for suffix, count in counts.most_common():
    print(f"{suffix:>12}: {count}")

source_exts = {".py", ".ipynb", ".md", ".yaml", ".yml"}
source_files = [p for p in files if p.suffix.lower() in source_exts]
print("\n可运行/可编辑源码文件:")
for p in source_files[:30]:
    print(" -", p.relative_to(root))
if not source_files:
    print("未发现 .py/.ipynb/.md/.yaml；该下载包更像已构建的静态讲义网站。")
