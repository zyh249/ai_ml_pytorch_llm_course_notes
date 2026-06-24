"""逐个运行本目录主要示例；未安装 torch 时会提示并跳过相关脚本。"""
import subprocess
import sys
from pathlib import Path

for script in sorted(Path(__file__).parent.glob("[0-9][0-9]_*.py")):
    print("\n" + "=" * 78)
    print(f"Running {script.name}")
    print("=" * 78)
    result = subprocess.run([sys.executable, str(script)], text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr.strip())
