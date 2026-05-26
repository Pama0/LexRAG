# scripts/check_layering.py
"""分层守卫：core/ 不得 import api/。CI/本地手动运行。"""
import pathlib
import re
import sys

CORE = pathlib.Path(__file__).resolve().parent.parent / "core"
PATTERN = re.compile(r"^\s*(?:from|import)\s+api(?:\.|\s|$)", re.MULTILINE)


def main() -> int:
    offenders = []
    for py in CORE.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if PATTERN.search(text):
            offenders.append(str(py))
    if offenders:
        print("分层违规：以下 core 文件 import 了 api：")
        for f in offenders:
            print("  -", f)
        return 1
    print("分层守卫通过：core/ 未 import api/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
