#!/usr/bin/env python3
"""Prepend the new report to build-report.md and demote the previous one to an appendix.

Keeps the previous report byte-for-byte (only its top-level heading is renamed and its
own leading STATUS line is dropped, since the new report carries the STATUS line).
"""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
target = root / "build-report.md"
new_section = (root / ".coding-dept" / "radar-absolute" / "report_new.md").read_text("utf-8")
previous = root / ".coding-dept" / "radar-absolute" / "build-report.prev.md"
# Always compose from the preserved previous report, so re-running is idempotent.
if previous.is_file():
    old_bytes = previous.read_bytes()
else:
    old_bytes = target.read_bytes()
    previous.write_bytes(old_bytes)

old = old_bytes.decode("utf-8")
lines = old.splitlines(keepends=True)
# drop the previous STATUS line(s) at the very top
while lines and not lines[0].strip():
    lines.pop(0)
if lines and lines[0].startswith("STATUS:"):
    lines.pop(0)
while lines and not lines[0].strip():
    lines.pop(0)
assert lines[0].startswith("# "), lines[0][:80]
old_heading = lines[0].rstrip("\r\n")
lines[0] = ("# 附录 Z（上一轮报告，原文保留）：TGSIM 真实数据 IDM 标定与 AV–HV 分析"
            "（Lab Session I 延伸；雷达图口径已由本轮报告取代）"
            + (lines[0][len(lines[0].rstrip("\r\n")):] or "\n"))
demoted = "".join(lines)

composed = new_section.rstrip("\n") + "\n\n---\n\n" + demoted
target.write_bytes(composed.encode("utf-8"))

print("previous heading :", old_heading)
print("new heading      :", lines[0].strip())
print("previous bytes   :", len(old_bytes))
print("new bytes        :", len(composed.encode("utf-8")))
print("first line of file:", target.read_text("utf-8").splitlines()[0])
