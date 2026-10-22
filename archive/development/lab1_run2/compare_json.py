"""Compare two result JSONs after dropping every ``timing`` block (read-only).

The brief's reproducibility clause: "two runs agree except for the timing
fields". Timing lives under a single ``timing`` key per file, but the blocks are
nested (``timing.per_fold_seconds`` etc.), so the key is removed recursively
before the structural comparison.

Usage:
    python compare_json.py A.json B.json
Prints the verdict and, on a mismatch, the first differing JSON path.
"""
import json
import sys


def strip_timing(node):
    if isinstance(node, dict):
        return {k: strip_timing(v) for k, v in node.items() if k != "timing"}
    if isinstance(node, list):
        return [strip_timing(v) for v in node]
    return node


def first_difference(a, b, path="$"):
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        if set(a) != set(b):
            return f"{path}: keys differ: only-A={sorted(set(a) - set(b))} only-B={sorted(set(b) - set(a))}"
        for key in a:
            difference = first_difference(a[key], b[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for index, (item_a, item_b) in enumerate(zip(a, b)):
            difference = first_difference(item_a, item_b, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    with open(argv[1], encoding="utf-8") as handle:
        left = strip_timing(json.load(handle))
    with open(argv[2], encoding="utf-8") as handle:
        right = strip_timing(json.load(handle))
    difference = first_difference(left, right)
    if difference is None:
        print(f"IDENTICAL (timing excluded): {argv[1]} == {argv[2]}")
        return 0
    print(f"DIFFERENT: {difference}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
