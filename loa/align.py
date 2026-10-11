"""Resolve block anchors to character spans of the target."""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")


def _norm_map(s: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed, NFKC-normalized text plus a map back to original offsets."""
    out, idx = [], []
    prev_space = False
    for i, ch in enumerate(s):
        for c in unicodedata.normalize("NFKC", ch):
            if c.isspace():
                if prev_space:
                    continue
                c, prev_space = " ", True
            else:
                prev_space = False
            out.append(c)
            idx.append(i)
    return "".join(out), idx


def _find(target: str, anchor: str, start: int) -> int:
    i = target.find(anchor, start)
    if i >= 0:
        return i
    nt, idx = _norm_map(target)
    na, _ = _norm_map(anchor.strip())
    if not na:
        return -1
    ns = next((k for k, o in enumerate(idx) if o >= start), len(nt))
    j = nt.find(na, ns)
    if j < 0:  # last resort: prefix of the anchor (teacher sometimes runs past a line break)
        for cut in range(len(na) - 1, max(8, len(na) // 2), -1):
            j = nt.find(na[:cut], ns)
            if j >= 0:
                break
    return idx[j] if j >= 0 else -1


def align(target: str, blocks: list[dict]) -> tuple[list[dict], list[str]]:
    """Return (spans, errors). Spans are contiguous and cover target[0:len]."""
    errors, starts, cursor = [], [], 0
    for b in blocks:
        pos = _find(target, b["anchor"], cursor)
        if pos < 0:
            errors.append(f"{b['id']}: anchor not found in order: {b['anchor'][:60]!r}")
            return [], errors
        starts.append(pos)
        cursor = pos + 1
    if starts and target[: starts[0]].strip():
        errors.append(f"{blocks[0]['id']}: first anchor skips leading text {target[:starts[0]][:60]!r}")
        starts[0] = 0
    if starts:
        starts[0] = 0
    ends = starts[1:] + [len(target)]
    return [{"block": b["id"], "start": s, "end": e} for b, s, e in zip(blocks, starts, ends)], errors
