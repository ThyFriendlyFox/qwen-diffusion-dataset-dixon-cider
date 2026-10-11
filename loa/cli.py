"""LoA dataset CLI.

    python -m loa.cli fetch  --mix mix.json --out data/raw/pool.jsonl
    python -m loa.cli label  --in data/raw/pool.jsonl --out data/labeled/rows.jsonl [--batch]
    python -m loa.cli build  --in data/labeled/rows.jsonl data/seed/seed.jsonl --out data/release
    python -m loa.cli check  --in data/seed/seed.jsonl
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path


def _read(paths):
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def cmd_fetch(a):
    from .sources import SOURCES, sample

    mix = json.load(open(a.mix))
    out = []
    for name, n in mix["sources"].items():
        got = list(sample(SOURCES[name], n, seed=a.seed))
        print(f"{name:16s} {len(got):6d}/{n}", file=sys.stderr)
        out += got
    print(f"wrote {_write(a.out, out)} -> {a.out}", file=sys.stderr)


def cmd_label(a):
    from . import teacher

    done = {r["id"] for r in _read([a.out])} if os.path.exists(a.out) and not a.batch_id else set()
    todo = [ex for ex in _read([a.inp]) if ex["id"] not in done][: a.limit or None]
    if a.batch and not a.batch_id:
        bid = teacher.submit_batch(todo, a.model, a.effort)
        Path(a.out + ".batch").write_text(json.dumps({"batch_id": bid, "ids": [e["id"] for e in todo]}))
        print(f"submitted batch {bid} ({len(todo)} requests); collect with --batch-id {bid}", file=sys.stderr)
        return
    if a.batch_id:
        ids = json.loads(Path(a.out + ".batch").read_text())["ids"]
        by_id = {ex["id"]: ex for ex in _read([a.inp])}
        results = teacher.collect_batch(a.batch_id, [by_id[i] for i in ids], a.model)
    else:
        results = teacher.label_sync(todo, a.model, a.effort, a.workers)
    ok = bad = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as good, open(a.out + ".rejects", "a") as rej:
        for r in results:
            if r["ok"]:
                good.write(json.dumps(r["row"], ensure_ascii=False) + "\n"); ok += 1
            else:
                rej.write(json.dumps(r, ensure_ascii=False) + "\n"); bad += 1
            print(f"\rok={ok} rejected={bad}", end="", file=sys.stderr)
    print(file=sys.stderr)


def cmd_check(a):
    """Re-validate already-labeled rows (e.g. hand-labeled seed) and fill alignment/evidence."""
    from .validate import check

    ok, out = 0, []
    for r in _read(a.inp):
        ex = {k: r.get(k) for k in ("id", "source", "domain", "license", "context", "target", "reference_answer", "revision", "stage")}
        ex["stage"] = ex["stage"] or r.get("meta", {}).get("stage", "plan")
        if ex.get("revision"):
            ex["revision"] = {k: ex["revision"].get(k) for k in ("draft", "human_feedback")}
        row, errors, warnings = check(ex, r["labels"], {k: v for k, v in r.get("meta", {}).items() if k not in ("warnings", "stage")})
        if row is None:
            print(f"REJECT {r['id']}: {errors}", file=sys.stderr)
            continue
        for w in warnings:
            print(f"warn   {r['id']}: {w}", file=sys.stderr)
        ok += 1
        out.append(row.model_dump(mode="json"))
    print(f"{ok} valid", file=sys.stderr)
    if a.out:
        _write(a.out, out)


def _split(row_id: str, val: float, test: float) -> str:
    h = int(hashlib.sha1(row_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else "validation" if h < test + val else "train"


def cmd_build(a):
    from .canvas import render

    seen, splits, stats = set(), collections.defaultdict(list), collections.Counter()
    for r in _read(a.inp):
        key = hashlib.sha1(json.dumps(r["context"], sort_keys=True).encode()).hexdigest()
        if key in seen or r["id"] in seen:
            stats["dup"] += 1
            continue
        seen |= {key, r["id"]}
        if r["labels"]["quality"]["score"] < a.min_quality and r["meta"].get("stage") != "critic":
            stats["low_quality"] += 1
            continue
        if len(r["labels"]["sketch"]["blocks"]) > a.slots:
            stats["too_many_blocks"] += 1
            continue
        r = {**r, **render(r, a.slots)}
        sp = "seed" if r["meta"].get("gold") else _split(r["id"], a.val, a.test)
        splits[sp].append(r)
        stats[f"{sp}:{r['domain']}"] += 1
    for sp, rows in splits.items():
        _write(Path(a.out) / f"{sp}.jsonl", rows)
    json.dump(dict(sorted(stats.items())), open(Path(a.out) / "stats.json", "w"), indent=2)
    print(json.dumps(dict(sorted(stats.items())), indent=2), file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(prog="loa")
    sub = p.add_subparsers(required=True)

    f = sub.add_parser("fetch"); f.set_defaults(fn=cmd_fetch)
    f.add_argument("--mix", default="mix.json"); f.add_argument("--out", default="data/raw/pool.jsonl")
    f.add_argument("--seed", type=int, default=0)

    l = sub.add_parser("label"); l.set_defaults(fn=cmd_label)
    l.add_argument("--in", dest="inp", default="data/raw/pool.jsonl"); l.add_argument("--out", default="data/labeled/rows.jsonl")
    l.add_argument("--model", default="claude-opus-5-5"); l.add_argument("--effort", default="medium")
    l.add_argument("--workers", type=int, default=8); l.add_argument("--limit", type=int, default=0)
    l.add_argument("--batch", action="store_true", help="submit via Message Batches (50%% cost)")
    l.add_argument("--batch-id", help="collect a previously submitted batch")

    c = sub.add_parser("check"); c.set_defaults(fn=cmd_check)
    c.add_argument("--in", dest="inp", nargs="+", required=True); c.add_argument("--out")

    b = sub.add_parser("build"); b.set_defaults(fn=cmd_build)
    b.add_argument("--in", dest="inp", nargs="+", required=True); b.add_argument("--out", default="data/release")
    b.add_argument("--slots", type=int, default=16); b.add_argument("--min-quality", type=int, default=3)
    b.add_argument("--val", type=float, default=0.02); b.add_argument("--test", type=float, default=0.02)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
