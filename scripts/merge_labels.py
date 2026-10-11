"""Merge hand/teacher labels ({"id", "labels"} per line) with an unlabeled pool and validate.

    python scripts/merge_labels.py data/seed/pool.jsonl data/seed/labels_*.jsonl --out data/seed/seed.jsonl --gold
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loa import PROMPT_VERSION  # noqa: E402
from loa.validate import check  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("pool"); p.add_argument("labels", nargs="+"); p.add_argument("--out"); p.add_argument("--gold", action="store_true")
p.add_argument("--teacher", default="claude-opus-5-5 (interactive)")
a = p.parse_args()

pool = {json.loads(l)["id"]: json.loads(l) for l in open(a.pool) if l.strip()}
rows, bad = [], 0
for path in a.labels:
    for line in open(path):
        if not line.strip():
            continue
        lab = json.loads(line)
        ex = pool[lab["id"]]
        meta = {"teacher": a.teacher, "prompt_version": PROMPT_VERSION, "gold": a.gold,
                "labeled_at": datetime.now(timezone.utc).isoformat()}
        row, errors, warnings = check(ex, lab["labels"], meta)
        for w in warnings:
            print(f"warn   {lab['id']}: {w}")
        if row is None:
            bad += 1
            print(f"REJECT {lab['id']}: {errors}")
        else:
            rows.append(row.model_dump(mode="json"))
print(f"{len(rows)} valid, {bad} rejected")
if a.out:
    with open(a.out, "w") as f:
        for r in sorted(rows, key=lambda r: r["id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
