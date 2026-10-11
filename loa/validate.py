"""Turn a teacher response into a validated Row, collecting errors and warnings.

Errors reject the row. Warnings are kept in row.meta["warnings"] so they can be
filtered or audited later.
"""

from __future__ import annotations

import difflib
import re

from pydantic import ValidationError

from .align import align
from .schema import Labels, Row

_TOOL_RESP = re.compile(r"<tool_response>.*?</tool_response>", re.S)
_TOOL_CALL = re.compile(r"<tool_call\b")


def check(ex: dict, labels: dict, meta: dict | None = None) -> tuple[Row | None, list[str], list[str]]:
    errors, warnings = [], []
    try:
        lab = Labels.model_validate(labels)
    except ValidationError as e:
        return None, [f"schema: {err['loc']}: {err['msg']}" for err in e.errors()], []

    secs = [s.id for s in lab.structure.outline]
    if secs != [f"S{i}" for i in range(1, len(secs) + 1)]:
        errors.append(f"section ids not S1..Sn: {secs}")
    blocks = lab.sketch.blocks
    if [b.id for b in blocks] != [f"B{i}" for i in range(1, len(blocks) + 1)]:
        errors.append(f"block ids not B1..Bk: {[b.id for b in blocks]}")
    order = []
    for b in blocks:
        if b.section not in secs:
            errors.append(f"{b.id}: unknown section {b.section}")
        else:
            order.append(secs.index(b.section))
        if b.support == "retrieve" and not b.retrieve:
            errors.append(f"{b.id}: support=retrieve but no query")
        if b.support != "retrieve" and b.retrieve:
            warnings.append(f"{b.id}: retrieve query on support={b.support}")
    if order != sorted(order):
        warnings.append("blocks do not follow outline order")
    if missing := set(secs) - {b.section for b in blocks}:
        warnings.append(f"sections without blocks: {sorted(missing)}")

    target = ex["target"]
    spans, align_err = align(target, [b.model_dump() for b in blocks])
    errors += align_err
    resp = [(m.start(), m.end()) for m in _TOOL_RESP.finditer(target)]
    for sp in spans:
        if any(s < sp["start"] < e for s, e in resp):
            errors.append(f"{sp['block']}: anchored inside a tool_response")

    by_id = {b.id: b for b in blocks}
    for sp in spans:
        b, text = by_id[sp["block"]], target[sp["start"] : sp["end"]]
        if not text.strip():
            errors.append(f"{b.id}: empty span")
            continue
        sim = difflib.SequenceMatcher(None, b.sketch.lower(), text.lower()[: len(b.sketch) * 2]).ratio()
        if sim > 0.8 and len(text) > 80:
            warnings.append(f"{b.id}: sketch nearly copies target (sim={sim:.2f})")
        if _TOOL_CALL.search(text) and b.kind != "tool_call" and not any(
            _TOOL_CALL.search(target[o["start"] : o["end"]]) and by_id[o["block"]].kind == "tool_call" for o in spans
        ):
            warnings.append(f"{b.id}: contains a tool call but no block is kind=tool_call")

    if errors:
        return None, errors, warnings
    row = Row(
        id=ex["id"], source=ex["source"], domain=ex["domain"], license=ex["license"],
        context=ex["context"], target=target, reference_answer=ex.get("reference_answer"),
        labels=lab, alignment=spans,
        evidence=[{"block": b.id, "query": b.retrieve} for b in blocks if b.support == "retrieve"],
        revision=({**ex["revision"], "critiques": [c.model_dump() for c in lab.quality.issues]} if ex.get("revision") else None),
        meta={**(meta or {}), "stage": ex.get("stage", "plan"), "warnings": warnings},
    )
    return row, errors, warnings
