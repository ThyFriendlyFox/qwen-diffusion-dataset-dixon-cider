"""Serialize a Row into the fixed-slot block canvas the diffusion planner is
trained on, plus the semantic noise schedule.

Canvas layout (one string; rung spans are returned as char offsets so a trainer
can map them to token positions via the tokenizer's offset mapping):

    <intent> ... </intent><structure> ... </structure><sketch> <B1 ...> ... <B16 NULL> </sketch>

Semantic schedule: at noise level t in [0, 1] (1 = fully masked), each rung has
a band [lo, hi]. Its mask rate rises linearly from 0 at lo to 1 at hi, so the
reverse process unmasks intent first, then structure, then sketch.
"""

from __future__ import annotations

import random

RUNGS = ("intent", "structure", "sketch")
DEFAULT_BANDS = {"sketch": (0.0, 0.4), "structure": (0.3, 0.7), "intent": (0.6, 1.0)}
NULL = "NULL"


def render(row: dict, slots: int = 16) -> dict:
    lab = row["labels"]
    it, st, blocks = lab["intent"], lab["structure"], lab["sketch"]["blocks"]
    if len(blocks) > slots:
        raise ValueError(f"{row['id']}: {len(blocks)} blocks > {slots} slots")

    intent = "\n".join(f"{k}: " + " | ".join(it[k]) for k in ("ideas", "emotions", "actions", "audience"))
    structure = f"form: {st['form']}\n" + "\n".join(f"{s['id']} [{s['glue']}] {s['purpose']}" for s in st["outline"])
    lines = [
        f"<{b['id']} {b['section']} {b['kind']} {b['support']} {b['confidence']:.2f}> {b['sketch']}"
        + (f" [RETRIEVE: {b['retrieve']}]" if b.get("retrieve") else "")
        for b in blocks
    ]
    lines += [f"<B{i} {NULL}>" for i in range(len(blocks) + 1, slots + 1)]
    sketch = "\n".join(lines)

    text, spans = "", []
    for rung, body in zip(RUNGS, (intent, structure, sketch)):
        text += f"<{rung}>\n"
        start = len(text)
        text += body
        spans.append({"rung": rung, "start": start, "end": len(text)})
        text += f"\n</{rung}>\n"
    return {"canvas": text, "rung_spans": spans, "n_blocks": len(blocks), "slots": slots}


def mask_rate(rung: str, t: float, bands: dict = DEFAULT_BANDS) -> float:
    lo, hi = bands[rung]
    return min(1.0, max(0.0, (t - lo) / (hi - lo)))


def sample_mask(n_tokens_by_rung: dict[str, int], t: float, rng: random.Random | None = None,
                bands: dict = DEFAULT_BANDS) -> dict[str, list[bool]]:
    """Per-rung boolean masks (True = masked) for a single training example at noise level t."""
    rng = rng or random.Random()
    return {r: [rng.random() < mask_rate(r, t, bands) for _ in range(n)] for r, n in n_tokens_by_rung.items()}
