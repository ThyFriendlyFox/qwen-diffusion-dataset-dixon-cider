"""Generate the README figures (docs/img/*.svg).

    python scripts/make_figures.py

Figures follow the OS light/dark setting through a CSS media query inside each
SVG, and draw their own background so they stay readable on any page color.
The worked example is the gsm8k row in data/seed/seed.jsonl.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
sys.path.insert(0, str(ROOT))
from loa.canvas import DEFAULT_BANDS, mask_rate  # noqa: E402

STYLE = """
<style>
  svg { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --ink3:#8a8984; --line:#d9d8d3; --card:#ffffff;
        --mask:#e4e3de; --intent:#2a78d6; --structure:#eb6834; --sketch:#1baf7a; --text:#52514e;
        --intent-bg:#e8f0fb; --structure-bg:#fdeee7; --sketch-bg:#e5f6ef; --text-bg:#efeeea; }
  @media (prefers-color-scheme: dark) {
    svg { --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --ink3:#8f8e86; --line:#3a3a37; --card:#232322;
          --mask:#34342f; --intent:#3987e5; --structure:#d95926; --sketch:#199e70; --text:#c3c2b7;
          --intent-bg:#1c2a3d; --structure-bg:#3a2419; --sketch-bg:#16302a; --text-bg:#2b2b29; }
  }
  text { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; fill: var(--ink); }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .h { font-size: 20px; font-weight: 700; }
  .t { font-size: 15px; font-weight: 600; }
  .b { font-size: 13px; }
  .s { font-size: 12px; fill: var(--ink2); }
  .xs { font-size: 11px; fill: var(--ink3); }
  .bg { fill: var(--bg); }
  .card { fill: var(--card); stroke: var(--line); stroke-width: 1; }
  .rule { stroke: var(--line); stroke-width: 1; fill: none; }
  .arrow { stroke: var(--ink3); stroke-width: 2; fill: none; marker-end: url(#ah); }
  .ah { fill: var(--ink3); }
</style>
"""

RUNG_COLOR = {"intent": "var(--intent)", "structure": "var(--structure)", "sketch": "var(--sketch)", "text": "var(--text)"}
RUNG_BG = {"intent": "var(--intent-bg)", "structure": "var(--structure-bg)", "sketch": "var(--sketch-bg)", "text": "var(--text-bg)"}


def svg(w: int, h: int, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{escape(title)}">'
        f"<title>{escape(title)}</title>{STYLE}"
        '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path class="ah" d="M0,0 L10,5 L0,10 z"/></marker></defs>'
        f'<rect class="bg" x="0" y="0" width="{w}" height="{h}" rx="12"/>{body}</svg>'
    )


def text(x, y, s, cls="b", anchor="start", extra=""):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}" {extra}>{escape(s)}</text>'


def rect(x, y, w, h, fill, rx=8, stroke=None, sw=1.5, extra=""):
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{st} {extra}/>'


def arrow(x1, y1, x2, y2, cls="arrow", dash=False):
    d = ' stroke-dasharray="5 4"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}"{d}/>'


def path_arrow(d, dash=False):
    da = ' stroke-dasharray="5 4"' if dash else ""
    return f'<path d="{d}" class="arrow"{da}/>'


def load_seed():
    rows = [json.loads(l) for l in open(ROOT / "data/seed/seed.jsonl")]
    ex = next(r for r in rows if r["id"] == "gsm8k:72bbd7e7e2ca")
    return rows, ex


# --------------------------------------------------------------------------- 1. the ladder (ELI5)


def fig_ladder(ex):
    lab = ex["labels"]
    W, H = 880, 640
    b = [text(32, 44, "Planning an answer, one rung at a time", "h"),
         text(32, 68, "Question: " + ex["context"][0]["content"][:96] + "...", "s")]
    rungs = [
        ("intent", "1  Intent", "What am I trying to say, and to whom?", [
            "ideas: " + "; ".join(lab["intent"]["ideas"]),
            "tone: " + ", ".join(lab["intent"]["emotions"]),
            "goal: " + ", ".join(lab["intent"]["actions"]),
            "reader: " + ", ".join(lab["intent"]["audience"])]),
        ("structure", "2  Structure", "What shape will the answer take?", [
            "form: " + lab["structure"]["form"]] + [
            f'{s["id"]} {s["purpose"]} (glue: {s["glue"]})' for s in lab["structure"]["outline"]]),
        ("sketch", "3  Sketch", "Rough notes for each part, like a first scribble", [
            f'{bl["id"]} "{bl["sketch"]}" | {bl["kind"]}, {bl["support"]}, {bl["confidence"]:.2f}' for bl in lab["sketch"]["blocks"]]),
        ("text", "4  Final text", "Written out neatly by the big model", ex["target"].replace("**", "").split("\n")),
    ]
    y = 92
    for i, (key, name, ask, lines) in enumerate(rungs):
        lines = [l for l in lines if l.strip()]
        h = 46 + 18 * len(lines)
        b.append(rect(32, y, W - 64, h, RUNG_BG[key], rx=10))
        b.append(rect(32, y, 6, h, RUNG_COLOR[key], rx=3))
        b.append(text(54, y + 26, name, "t", extra='xml:space="preserve"'))
        b.append(text(190, y + 26, ask, "s"))
        for j, l in enumerate(lines):
            b.append(text(54, y + 48 + 18 * j, l[:118], "b mono" if key in ("sketch", "text") else "b"))
        y += h
        if i < len(rungs) - 1:
            b.append(arrow(W / 2, y + 2, W / 2, y + 20))
            y += 24
    b.append(text(32, y + 26, "Each rung gets more concrete. Only the last one has finished sentences.", "s"))
    return svg(W, y + 44, "".join(b), "Ladder of abstraction for one math answer")


# --------------------------------------------------------------------------- 2. architecture


def fig_architecture():
    W, H = 960, 420
    b = [text(32, 44, "Where the dataset is used", "h"),
         text(32, 68, "A small diffusion model plans; a large autoregressive model writes. This repo builds the training data for both sides.", "s")]
    # prompt
    b.append(rect(32, 170, 120, 70, "var(--card)", stroke="var(--line)"))
    b += [text(92, 200, "Prompt", "t", "middle"), text(92, 220, "user question", "xs", "middle")]
    b.append(arrow(154, 205, 194, 205))
    # planner
    b.append(rect(196, 110, 250, 190, "var(--card)", stroke="var(--line)"))
    b += [text(321, 138, "Diffusion planner", "t", "middle"), text(321, 156, "small (< 14B), masked discrete diffusion", "xs", "middle")]
    for i, k in enumerate(["intent", "structure", "sketch"]):
        b.append(rect(216, 172 + i * 36, 210, 28, RUNG_BG[k], rx=6))
        b.append(rect(216, 172 + i * 36, 5, 28, RUNG_COLOR[k], rx=2))
        b.append(text(232, 191 + i * 36, ["intent rung", "structure rung", "sketch blocks B1..B16"][i], "b"))
    # critic loop
    b.append(path_arrow("M 446 260 C 500 260, 500 330, 400 330 L 330 330 C 290 330, 290 305, 300 302", dash=True))
    b.append(text(480, 352, "critic: re-mask a wrong block and redo it", "s", "middle"))
    b.append(arrow(448, 205, 500, 205))
    # projector
    b.append(rect(502, 160, 130, 90, "var(--card)", stroke="var(--line)"))
    b += [text(567, 196, "Projector", "t", "middle"), text(567, 216, "2-layer MLP", "xs", "middle"),
          text(567, 232, "block → AR space", "xs", "middle")]
    b.append(arrow(634, 205, 674, 205))
    # AR
    b.append(rect(676, 130, 170, 150, "var(--card)", stroke="var(--line)"))
    b += [text(761, 160, "AR verbalizer", "t", "middle"), text(761, 178, "large model, multi-token", "xs", "middle"),
          text(761, 194, "prediction heads", "xs", "middle"),
          text(761, 224, "ranks the plan's claims", "s", "middle"), text(761, 242, "by support + confidence,", "s", "middle"),
          text(761, 260, "then writes the answer", "s", "middle")]
    b.append(arrow(848, 205, 888, 205))
    b += [text(912, 200, "Final", "t", "middle"), text(912, 218, "text", "t", "middle")]
    # retrieval
    b.append(rect(676, 330, 170, 52, "var(--card)", stroke="var(--line)"))
    b += [text(761, 352, "Retrieval", "t", "middle"), text(761, 370, "fills RETRIEVE slots", "xs", "middle")]
    b.append(arrow(761, 328, 761, 284))
    # dataset callouts
    b.append(rect(196, 84, 436, 18, "none", rx=4))
    b.append(text(321, 96, "learns from: labels.intent / structure / sketch", "xs", "middle"))
    b.append(text(567, 148, "learns from: alignment", "xs", "middle"))
    b.append(text(761, 118, "learns from: target", "xs", "middle"))
    return svg(W, H, "".join(b), "Architecture: diffusion planner, projector, AR verbalizer")


# --------------------------------------------------------------------------- 3. denoising snapshots


def _tokens(ex):
    lab = ex["labels"]
    intent = " ".join(["solve", "show", "work", "|", "grade-school", "student", "|", "neutral", "precise"])
    struct = " ".join(["S1", "compute", "each", "distance", "S2", "take", "difference", "S3", "answer"])
    sketch = " ".join(["B1", "Tamika", "=", "hrs", "x", "mph", "B2", "same", "for", "Logan", "B3", "subtract", "B4", "85"])
    return {"intent": intent.split(), "structure": struct.split(), "sketch": sketch.split()}


def fig_denoise(ex):
    toks = _tokens(ex)
    steps = [1.0, 0.75, 0.5, 0.25, 0.0]
    colW, W = 228, 32 + 228 * 5 + 16
    b = [text(32, 44, "Denoising reveals the plan from the top rung down", "h"),
         text(32, 68, "Same canvas at five noise levels t. Grey cells are still masked. Masks are drawn deterministically here; training samples them.", "s")]
    y0 = 100
    for c, t in enumerate(steps):
        x0 = 32 + c * colW
        b.append(text(x0 + 4, y0, f"t = {t:.2f}", "t"))
        b.append(text(x0 + 4, y0 + 18, ["all masked", "", "", "", "plan complete"][c], "xs"))
        y = y0 + 34
        for rung in ("intent", "structure", "sketch"):
            rate = mask_rate(rung, t)
            tk = toks[rung]
            n_mask = round(rate * len(tk))
            # reveal from the start of the rung; deterministic for the picture
            masked = set(range(len(tk) - n_mask, len(tk)))
            x, ry = x0 + 4, y
            b.append(text(x0 + 4, y + 2, rung, "xs"))
            ry += 10
            for i, w in enumerate(tk):
                tw = max(18, 6.4 * len(w) + 10)
                if x + tw > x0 + colW - 12:
                    x, ry = x0 + 4, ry + 24
                if i in masked:
                    b.append(rect(x, ry, tw, 20, "var(--mask)", rx=4))
                else:
                    b.append(rect(x, ry, tw, 20, RUNG_BG[rung], rx=4, stroke=RUNG_COLOR[rung], sw=1))
                    b.append(text(x + tw / 2, ry + 14, w, "xs mono", "middle", 'style="fill:var(--ink)"'))
                x += tw + 4
            y = ry + 36
        if c < len(steps) - 1:
            b.append(arrow(x0 + 110, y0 - 5, x0 + colW - 20, y0 - 5))
    H = y0 + 360
    b.append(text(32, H - 18, "Intent settles first, then structure, then the sketch. The AR model writes the text from the finished plan.", "s"))
    return svg(W, H, "".join(b), "Denoising snapshots of the plan canvas")


# --------------------------------------------------------------------------- 4. noise schedule chart


def fig_schedule():
    W, H = 720, 360
    L, R, T, B = 70, 160, 84, 300
    pw, ph = W - L - R + 60, B - T
    b = [text(32, 40, "Semantic noise schedule: mask rate by rung", "h")]
    for v in (0, 0.5, 1):
        y = B - v * ph
        b.append(f'<line x1="{L}" y1="{y}" x2="{L + pw}" y2="{y}" class="rule"/>')
        b.append(text(L - 10, y + 4, f"{v:.0%}" if v else "0%", "xs", "end"))
    for v in (0, 0.25, 0.5, 0.75, 1):
        b.append(text(L + v * pw, B + 20, f"{v:g}", "xs", "middle"))
    b.append(text(L + pw / 2, B + 42, "noise level t  (training samples t; generation runs from 1 down to 0)", "s", "middle"))
    b.append(text(20, T + ph / 2, "masked", "s", "middle", f'transform="rotate(-90 20 {T + ph / 2})"'))
    labels_y = {}
    for rung in ("intent", "structure", "sketch"):
        pts = [(L + t / 100 * pw, B - mask_rate(rung, t / 100) * ph) for t in range(0, 101)]
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        b.append(f'<path d="{d}" fill="none" stroke="{RUNG_COLOR[rung]}" stroke-width="2.5" stroke-linejoin="round"/>')
        lo, hi = DEFAULT_BANDS[rung]
        labels_y[rung] = (lo + hi) / 2
    # direct labels at the midpoint of each ramp
    for rung, dy in (("sketch", -10), ("structure", -10), ("intent", -10)):
        lo, hi = DEFAULT_BANDS[rung]
        mx = L + (lo + hi) / 2 * pw
        my = B - 0.5 * ph
        b.append(text(mx - 8, my + dy, rung, "b", "end", 'style="font-weight:600"'))
    b.append(text(32, 60, "Reading right to left (generation): intent unmasks first, then structure, then sketch.", "s"))
    return svg(W - 40, H + 10, "".join(b), "Mask rate versus noise level for each rung")


# --------------------------------------------------------------------------- 5. pipeline


def fig_pipeline():
    W, H = 1150, 300
    b = [text(32, 44, "How the dataset is built", "h"),
         text(32, 68, "python -m loa.cli fetch → label → build", "s mono")]
    boxes = [
        ("Sources", ["9 public HF datasets", "chat, math, science,", "agentic, critic"], "card"),
        ("fetch", ["sample rows,", "normalize to", "context + target"], "card"),
        ("label", ["teacher model writes", "intent, structure,", "sketch, quality"], "card"),
        ("validate", ["anchors → spans", "reject bad labels", "(.rejects file)"], "card"),
        ("build", ["dedup, quality ≥ 3,", "canvas + spans,", "split by id hash"], "card"),
    ]
    bw, gap, x, y = 160, 30, 32, 100
    for i, (title, lines, _) in enumerate(boxes):
        b.append(rect(x, y, bw, 110, "var(--card)", stroke="var(--line)"))
        b.append(text(x + bw / 2, y + 28, title, "t", "middle"))
        for j, l in enumerate(lines):
            b.append(text(x + bw / 2, y + 54 + 18 * j, l, "s", "middle"))
        if i < len(boxes) - 1:
            b.append(arrow(x + bw + 2, y + 55, x + bw + gap - 2, y + 55))
        x += bw + gap
    b.append(arrow(x - gap + 2, y + 55, x + 2, y + 55))
    for j, (name, sub) in enumerate((("train", "96%"), ("validation", "2%"), ("test", "2%"), ("seed", "gold rows"))):
        yy = y - 14 + j * 36
        b.append(rect(x + 6, yy, 140, 28, "var(--text-bg)", rx=6))
        b.append(text(x + 16, yy + 19, name, "b"))
        b.append(text(x + 138, yy + 19, sub, "xs", "end"))
    b.append(text(32, 256, "Gold seed set: 37 rows labeled by a frontier model and passed through the same checks, kept as its own split for audits and benchmarks.", "s"))
    return svg(W, H, "".join(b), "Dataset build pipeline")


# --------------------------------------------------------------------------- 6. alignment


def fig_alignment(ex):
    blocks = ex["labels"]["sketch"]["blocks"]
    spans = ex["alignment"]
    tgt = ex["target"]
    W = 900
    b = [text(32, 44, "Each sketch block is tied to the exact text it became", "h"),
         text(32, 68, "These (block, span) pairs train the projector. Spans are contiguous and cover the whole answer.", "s")]
    cols = ["var(--intent)", "var(--structure)", "var(--sketch)", "var(--ink3)"]
    y = 100
    for i, (bl, sp) in enumerate(zip(blocks, spans)):
        c = cols[i % len(cols)]
        piece = tgt[sp["start"]:sp["end"]].strip().replace("**", "")
        b.append(rect(32, y, 330, 46, "var(--card)", stroke=c, sw=2))
        b.append(text(46, y + 20, f'{bl["id"]}  {bl["kind"]}', "b", extra='style="font-weight:600"'))
        b.append(text(46, y + 37, f'"{bl["sketch"]}"', "s mono"))
        b.append(arrow(364, y + 23, 430, y + 23))
        lines = piece.split("\n")
        b.append(rect(432, y, 436, 46, "var(--text-bg)", stroke=c, sw=2))
        for j, l in enumerate(lines[:2]):
            b.append(text(446, y + (28 if len(lines) == 1 else 20 + 17 * j), l, "b mono"))
        b.append(text(860, y + 40, f'chars {sp["start"]}-{sp["end"]}', "xs", "end"))
        y += 58
    return svg(W, y + 20, "".join(b), "Sketch blocks aligned to spans of the answer")


# --------------------------------------------------------------------------- 7. seed stats


def fig_seed(rows):
    W, H = 900, 300
    dom = Counter(r["domain"] for r in rows)
    order = ["chat", "math", "science", "agentic", "code", "writing"]
    dom = [(d, dom[d]) for d in order if dom[d]]
    q = Counter(r["labels"]["quality"]["score"] for r in rows)
    sup = Counter(bl["support"] for r in rows for bl in r["labels"]["sketch"]["blocks"])
    b = [text(32, 40, f"Seed set: {len(rows)} rows, {sum(sup.values())} blocks", "h")]

    def bars(x0, title, items, color, maxv, note=None):
        out = [text(x0, 78, title, "t")]
        y = 96
        for name, v in items:
            w = 150 * v / maxv
            out.append(text(x0 + 92, y + 13, name, "s", "end"))
            out.append(rect(x0 + 100, y, max(w, 3), 18, color, rx=4))
            out.append(text(x0 + 106 + w, y + 13, str(v), "xs"))
            y += 24
        if note:
            out.append(text(x0, y + 14, note, "xs"))
        return out

    b += bars(32, "Rows by domain", dom, "var(--intent)", max(v for _, v in dom),
              "critic rows are counted under their topic")
    b += bars(320, "Quality score", [(f"score {k}", q[k]) for k in (5, 4, 3, 2, 1)], "var(--structure)", max(q.values()),
              "build keeps scores 3 to 5")
    sup_order = ["supported", "derived", "context", "none_needed", "tool", "uncertain", "unsupported", "retrieve"]
    b += bars(608, "Block support", [(s, sup[s]) for s in sup_order if sup[s]], "var(--sketch)", max(sup.values()))
    return svg(W, H, "".join(b), "Seed set statistics")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, ex = load_seed()
    figs = {
        "ladder.svg": fig_ladder(ex),
        "architecture.svg": fig_architecture(),
        "denoise.svg": fig_denoise(ex),
        "schedule.svg": fig_schedule(),
        "pipeline.svg": fig_pipeline(),
        "alignment.svg": fig_alignment(ex),
        "seed.svg": fig_seed(rows),
    }
    for name, s in figs.items():
        (OUT / name).write_text(s)
        print(OUT / name)


if __name__ == "__main__":
    main()
