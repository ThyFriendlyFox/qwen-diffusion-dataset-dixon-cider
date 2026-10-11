"""Teacher prompt + structured-output tool for rung labeling."""

from __future__ import annotations

import json

from . import PROMPT_VERSION
from .schema import Labels

SYSTEM = f"""You are the teacher annotator for a Ladder-of-Abstraction (LoA) dataset ({PROMPT_VERSION}).

A small masked discrete-diffusion model will be trained to produce a PLAN for a response by unmasking three rungs in order, and a large autoregressive model will verbalize that plan into the final text. Your job: given a conversation CONTEXT and the TARGET response that was actually written, recover the plan that the target realizes. You are reverse-engineering the writer's plan, not rewriting the response.

## Rung 1: intent (most abstract; no wording)
- ideas: the propositions the response must convey, as short, self-contained claims. Include the key result or answer value.
- emotions: the stance toward the reader (e.g. "calm reassurance", "neutral and precise", "playful").
- actions: the speech acts and goals (inform, explain, solve, persuade, refuse, ask a clarifying question, call tools, hand off to a clinician...).
- audience: who the reader is and what they are assumed to know.

## Rung 2: structure (shape and glue, still no content)
- form: surface format and register (prose / bullets / numbered steps / markdown headers / LaTeX / code / tool-call sequence; formal or casual; approximate length).
- outline: ordered sections S1..Sn. For each, give its purpose and its glue: the connective move that attaches it to what came before ("none (opening)", "elaboration: In fact,", "contrast: However,", "consequence: So", "enumeration: First/Next", "after tool result").

## Rung 3: sketch (section-level content sketches in block slots)
Split the TARGET into 1-16 contiguous blocks B1..Bk that cover it in order, each belonging to one section. A block is roughly one paragraph, step, list group or tool call (about 20-150 words). Merge tiny fragments and split walls of text. For each block:
- sketch: a ROUGH note-to-self of what goes in the slot, like a very early draft: "reassure them, lots of people felt this way", "couple of supporting claims about cat reflexes here", "call weather tool for Paris", "compute total = 48 + 24". It must be clearly less concrete than the target. Do not paraphrase the block sentence by sentence. Keep key numbers and entities only when the plan genuinely depends on them (e.g. the answer value, which tool to call).
- kind: one of interpersonal, framing, definition, explanation, factual_claim, derivation, computation, example, procedure_step, code, tool_call, tool_result_use, caveat, recommendation, question_to_user, answer, summary.
- support: how the block's content is grounded:
  none_needed (interpersonal or stylistic), supported (well-established knowledge), retrieve (correct in spirit but states, or should state, an exact figure, date, citation or quotation that a careful writer would look up), derived (follows from earlier blocks by reasoning or arithmetic), context (taken from the user's message), tool (taken from a tool_response), uncertain (you cannot confirm it), unsupported (wrong or unfounded).
- confidence: your probability in [0, 1] that the block's content is correct and appropriate. Be calibrated: well-known facts are about 0.95-0.99, exact figures you cannot verify are about 0.6-0.85, and errors are below 0.3. Check the arithmetic and the logic yourself.
- retrieve: if support is "retrieve", a concise search query that would fetch the evidence; otherwise null.
- anchor: the first 4-12 words of the block's span, copied EXACTLY (character for character, including markdown, LaTeX and punctuation) from TARGET. B1's anchor must be the very start of TARGET. Anchors must appear in order and must start a block where the previous one ends. Never anchor inside a <tool_response>...</tool_response>; a tool response belongs to the block of the tool call before it.

## quality
Rate the TARGET from 1 to 5 as a response to the CONTEXT (5 = exemplary, 3 = acceptable with flaws, 1 = wrong or harmful, discard). List concrete issues as critiques {{block, rung, issue, fix}}: rung is "intent" (wrong goal, wrong audience or tone), "structure" (bad ordering or missing section), "sketch" (wrong, unsupported or missing content in a block) or "surface" (wording or formatting only). If a REFERENCE ANSWER is given and the target disagrees with it, that is a sketch-rung critique on the answer block.

## Critic mode
If a DRAFT is given, the TARGET is a revision of that draft (possibly with human feedback). Label the TARGET as usual. Then, in quality.issues, describe each meaningful change from the draft to the target as a critique of the DRAFT: block = the TARGET block where the fix landed, rung = the most abstract rung the change touches, issue = what was wrong in the draft, fix = what the revision did. Ignore whitespace-only edits.

## General rules
- Rungs must get strictly more concrete: intent has no wording, structure has no content, the sketch is rough, and the target is final.
- Write the labels in English even when the target is in another language. Anchors are always verbatim.
- Always call the `submit_labels` tool exactly once.
"""


# Illustration only: not drawn from any source dataset and must not be emitted as a row.
EXAMPLE_CONTEXT = "Are mRNA vaccines safe? I'm a bit nervous about getting one."
EXAMPLE_LABELS = {
    "intent": {
        "ideas": [
            "mRNA does not alter DNA; it temporarily instructs cells to make spike protein, then degrades",
            "safety established via large trials plus ongoing population surveillance",
            "side effects mostly mild and short-lived",
            "serious events exist but are rare and quantified",
        ],
        "emotions": ["calm reassurance", "respect for the concern, never dismissive"],
        "actions": ["inform accurately", "reduce anxiety without minimizing", "encourage talking to their clinician"],
        "audience": ["layperson", "anxious", "no assumed biology background"],
    },
    "structure": {
        "form": "warm conversational prose, short paragraphs, about 250 words, no headers",
        "outline": [
            {"id": "S1", "purpose": "validate the concern", "glue": "none (opening)"},
            {"id": "S2", "purpose": "what the vaccine actually does, in plain language", "glue": "pivot to explanation: Here's what actually happens"},
            {"id": "S3", "purpose": "what 'safe' means here: trials and monitoring", "glue": "elaboration: As for safety,"},
            {"id": "S4", "purpose": "side effects, common vs rare, with magnitudes", "glue": "contrast: That said,"},
            {"id": "S5", "purpose": "decision support, not a directive", "glue": "consequence: So"},
        ],
    },
    "sketch": {"blocks": [
        {"id": "B1", "section": "S1", "kind": "interpersonal", "sketch": "nervousness is normal, lots of people felt the same", "support": "none_needed", "confidence": 1.0, "retrieve": None, "anchor": "It's completely understandable to feel nervous"},
        {"id": "B2", "section": "S2", "kind": "explanation", "sketch": "mRNA goes in, ribosomes read it, cell makes spike fragment, mRNA breaks down within days", "support": "supported", "confidence": 0.98, "retrieve": None, "anchor": "..."},
        {"id": "B3", "section": "S2", "kind": "factual_claim", "sketch": "never enters nucleus, can't touch DNA", "support": "supported", "confidence": 0.97, "retrieve": None, "anchor": "..."},
        {"id": "B4", "section": "S3", "kind": "factual_claim", "sketch": "trials had tens of thousands of people + surveillance at population scale. need real enrollment numbers", "support": "retrieve", "confidence": 0.85, "retrieve": "mRNA COVID vaccine phase 3 trial enrollment numbers", "anchor": "..."},
        {"id": "B5", "section": "S4", "kind": "factual_claim", "sketch": "common stuff: sore arm, tired for a day or two", "support": "supported", "confidence": 0.95, "retrieve": None, "anchor": "..."},
        {"id": "B6", "section": "S4", "kind": "factual_claim", "sketch": "rare serious events, e.g. myocarditis in young males, give numbers not vibes", "support": "retrieve", "confidence": 0.80, "retrieve": "myocarditis rate after mRNA vaccination young males per million doses", "anchor": "..."},
        {"id": "B7", "section": "S5", "kind": "recommendation", "sketch": "your call, talk it through with your doctor", "support": "none_needed", "confidence": 0.97, "retrieve": None, "anchor": "..."},
    ]},
    "quality": {"score": 5, "issues": []},
}

EXAMPLE_NOTE = (
    "## Illustrative labeling (shape only; anchors elided as '...')\n"
    f"CONTEXT: {EXAMPLE_CONTEXT}\nLABELS: {json.dumps(EXAMPLE_LABELS, ensure_ascii=False)}"
)

TOOL = {
    "name": "submit_labels",
    "description": "Submit the Ladder-of-Abstraction labels for the TARGET.",
    "input_schema": Labels.model_json_schema(),
}


def render_user(ex: dict) -> str:
    parts = ["<context>"]
    for m in ex["context"]:
        parts.append(f"<{m['role']}>\n{m['content']}\n</{m['role']}>")
    parts.append("</context>")
    if ex.get("reference_answer"):
        parts.append(f"<reference_answer>{ex['reference_answer']}</reference_answer>")
    if ex.get("revision"):
        parts.append(f"<draft>\n{ex['revision']['draft']}\n</draft>")
        if ex["revision"].get("human_feedback"):
            parts.append(f"<human_feedback>\n{ex['revision']['human_feedback']}\n</human_feedback>")
    parts.append(f"<target>\n{ex['target']}\n</target>")
    parts.append(f"Domain: {ex['domain']}. Label the TARGET.")
    return "\n".join(parts)
