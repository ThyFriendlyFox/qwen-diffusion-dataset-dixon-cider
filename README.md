# LoA dataset: Ladder-of-Abstraction labels for a diffusion planner + AR verbalizer

This is the training data for the "LoA Trained Diffuse Projection onto AR Verbalizer" project. Each row pairs a real response (`target`) with the plan a frontier teacher model recovered from it. The plan has three rungs, and the diffusion model unmasks them in this order:

```
Intent      ideas · emotions · actions · audience                 (no wording)
  │
Structure   form + outline S1..Sn, each with its grammatical glue (no content)
  │
Sketch      block slots B1..B16: rough notes-to-self per section  (no polish)
  │            + kind, support, confidence, RETRIEVE query, anchor → span of target
  ▼
Target      final text, produced by the AR verbalizer
```

## What's in the repo

| Path | What it is |
|---|---|
| `loa/schema.py` | Pydantic schema for a row (rungs, blocks, critiques, alignment). |
| `loa/prompt.py` | Teacher system prompt (the labeling spec) and the mRNA example. The example is shape-only and is never emitted as a row. |
| `loa/sources.py` | Hugging Face source registry and per-source normalizers. Fetching uses the datasets-server API, so nothing is downloaded in bulk. |
| `loa/teacher.py` | Claude API labeler. It returns structured JSON output, caches the system prompt, and runs either sync with a thread pool or through Message Batches (50% cost). |
| `loa/align.py`, `loa/validate.py` | Resolve each block anchor to a char span of `target`, then reject or warn on bad labels. |
| `loa/canvas.py` | Fixed-slot block canvas (NULL-padded) and the semantic noise schedule for the diffusion trainer. |
| `loa/cli.py` | `fetch` → `label` → `build` (plus `check` to re-validate). |
| `mix.json` | Rows per source for the full run, about 26k rows. |
| `data/seed/seed.jsonl` | **Gold seed set**: real source rows labeled by a frontier model and validated. Use it for few-shot examples, for auditing the bulk teacher, and as the first benchmark. |
| `data/seed/pool.jsonl` | The unlabeled source rows the seed set came from. |

## One row

```jsonc
{
  "id": "gsm8k:72bbd7e7e2ca", "source": "openai/gsm8k", "domain": "math", "license": "mit",
  "context": [{"role": "user", "content": "..."}],          // conversation so far (incl. tool catalog for agentic)
  "target": "...",                                           // what the AR model must produce
  "reference_answer": "85",                                  // math/science only
  "labels": {
    "intent":    {"ideas": [...], "emotions": [...], "actions": [...], "audience": [...]},
    "structure": {"form": "...", "outline": [{"id": "S1", "purpose": "...", "glue": "none (opening)"}, ...]},
    "sketch":    {"blocks": [{"id": "B1", "section": "S1", "kind": "computation",
                              "sketch": "Tamika's distance = hrs x mph",
                              "support": "derived", "confidence": 0.99, "retrieve": null,
                              "anchor": "Tamika drove 8 hours * 45 mph"}, ...]},
    "quality":   {"score": 5, "issues": [{"block": "B3", "rung": "sketch", "issue": "...", "fix": "..."}]}
  },
  "alignment": [{"block": "B1", "start": 0, "end": 42}, ...], // contiguous spans covering target
  "evidence":  [{"block": "B4", "query": "...", "passage": null}],   // RETRIEVE slots, filled by a later retrieval pass
  "revision":  {"draft": "...", "human_feedback": "...", "critiques": [...]},  // critic-stage rows only
  "meta": {"teacher": "...", "prompt_version": "loa-v1", "gold": true, "warnings": []}
}
```

`kind` is one of: interpersonal, framing, definition, explanation, factual_claim, derivation, computation, example, procedure_step, code, tool_call, tool_result_use, caveat, recommendation, question_to_user, answer, summary.

`support` is one of: none_needed, supported, retrieve, derived, context, tool, uncertain, unsupported. Each block carries this, plus a calibrated `confidence`. That's the signal the AR model uses to rank the plan's claims instead of trusting them blindly.

**Agentic rows** serialize the whole assistant episode as the target: assistant text, `<tool_call name=…>{args}</tool_call>`, and `<tool_response>…</tool_response>`. Tool responses are observations. No block may be anchored inside one, and they should be loss-masked for the AR model.

## Training views produced by `build`

`build` adds a `canvas` field to every row, plus `rung_spans` (char offsets) for mapping to tokens through the tokenizer's offset mapping:

```
<intent>
ideas: ... | ...
emotions: ...
actions: ...
audience: ...
</intent>
<structure>
form: ...
S1 [none (opening)] validate the concern
...
</structure>
<sketch>
<B1 S1 interpersonal none_needed 1.00> nervousness is normal
<B4 S3 factual_claim retrieve 0.85> trials had tens of thousands ... [RETRIEVE: mRNA vaccine phase 3 enrollment]
<B8 NULL>
...
</sketch>
```

**Semantic noise schedule** (`canvas.mask_rate`): each rung has a noise band (sketch `[0, .4]`, structure `[.3, .7]`, intent `[.6, 1]`). The forward process masks the concrete rung first, so the reverse process reveals intent, then structure, then sketch. You can see this on the block canvas, as in diffusion-gemma. The bands overlap slightly so the rungs can condition on each other.

**Projector pairs**: `alignment` maps each sketch block to its exact span of `target`. Those (block, span) pairs are what the 2-layer MLP adapter trains on: pooled block latent → AR conditioning for that span.

## Sources (`mix.json`)

| Name | HF dataset | Domain | License | Rows | Notes |
|---|---|---|---|---|---|
| ultrachat | HuggingFaceH4/ultrachat_200k | chat | MIT | 6000 | Random assistant turn, with prior turns as context. |
| gsm8k | openai/gsm8k | math | MIT | 2500 | Calculator annotations stripped; `reference_answer` kept. |
| numina | AI-MO/NuminaMath-CoT | math | Apache-2.0 | 4000 | Competition math; reference is the last `\boxed{}`. |
| osr2 | nvidia/OpenScienceReasoning-2 | science | CC-BY-4.0 | 3000 | Post-`</think>` answer only. |
| webinstruct | TIGER-Lab/WebInstructSub | science | Apache-2.0 | 2500 | Socratic/StackExchange STEM Q&A. |
| toucan | Agent-Ark/Toucan-1.5M (SFT) | agentic | Apache-2.0 | 4000 | Real MCP-server tool trajectories. |
| glaive | glaiveai/glaive-function-calling-v2 | agentic | Apache-2.0 | 1500 | Only rows that actually call a tool. |
| helpsteer3_edit | nvidia/HelpSteer3 (edit) | critic | CC-BY-4.0 | 1500 | Human feedback, then a human edit. |
| iterater | wanyu/IteraTeR_full_doc | critic | Apache-2.0 | 1000 | Real author revisions (arXiv, Wikipedia, news). |

All of these are commercially usable. The `camel-ai/*` science sets and `allenai/sciq` were deliberately left out because they're CC-BY-NC.

## Critic data: drafts and revisions that aren't AI slop

The critic stage needs a real draft → critique → revision. These are the best human-sourced options I found:

1. **HelpSteer3 `edit`** (CC-BY-4.0, wired in): human annotators wrote feedback on a model response and then edited it. You get draft, critique and revision directly, across code, STEM, general and multilingual.
2. **IteraTeR** (Apache-2.0, wired in): real author revision history mined from arXiv versions, Wikipedia and Wikinews, with edit-intent labels (fluency, coherence, clarity, meaning-changed). These are human drafts, but the edits are small and mostly at the surface or sketch level.
3. **Step-level math critique** (not wired in yet): PRM800K (human step labels on model solutions; the HF mirror's viewer was broken when I checked, so pull it from openai/prd) and Math-Shepherd. These map directly onto "block Bk is wrong → re-mask it".
4. **Code revisions** (not wired in yet): `bigcode/commitpackft` (MIT). A commit message is the critique, and before/after is the revision.
5. **Planned**: arXiv v1→vN abstract and section diffs, StackExchange post-edit history (CC-BY-SA), and Wikipedia revision history for longer, structure-level edits.

In critic mode the teacher labels the revised target. It then writes each meaningful draft → target change as a critique `{block, rung, issue, fix}`, which records which rung was wrong. That's the supervision for "re-mask block B3 at the sketch rung" during diffusion. Ordinary plan rows also contribute critic signal: whenever the teacher finds an error in a source response, it marks the block `unsupported` and adds a critique.

## Gold seed set (`data/seed/seed.jsonl`)

There are 37 real source rows, labeled by a frontier model working directly against this spec. Every row passes the validator: anchors are verbatim and contiguous, with no sketch that copies its target. The labelers re-did the math and checked the facts rather than trusting the source.

- **Mix**: 6 UltraChat, 8 math (GSM8K, Numina), 8 science (OSR2, WebInstruct), 8 agentic (Toucan, Glaive), and 7 critic (HelpSteer3-edit, IteraTeR).
- **Shape**: 1–8 blocks per row (mean 4.2). There are 77 critiques in total, and 10 blocks are flagged `unsupported`.
- **Quality**: mean 3.7. 31 of 37 rows reach the default `build` floor (score ≥ 3).
- **Source errors the teacher caught**, all of which `build` filters out:
  - WebInstruct determinant: the source gives −2/9, but the true value is 1.
  - WebInstruct Taylor series: the source has the wrong f‴.
  - Toucan time-zone answer: it contradicts its own tool output.
  - IteraTeR: one revision is page spam, and another is vandalism that flips "disproved" to "proved".
- **What this says about the sources**:
  - WebInstruct and IteraTeR are the noisiest. `norm_iterater` now drops page replacements (draft/revision similarity < 0.6) and fragments.
  - HelpSteer3-edit is the richest critic source, with 2–6 critiques per row across all four rungs. Its human change summaries still need verifying, though: one was wrong, and one revision introduced a new error. The teacher records the revision as a regression critique in that case.
- **`retrieve` is rare** (2 of 154 blocks). Most source responses avoid exact figures. If the AR grounding step needs more RETRIEVE slots, add citation-heavy sources such as long-form science explanations.

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...

python -m loa.cli fetch --mix mix.json --out data/raw/pool.jsonl              # ~26k unlabeled rows
python -m loa.cli label --in data/raw/pool.jsonl --out data/labeled/rows.jsonl --limit 200   # pilot, sync
python -m loa.cli label --in data/raw/pool.jsonl --out data/labeled/rows.jsonl --batch       # bulk, 50% cost
python -m loa.cli label --in data/raw/pool.jsonl --out data/labeled/rows.jsonl --batch-id <id>  # collect
python -m loa.cli build --in data/labeled/rows.jsonl data/seed/seed.jsonl --out data/release
python -m pytest -q tests
```

`label` is resumable: it skips ids already in `--out`. Rejected rows go to `--out.rejects` with their errors. `build` dedups by context, drops non-critic rows with quality < 3, and splits train/validation/test by id hash (2% / 2%). Gold rows go in a separate `seed` split.

**Teacher model**: the default is `claude-opus-5-5` at effort `medium`, with `fallbacks: "default"` enabled on sync calls so a safety-classifier refusal reruns on a fallback model instead of failing. A rough estimate is $0.05–0.08 per row sync, about half that with `--batch`, so about $1–2k for the full mix. Run the 200-row pilot first and check `meta.usage` before trusting that number. `--model claude-sonnet-5-5` halves the price. Check its labels against the gold seed before switching.

## Suggested next steps

1. Run a 200-row pilot. Compare the teacher's labels against `data/seed/seed.jsonl` (block count, support/kind agreement, confidence calibration), then adjust `prompt.py` and bump `PROMPT_VERSION`.
2. Add a retrieval pass that fills `evidence[].passage` for RETRIEVE blocks. This is the "evidence bindings" step in the plan.
3. Add PRM800K and CommitPackFT as critic sources.
