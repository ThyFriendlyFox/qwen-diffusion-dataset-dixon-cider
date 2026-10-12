# LoA dataset

Ladder-of-Abstraction labels for training a diffusion planner that hands off to an autoregressive (AR) verbalizer.

## The short version (ELI5)

When you write a school report, you don't start with the first word. First you decide what you want to say and who it's for. Then you make an outline. Then you scribble rough notes in each part of the outline. Only at the end do you write it out neatly.

Most chatbots skip all that. They write one word at a time from the very start, and they have to figure out the plan as they go.

This project splits the job between two models:

- A **small, fast planner** makes the plan first. It doesn't write left to right. It starts with a blank page and fills in the whole plan at once, a little more each step, the way a blurry picture comes into focus. The big ideas show up first, then the outline, then the rough notes.
- A **big, smart writer** turns that plan into the final answer. It also checks the plan as it goes, so a bad note from the planner doesn't get copied in blindly.

To learn to plan, the planner needs examples: lots of good answers, each with the plan behind it written out. That is what this repo makes. It takes answers from public datasets, asks a strong AI model to write down the plan behind each one, checks the result, and packs everything into training files.

Here is one real row from the dataset. The question is a word problem; each colored box is one rung of the plan.

![The four rungs of the plan for one math answer](docs/img/ladder.svg)

The planner learns to produce the colored boxes from top to bottom. During training, parts of the plan are hidden (grey) and the model learns to fill them back in. The most abstract rung is always filled in first:

![The plan canvas filling in over five steps](docs/img/denoise.svg)

And this is where each piece of the dataset goes in the full system:

![Planner, projector and writer, with the critic loop](docs/img/architecture.svg)

Words used below:

| Word | Meaning |
|---|---|
| Rung | One level of the plan: intent, structure or sketch. |
| Block | One slot in the sketch, about a paragraph, a step or a tool call. |
| Target | The finished answer from the source dataset. |
| Teacher | The strong model that writes the plan for each target. |
| Canvas | The plan laid out as text with a fixed number of block slots, so the planner always sees the same shape. |
| Critic | The part that notices a wrong block and redoes it. |

## How it works

Each row pairs a response from a public dataset (`target`) with the plan behind it, as labeled by a teacher model. The plan has three rungs, which the diffusion model unmasks in order:

1. **Intent**: ideas, emotions, actions, audience. No wording.
2. **Structure**: format, plus an outline S1..Sn where each section has its grammatical glue. No content.
3. **Sketch**: up to 16 block slots of rough notes, one per paragraph, step or tool call. Each block has a kind, a support label, a confidence, an optional search query, and an anchor that ties it to a span of `target`.

The AR model then writes the final text from the plan.

## Layout

| Path | Contents |
|---|---|
| `loa/schema.py` | Pydantic schema for a row. |
| `loa/prompt.py` | Teacher system prompt and the mRNA example. The example only shows the format and is never written as a row. |
| `loa/sources.py` | Hugging Face sources and per-source normalizers. Rows come from the datasets-server API, so nothing is downloaded in bulk. |
| `loa/teacher.py` | Claude API labeler: structured JSON output, cached system prompt, sync (thread pool) or Message Batches. |
| `loa/align.py` | Resolves block anchors to character spans of `target`. |
| `loa/validate.py` | Rejects or warns on bad labels. |
| `loa/canvas.py` | Fixed-slot block canvas and the semantic noise schedule. |
| `loa/cli.py` | `fetch`, `label`, `build`, `check`. |
| `scripts/make_figures.py` | Regenerates the images in `docs/img/` from the seed set. |
| `mix.json` | Rows per source for the full run (about 26k). |
| `data/seed/seed.jsonl` | Gold seed set: 37 labeled and validated rows. |
| `data/seed/pool.jsonl` | The unlabeled rows the seed set was drawn from. |

![Build pipeline from sources to splits](docs/img/pipeline.svg)

## Row format

```jsonc
{
  "id": "gsm8k:72bbd7e7e2ca", "source": "openai/gsm8k", "domain": "math", "license": "mit",
  "context": [{"role": "user", "content": "..."}],   // conversation so far; agentic rows include a tool catalog
  "target": "...",                                    // text the AR model must produce
  "reference_answer": "85",                           // math and science only
  "labels": {
    "intent":    {"ideas": [...], "emotions": [...], "actions": [...], "audience": [...]},
    "structure": {"form": "...", "outline": [{"id": "S1", "purpose": "...", "glue": "none (opening)"}]},
    "sketch":    {"blocks": [{"id": "B1", "section": "S1", "kind": "computation",
                              "sketch": "Tamika's distance = hrs x mph",
                              "support": "derived", "confidence": 0.99, "retrieve": null,
                              "anchor": "Tamika drove 8 hours * 45 mph"}]},
    "quality":   {"score": 5, "issues": [{"block": "B3", "rung": "sketch", "issue": "...", "fix": "..."}]}
  },
  "alignment": [{"block": "B1", "start": 0, "end": 42}],   // contiguous spans covering target
  "evidence":  [{"block": "B4", "query": "...", "passage": null}],   // filled by a later retrieval pass
  "revision":  {"draft": "...", "human_feedback": "...", "critiques": [...]},   // critic rows only
  "meta": {"teacher": "...", "prompt_version": "loa-v1", "gold": true, "warnings": []}
}
```

`kind`: interpersonal, framing, definition, explanation, factual_claim, derivation, computation, example, procedure_step, code, tool_call, tool_result_use, caveat, recommendation, question_to_user, answer, summary.

`support`: none_needed, supported, retrieve, derived, context, tool, uncertain, unsupported. Together with `confidence`, this lets the AR model weigh each claim in the plan rather than accept it as given.

Agentic rows use the whole assistant episode as the target: assistant text, `<tool_call name="...">{args}</tool_call>` and `<tool_response>...</tool_response>`. Tool responses are observations. Blocks can't be anchored inside them, and they should be loss-masked when training the AR model.

## Training views

`build` adds a `canvas` string to every row, plus `rung_spans` (character offsets you can map to tokens with the tokenizer's offset mapping):

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
</structure>
<sketch>
<B1 S1 interpersonal none_needed 1.00> nervousness is normal
<B4 S3 factual_claim retrieve 0.85> trials had tens of thousands ... [RETRIEVE: mRNA vaccine phase 3 enrollment]
<B8 NULL>
</sketch>
```

![Mask rate by rung as noise rises](docs/img/schedule.svg)

Noise schedule (`canvas.mask_rate`): each rung has a noise band. Sketch is `[0, 0.4]`, structure `[0.3, 0.7]`, intent `[0.6, 1]`. The forward process masks the sketch first, so denoising reveals intent, then structure, then sketch. The bands overlap so neighboring rungs can condition on each other.

Projector pairs: `alignment` maps each block to its span of `target`. These (block, span) pairs are the training data for the 2-layer MLP adapter.

![Sketch blocks matched to the text they became](docs/img/alignment.svg)

## Sources

| Name | HF dataset | Domain | License | Rows | Notes |
|---|---|---|---|---|---|
| ultrachat | HuggingFaceH4/ultrachat_200k | chat | MIT | 6000 | Random assistant turn, earlier turns as context |
| gsm8k | openai/gsm8k | math | MIT | 2500 | Calculator annotations stripped |
| numina | AI-MO/NuminaMath-CoT | math | Apache-2.0 | 4000 | Reference answer is the last `\boxed{}` |
| osr2 | nvidia/OpenScienceReasoning-2 | science | CC-BY-4.0 | 3000 | Text after `</think>` only |
| webinstruct | TIGER-Lab/WebInstructSub | science | Apache-2.0 | 2500 | STEM Q&A |
| toucan | Agent-Ark/Toucan-1.5M (SFT) | agentic | Apache-2.0 | 4000 | MCP tool-use trajectories |
| glaive | glaiveai/glaive-function-calling-v2 | agentic | Apache-2.0 | 1500 | Only rows with a tool call |
| helpsteer3_edit | nvidia/HelpSteer3 (edit) | critic | CC-BY-4.0 | 1500 | Human feedback, then a human edit |
| iterater | wanyu/IteraTeR_full_doc | critic | Apache-2.0 | 1000 | Author revisions from arXiv, Wikipedia, news |

All licenses allow commercial use. `camel-ai/*` and `allenai/sciq` are excluded because they are CC-BY-NC.

## Critic data

The critic stage needs drafts, critiques and revisions written by people. Options, in order of usefulness:

1. HelpSteer3 `edit` (included). Annotators wrote feedback on a model response, then edited it. Covers code, STEM, general and multilingual.
2. IteraTeR (included). Revision history from arXiv, Wikipedia and Wikinews, with edit-intent labels. Edits are small and mostly surface-level.
3. PRM800K and Math-Shepherd (not yet included). Step-level correctness labels on math solutions, which map to re-masking a wrong block. The PRM800K viewer on the HF mirror was broken at the time of writing; use the openai/prm800k GitHub repo instead.
4. `bigcode/commitpackft` (not yet included, MIT). The commit message is the critique; before and after are the draft and revision.
5. Later: arXiv v1 to vN diffs, StackExchange edit history (CC-BY-SA), and Wikipedia revision history for larger structural edits.

In critic mode the teacher labels the revised text, then records each change from the draft as `{block, rung, issue, fix}`. That says which rung was wrong, which is the signal for re-masking a block during diffusion. Regular rows contribute too: when the teacher finds an error in a source response, it marks the block `unsupported` and adds a critique.

## Seed set

`data/seed/seed.jsonl` has 37 rows, all passing the validator.

![Seed set by domain, quality and block support](docs/img/seed.svg)

| | |
|---|---|
| Mix by source | 6 chat, 8 math, 8 science, 8 agentic, 7 critic (the chart groups critic rows by topic) |
| Blocks per row | 1 to 8, mean 4.2 |
| Critiques | 77 total; 10 blocks marked `unsupported` |
| Quality | Mean 3.7; 31 rows score 3 or higher |
| `retrieve` blocks | 2 of 154 |

Source errors found during labeling (all filtered out by `build`):

- WebInstruct determinant: the source gives -2/9; the correct value is 1.
- WebInstruct Taylor series: wrong third derivative.
- Toucan time-zone answer: contradicts its own tool output.
- IteraTeR: one revision is page spam, another is vandalism ("disproved" changed to "proved").

WebInstruct and IteraTeR are the noisiest sources. `norm_iterater` now drops revisions with draft similarity below 0.6, and revisions under 200 characters. HelpSteer3-edit gives the most critiques (2 to 6 per row), but its change summaries need checking: one was wrong, and one revision introduced a new error.

Few rows need `retrieve`, because most source responses don't state exact figures. If the grounding step needs more of them, add sources with citations.

## Usage

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...

python -m loa.cli fetch --mix mix.json --out data/raw/pool.jsonl
python -m loa.cli label --in data/raw/pool.jsonl --out data/labeled/rows.jsonl --limit 200     # pilot
python -m loa.cli label --in data/raw/pool.jsonl --out data/labeled/rows.jsonl --batch         # submit batch
python -m loa.cli label --in data/raw/pool.jsonl --out data/labeled/rows.jsonl --batch-id <id> # collect batch
python -m loa.cli build --in data/labeled/rows.jsonl data/seed/seed.jsonl --out data/release
python -m pytest -q tests
python scripts/make_figures.py      # redraw docs/img after the seed set changes
```

`label` skips ids already in `--out`, so it can be resumed. Rejected rows go to `<out>.rejects` with their errors.

`build` removes duplicates (same context and target) and drops rows with quality below 3. It splits the rest into train, validation and test by a hash of the id (2% validation, 2% test). Seed rows go to their own `seed` split.

Teacher model: `claude-opus-5-5` at effort `medium` by default. Sync calls set `fallbacks: "default"`, so a refused request is retried on a fallback model. Estimated cost is $0.05 to $0.08 per row sync and about half that in batch mode, or $1k to $2k for the full mix. Check `meta.usage` from the pilot before relying on that. `--model claude-sonnet-5-5` costs half as much; compare its labels with the seed set first.

## Next steps

1. Run the 200-row pilot and compare its labels with the seed set (block counts, kind and support agreement, confidence calibration). Update `prompt.py` and bump `PROMPT_VERSION` as needed.
2. Add a retrieval pass to fill `evidence[].passage` for `retrieve` blocks.
3. Add PRM800K and CommitPackFT as critic sources.
