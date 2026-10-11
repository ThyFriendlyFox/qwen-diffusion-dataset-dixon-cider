"""Pydantic schema for one labeled LoA row.

A row = (context, target) from a source dataset + the teacher's three rungs.
`target` is the text the AR model must produce; the rungs are the plan the
diffusion model must produce. Block anchors tie each sketch block to a
contiguous span of `target` so the projector can be trained on (block, span)
pairs.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Domain = Literal["chat", "agentic", "math", "science", "code", "writing"]

BlockKind = Literal[
    "interpersonal",    # acknowledgement, empathy, rapport
    "framing",          # restating the problem, setting up what follows
    "definition",
    "explanation",
    "factual_claim",
    "derivation",       # symbolic reasoning step
    "computation",      # arithmetic / numeric step
    "example",
    "procedure_step",   # instructions the user should follow
    "code",
    "tool_call",        # the assistant invokes a tool
    "tool_result_use",  # the assistant reads / interprets a tool result
    "caveat",
    "recommendation",
    "question_to_user",
    "answer",           # the final answer / bottom line
    "summary",
]

Support = Literal[
    "none_needed",  # interpersonal / stylistic, nothing to verify
    "supported",    # well-established knowledge, no lookup required
    "retrieve",     # true-ish but needs an exact figure/citation -> RETRIEVE
    "derived",      # follows from earlier blocks by reasoning or arithmetic
    "context",      # grounded in user-provided context
    "tool",         # grounded in a tool result in the context
    "uncertain",    # plausible but the teacher cannot confirm
    "unsupported",  # the target asserts something wrong / unfounded (critic signal)
]


class Intent(BaseModel):
    ideas: list[str] = Field(min_length=1, description="Core ideas/claims the response must convey, as short propositions.")
    emotions: list[str] = Field(min_length=1, description="Emotional stance / tone toward the reader.")
    actions: list[str] = Field(min_length=1, description="Speech acts / goals: inform, persuade, solve, call tools, ask, refuse...")
    audience: list[str] = Field(min_length=1, description="Who the reader is and what they are assumed to know.")


class Section(BaseModel):
    id: str = Field(pattern=r"^S\d+$")
    purpose: str = Field(description="What this section does, in a few words.")
    glue: str = Field(description="How it attaches to what precedes it: the connective/transition move, e.g. 'contrast: That said,' or 'none (opening)'.")


class Structure(BaseModel):
    form: str = Field(description="Surface form: prose/bullets/numbered steps/markdown headers/LaTeX/code blocks/tool-call sequence, register, approximate length.")
    outline: list[Section] = Field(min_length=1)


class Block(BaseModel):
    id: str = Field(pattern=r"^B\d+$")
    section: str = Field(pattern=r"^S\d+$")
    kind: BlockKind
    sketch: str = Field(description="Rough note-to-self for this slot, NOT polished prose. e.g. 'couple of supporting claims about cat reflexes here'.")
    support: Support
    confidence: float = Field(ge=0.0, le=1.0)
    retrieve: Optional[str] = Field(default=None, description="Search query if support == 'retrieve', else null.")
    anchor: str = Field(description="Verbatim first 4-12 words of the target span this block covers (copied exactly from target).")

    @field_validator("retrieve")
    @classmethod
    def _strip(cls, v):
        return v.strip() or None if isinstance(v, str) else v


class Sketch(BaseModel):
    blocks: list[Block] = Field(min_length=1, max_length=16)


class Critique(BaseModel):
    block: Optional[str] = Field(default=None, description="Block id the issue lives in (null = global).")
    rung: Literal["intent", "structure", "sketch", "surface"]
    issue: str
    fix: str


class Quality(BaseModel):
    score: int = Field(ge=1, le=5, description="5 = exemplary target text, 1 = should be discarded.")
    issues: list[Critique] = Field(default_factory=list)


class Labels(BaseModel):
    """What the teacher model returns."""

    intent: Intent
    structure: Structure
    sketch: Sketch
    quality: Quality


class Revision(BaseModel):
    """Critic-stage supervision: a human draft and how the plan changed to get to `target`."""

    draft: str
    critiques: list[Critique] = Field(default_factory=list)
    human_feedback: Optional[str] = None


class Message(BaseModel):
    role: str
    content: str


class Evidence(BaseModel):
    block: str
    query: str
    passage: Optional[str] = None
    source: Optional[str] = None


class Span(BaseModel):
    block: str
    start: int
    end: int


class Row(BaseModel):
    id: str
    source: str
    domain: Domain
    license: str
    context: list[Message]
    target: str
    reference_answer: Optional[str] = None
    labels: Labels
    alignment: list[Span] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    revision: Optional[Revision] = None
    meta: dict = Field(default_factory=dict)
