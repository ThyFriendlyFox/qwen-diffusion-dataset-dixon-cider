"""Ladder-of-Abstraction (LoA) dataset tooling.

Rungs (unmasked in this order by the diffusion planner):
    1. intent     - ideas, emotions, actions, audience
    2. structure  - outline + grammatical glue
    3. sketch     - section-level content sketches in block slots
followed by the final text, which the AR verbalizer produces.
"""

PROMPT_VERSION = "loa-v1"
