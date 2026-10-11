"""Teacher labeling via the Claude API (sync with a thread pool, or Message Batches at 50% cost)."""

from __future__ import annotations

import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Iterable, Iterator

from . import PROMPT_VERSION
from .prompt import EXAMPLE_NOTE, SYSTEM, render_user
from .schema import Labels
from .validate import check

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
_UNSUPPORTED = {"minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                "minItems", "maxItems", "pattern", "default", "title", "multipleOf"}


def output_schema() -> dict:
    """Labels JSON schema reduced to what structured outputs accept. The removed
    constraints are re-checked client-side by validate.check (full pydantic)."""

    def clean(node):
        if isinstance(node, dict):
            node = {k: clean(v) for k, v in node.items() if k not in _UNSUPPORTED}
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            return node
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    return clean(copy.deepcopy(Labels.model_json_schema()))


SCHEMA = output_schema()


def build_params(ex: dict, model: str, effort: str, max_tokens: int = 16000) -> dict:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": SYSTEM + "\n\n" + EXAMPLE_NOTE, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": render_user(ex)}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
    }


def _finish(ex: dict, msg, model: str) -> dict:
    meta = {"teacher": model, "prompt_version": PROMPT_VERSION, "labeled_at": datetime.now(timezone.utc).isoformat()}
    if msg.stop_reason in ("refusal", "max_tokens"):
        return {"id": ex["id"], "ok": False, "errors": [f"stop_reason={msg.stop_reason}"]}
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        labels = json.loads(text)
    except json.JSONDecodeError as e:
        return {"id": ex["id"], "ok": False, "errors": [f"json: {e}"]}
    meta["usage"] = {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens,
                     "cache_read": getattr(msg.usage, "cache_read_input_tokens", 0) or 0}
    row, errors, warnings = check(ex, labels, meta)
    if row is None:
        return {"id": ex["id"], "ok": False, "errors": errors, "labels": labels}
    return {"id": ex["id"], "ok": True, "row": row.model_dump(mode="json")}


def label_sync(examples: Iterable[dict], model: str = DEFAULT_MODEL, effort: str = "medium",
               workers: int = 8, retries: int = 1) -> Iterator[dict]:
    """Label examples concurrently. Rows that fail validation are retried `retries` times."""
    import anthropic

    client = anthropic.Anthropic()

    def one(ex):
        out = None
        for _ in range(retries + 1):
            msg = client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **build_params(ex, model, effort))
            out = _finish(ex, msg, model)
            if out["ok"]:
                break
        return out

    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(one, ex): ex for ex in examples}
        for f in as_completed(futs):
            try:
                yield f.result()
            except Exception as e:  # keep going; record the failure
                yield {"id": futs[f]["id"], "ok": False, "errors": [f"{type(e).__name__}: {e}"]}


def submit_batch(examples: list[dict], model: str = DEFAULT_MODEL, effort: str = "medium") -> str:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    reqs = [Request(custom_id=f"r{i}", params=MessageCreateParamsNonStreaming(**build_params(ex, model, effort)))
            for i, ex in enumerate(examples)]
    return client.messages.batches.create(requests=reqs).id


def collect_batch(batch_id: str, examples: list[dict], model: str = DEFAULT_MODEL, poll: int = 60) -> Iterator[dict]:
    import anthropic

    client = anthropic.Anthropic()
    while client.messages.batches.retrieve(batch_id).processing_status != "ended":
        time.sleep(poll)
    for res in client.messages.batches.results(batch_id):
        ex = examples[int(res.custom_id[1:])]  # results arrive in any order; key by custom_id
        if res.result.type == "succeeded":
            yield _finish(ex, res.result.message, model)
        else:
            yield {"id": ex["id"], "ok": False, "errors": [f"batch result: {res.result.type}"]}
