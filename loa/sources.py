"""Source registry: pull rows from Hugging Face and normalize them into
unlabeled examples ready for the teacher.

An unlabeled example is a plain dict:
    id, source, domain, license, context [{role, content}], target,
    reference_answer?, revision? {draft, human_feedback}

Rows are fetched through the public datasets-server API, so nothing needs to be
downloaded in bulk and no `datasets` install is required.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import random
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

API = "https://datasets-server.huggingface.co"
MAX_TOOL_RESPONSE = 1500
MAX_TARGET = 6000
MAX_CONTEXT = 8000


def _get(path: str, **params) -> dict:
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _num_rows(dataset: str, config: str, split: str) -> int:
    for s in _get("size", dataset=dataset)["size"]["splits"]:
        if s["config"] == config and s["split"] == split:
            return s["num_rows"]
    raise KeyError(f"{dataset}/{config}/{split}")


def _sid(*parts) -> str:
    return hashlib.sha1("\x1f".join(map(str, parts)).encode()).hexdigest()[:12]


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + " ...[truncated]"


def _tool_call(name: str, args) -> str:
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return f'<tool_call name="{name}">{args}</tool_call>'


def _tool_response(content: str) -> str:
    return f"<tool_response>{_truncate(content.strip(), MAX_TOOL_RESPONSE)}</tool_response>"


# --------------------------------------------------------------------------- normalizers
# Each takes (row, rng) and returns an example dict without id/source/license, or None to skip.


def norm_ultrachat(row, rng):
    msgs = row["messages"]
    turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 0]
    if not turns:
        return None
    t = rng.choice(turns)
    return {
        "domain": "chat",
        "context": [{"role": m["role"], "content": m["content"]} for m in msgs[:t]],
        "target": msgs[t]["content"],
        "_key": f'{row["prompt_id"]}:{t}',
    }


def norm_gsm8k(row, rng):
    sol, _, ans = row["answer"].rpartition("####")
    sol = re.sub(r"<<[^>]*>>", "", sol).strip()
    ans = ans.strip()
    return {
        "domain": "math",
        "context": [{"role": "user", "content": row["question"]}],
        "target": f"{sol}\n\n**Answer: {ans}**",
        "reference_answer": ans,
        "_key": row["question"],
    }


def _last_boxed(s: str) -> Optional[str]:
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    k = j
    while k < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[k], 0)
        k += 1
    return s[j : k - 1] if depth == 0 else None


def norm_numina(row, rng):
    return {
        "domain": "math",
        "context": [{"role": "user", "content": row["problem"]}],
        "target": row["solution"],
        "reference_answer": _last_boxed(row["solution"]),
        "_key": row["problem"],
    }


def norm_osr2(row, rng):
    if "</think>" not in row["output"]:
        return None
    return {
        "domain": "science",
        "context": [{"role": "user", "content": row["input"]}],
        "target": row["output"].split("</think>")[-1].strip(),
        "reference_answer": row.get("expected_answer"),
        "_key": row["input"],
    }


def norm_webinstruct(row, rng):
    ans = re.sub(r"\n\s*\n+", "\n\n", row["answer"]).strip()
    if len(ans) < 200:
        return None
    return {
        "domain": "science",
        "context": [{"role": "user", "content": row["question"].strip()}],
        "target": ans,
        "_key": row["question"],
    }


_GLAIVE_SPLIT = re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)")


def norm_glaive(row, rng):
    parts = _GLAIVE_SPLIT.split(row["chat"])
    turns = []
    for tag, body in zip(parts[1::2], parts[2::2]):
        body = body.replace("<|endoftext|>", "").strip()
        turns.append((tag[:-1], body))
    if not turns or turns[0][0] != "USER":
        return None
    episode = []
    for tag, body in turns[1:]:
        if tag == "USER":
            break
        if tag == "ASSISTANT" and body.startswith("<functioncall>"):
            call = body[len("<functioncall>") :].strip()
            try:  # arguments are a single-quoted JSON string inside JSON
                obj = json.loads(re.sub(r"'(\{.*\})'", r"\1", call))
                episode.append(_tool_call(obj["name"], obj.get("arguments", {})))
            except Exception:
                episode.append(f"<tool_call>{call}</tool_call>")
        elif tag == "ASSISTANT":
            episode.append(body)
        else:
            episode.append(_tool_response(body))
    if not any("<tool_call" in e for e in episode):
        return None  # keep only rows that call a tool
    return {
        "domain": "agentic",
        "context": [
            {"role": "system", "content": row["system"].removeprefix("SYSTEM:").strip()},
            {"role": "user", "content": turns[0][1]},
        ],
        "target": "\n\n".join(episode),
        "_key": row["chat"][:2000],
    }


def _tool_catalog(tools_json: str, used: set[str]) -> str:
    out = []
    for t in json.loads(tools_json):
        f = t.get("function", t)
        name = f["name"]
        if used and not any(name.endswith(u) for u in used):
            continue
        out.append(f"- {name}: {_truncate((f.get('description') or '').strip(), 300)}\n  params: {json.dumps(f.get('parameters', {}).get('properties', {}))[:400]}")
    return "Available tools:\n" + "\n".join(out)


def norm_toucan(row, rng):
    msgs = json.loads(row["messages"])
    if not msgs or msgs[0]["role"] != "user":
        msgs = [m for m in msgs if m["role"] != "system"]
    if not msgs or msgs[0]["role"] != "user":
        return None
    episode, used = [], set()
    for m in msgs[1:]:
        if m["role"] == "user":
            break
        if m["role"] == "assistant":
            if m["content"].strip():
                episode.append(m["content"].strip())
        elif m["role"] == "tool_call":
            try:
                c = ast.literal_eval(m["content"])
                used.add(c["name"].split("-")[-1])
                episode.append(_tool_call(c["name"], c["arguments"]))
            except Exception:
                episode.append(f"<tool_call>{m['content']}</tool_call>")
        elif m["role"] == "tool_response":
            episode.append(_tool_response(m["content"]))
    if not any("<tool_call" in e for e in episode):
        return None
    return {
        "domain": "agentic",
        "context": [
            {"role": "system", "content": _tool_catalog(row["tools"], {u for u in (row.get("target_tools") or "").replace(" ", "").split(",") if u} | used)},
            {"role": "user", "content": msgs[0]["content"]},
        ],
        "target": "\n\n".join(episode),
        "_key": row["uuid"],
    }


_HS_DOMAIN = {"code": "code", "stem": "science", "general": "chat", "multilingual": "chat"}


def norm_helpsteer3_edit(row, rng):
    fb = row.get("feedback") or []
    fb = "\n".join(fb) if isinstance(fb, list) else str(fb)
    return {
        "domain": _HS_DOMAIN.get(row.get("domain"), "chat"),
        "context": [{"role": m["role"], "content": m["content"]} for m in row["context"]],
        "target": row["edited_response"].replace("\r\n", "\n").strip(),
        "revision": {
            "draft": row["original_response"].replace("\r\n", "\n").strip(),
            "human_feedback": (fb + ("\n\nChange summary: " + row["change_summary"] if row.get("change_summary") else "")).strip() or None,
        },
        "_key": row["original_response"][:2000],
    }


def norm_iterater(row, rng):
    before, after = row["before_revision"].strip(), row["after_revision"].strip()
    # drop no-ops, fragments, and page replacements/spam (revision unrelated to the draft)
    if before == after or len(after) < 200 or difflib.SequenceMatcher(None, before, after).ratio() < 0.6:
        return None
    return {
        "domain": "writing",
        "context": [{"role": "user", "content": "Write the opening paragraph of this document. (Source: author revision history; no original prompt.)"}],
        "target": row["after_revision"].strip(),
        "revision": {"draft": row["before_revision"].strip(), "human_feedback": None},
        "_key": f'{row["doc_id"]}:{row["revision_depth"]}',
    }


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class Source:
    name: str
    dataset: str
    config: str
    split: str
    license: str
    normalize: Callable
    stage: str = "plan"  # "plan" (rung labeling) or "critic" (has human drafts)
    note: str = ""


SOURCES: dict[str, Source] = {s.name: s for s in [
    # chat
    Source("ultrachat", "HuggingFaceH4/ultrachat_200k", "default", "train_sft", "mit", norm_ultrachat),
    # math
    Source("gsm8k", "openai/gsm8k", "main", "train", "mit", norm_gsm8k),
    Source("numina", "AI-MO/NuminaMath-CoT", "default", "train", "apache-2.0", norm_numina),
    # science
    Source("osr2", "nvidia/OpenScienceReasoning-2", "default", "train", "cc-by-4.0", norm_osr2),
    Source("webinstruct", "TIGER-Lab/WebInstructSub", "default", "train", "apache-2.0", norm_webinstruct),
    # agentic
    Source("toucan", "Agent-Ark/Toucan-1.5M", "SFT", "train", "apache-2.0", norm_toucan),
    Source("glaive", "glaiveai/glaive-function-calling-v2", "default", "train", "apache-2.0", norm_glaive),
    # critic stage: human drafts -> revisions
    Source("helpsteer3_edit", "nvidia/HelpSteer3", "edit", "train", "cc-by-4.0", norm_helpsteer3_edit, "critic",
           "Human annotators wrote feedback on a model response and then edited it."),
    Source("iterater", "wanyu/IteraTeR_full_doc", "default", "train", "apache-2.0", norm_iterater, "critic",
           "Real author revisions (arXiv / Wikipedia / news) mined from version history."),
]}


def sample(source: Source, n: int, seed: int = 0, page: int = 20) -> Iterator[dict]:
    """Yield up to n normalized examples drawn from random offsets of the split."""
    rng = random.Random(f"{seed}:{source.name}")
    total = _num_rows(source.dataset, source.config, source.split)
    seen, got, tries = set(), 0, 0
    while got < n and tries < n * 10 + 20:
        tries += 1
        off = rng.randrange(0, max(1, total - page))
        rows = _get("rows", dataset=source.dataset, config=source.config, split=source.split, offset=off, length=page)["rows"]
        rng.shuffle(rows)
        for r in rows[: max(1, page // 4)]:  # take a few per page for diversity
            ex = source.normalize(r["row"], rng)
            if ex is None:
                continue
            key = ex.pop("_key")
            ctx_len = sum(len(m["content"]) for m in ex["context"])
            if not ex["target"].strip() or len(ex["target"]) > MAX_TARGET or ctx_len > MAX_CONTEXT:
                continue
            ex_id = f"{source.name}:{_sid(source.dataset, key)}"
            if ex_id in seen:
                continue
            seen.add(ex_id)
            got += 1
            yield {"id": ex_id, "source": source.dataset, "license": source.license, "stage": source.stage, **ex}
            if got >= n:
                return
