import copy
import random

from loa.align import align
from loa.canvas import mask_rate, render, sample_mask
from loa.prompt import EXAMPLE_LABELS
from loa.sources import norm_glaive, norm_gsm8k, _last_boxed
from loa.teacher import SCHEMA
from loa.validate import check

TARGET = (
    "It's completely understandable to feel nervous. Lots of people did.\n\n"
    "Here's what actually happens: the mRNA is read by ribosomes,\n   then degrades.\n\n"
    "It never enters the nucleus."
)


def _labels():
    lab = copy.deepcopy(EXAMPLE_LABELS)
    lab["structure"]["outline"] = lab["structure"]["outline"][:2]
    lab["sketch"]["blocks"] = [
        {**lab["sketch"]["blocks"][0], "anchor": "It's completely understandable"},
        {**lab["sketch"]["blocks"][1], "anchor": "Here's what actually happens: the mRNA is read by ribosomes, then"},  # whitespace differs
        {**lab["sketch"]["blocks"][2], "anchor": "It never enters the nucleus."},
    ]
    return lab


EX = {"id": "t:1", "source": "test", "domain": "chat", "license": "mit",
      "context": [{"role": "user", "content": "Are mRNA vaccines safe?"}], "target": TARGET}


def test_align_contiguous_with_whitespace_normalization():
    spans, errs = align(TARGET, _labels()["sketch"]["blocks"])
    assert not errs
    assert spans[0]["start"] == 0 and spans[-1]["end"] == len(TARGET)
    assert all(a["end"] == b["start"] for a, b in zip(spans, spans[1:]))
    assert TARGET[spans[1]["start"]:].startswith("Here's what")


def test_align_out_of_order_fails():
    blocks = _labels()["sketch"]["blocks"]
    blocks[1], blocks[2] = blocks[2], blocks[1]
    assert align(TARGET, blocks)[1]


def test_check_accepts_and_fills_evidence():
    row, errs, _ = check(EX, _labels())
    assert row and not errs
    assert len(row.alignment) == 3 and row.evidence == []


def test_check_rejects_bad_retrieve_and_section():
    lab = _labels()
    lab["sketch"]["blocks"][1].update(support="retrieve", retrieve=None, section="S9")
    row, errs, _ = check(EX, lab)
    assert row is None and any("no query" in e for e in errs) and any("S9" in e for e in errs)


def test_check_rejects_anchor_in_tool_response():
    ex = {**EX, "domain": "agentic", "target": '<tool_call name="x">{}</tool_call>\n\n<tool_response>{"a": 1} done</tool_response>\n\nAll done.'}
    lab = _labels()
    lab["sketch"]["blocks"] = [
        {**lab["sketch"]["blocks"][0], "kind": "tool_call", "anchor": '<tool_call name="x">'},
        {**lab["sketch"]["blocks"][1], "anchor": '{"a": 1} done'},
    ]
    row, errs, _ = check(ex, lab)
    assert row is None and any("tool_response" in e for e in errs)


def test_canvas_slots_and_spans():
    row, _, _ = check(EX, _labels())
    c = render(row.model_dump(mode="json"), slots=8)
    assert c["canvas"].count("NULL>") == 5
    for s in c["rung_spans"]:
        assert c["canvas"][s["start"]:s["end"]].strip()
    assert [s["rung"] for s in c["rung_spans"]] == ["intent", "structure", "sketch"]


def test_semantic_schedule_order():
    # forward process masks the concrete rung first, so the reverse process reveals intent first
    assert mask_rate("sketch", 0.5) == 1.0 and mask_rate("intent", 0.5) == 0.0
    assert mask_rate("intent", 1.0) == 1.0 and mask_rate("sketch", 0.0) == 0.0
    m = sample_mask({"intent": 50, "sketch": 50}, 0.5, random.Random(0))
    assert not any(m["intent"]) and all(m["sketch"])


def test_gsm8k_strips_calculator_annotations():
    ex = norm_gsm8k({"question": "q", "answer": "48/2 = <<48/2=24>>24\n#### 24"}, None)
    assert "<<" not in ex["target"] and ex["reference_answer"] == "24"


def test_glaive_parses_tool_calls():
    row = {"system": "SYSTEM: tools", "chat": "USER: convert 100 USD\n\n\nASSISTANT: <functioncall> {\"name\": \"conv\", \"arguments\": '{\"amt\": 100}'} <|endoftext|>\n\n\nFUNCTION RESPONSE: {\"eur\": 92}\n\n\nASSISTANT: That's 92 EUR. <|endoftext|>"}
    ex = norm_glaive(row, None)
    assert '<tool_call name="conv">{"amt": 100}</tool_call>' in ex["target"]
    assert "<tool_response>" in ex["target"] and ex["target"].endswith("92 EUR.")


def test_last_boxed_nested():
    assert _last_boxed(r"so \boxed{\frac{1}{2}} and \boxed{x^{2}}") == "x^{2}"


def test_output_schema_is_structured_output_safe():
    def walk(n):
        if isinstance(n, dict):
            assert not {"minLength", "maximum", "minimum", "maxItems", "pattern"} & n.keys()
            if n.get("type") == "object" and "properties" in n:
                assert n["additionalProperties"] is False and set(n["required"]) == set(n["properties"])
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(SCHEMA)
