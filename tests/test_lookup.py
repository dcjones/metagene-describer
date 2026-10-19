import json
from types import SimpleNamespace as NS

import numpy as np

from metagene_describer import AnthropicBackend, OpenAIBackend, Tool, describe_metagenes
from metagene_describer.lookup import GeneLookup

GENES = ["CD3E", "CD3D", "MS4A1", "CD79A", "ACTB"]


def make_H():
    H = np.zeros((2, len(GENES)))
    H[0, [0, 1]] = [5, 4]
    H[1, [2, 3]] = [6, 5]
    H[:, 4] = 10
    return H


def test_describe_lookup():
    lk = GeneLookup(GENES, make_H(), None)
    out = lk.describe(0, ["MS4A1", "cd3e", "PECAM1"]).splitlines()
    assert out[0].startswith("MS4A1: rank") and "most specific to metagene 1" in out[0]
    assert "its top genes: MS4A1, CD79A" in out[0]
    assert out[1].startswith("CD3E:") and "most specific to this metagene" in out[1]
    assert out[2] == "PECAM1: not measured in this dataset"


class FakeAnthropic:
    """Returns a tool_use turn first, then a JSON answer; records requests."""

    def __init__(self):
        self.requests = []
        self.messages = self

    def create(self, **kw):
        self.requests.append(kw)
        if len(self.requests) == 1:
            content = [NS(type="tool_use", id="t1", name="lookup_genes", input={"genes": ["MS4A1"]})]
            return NS(stop_reason="tool_use", content=content)
        return NS(stop_reason="end_turn", content=[NS(type="text", text='{"ok": 1}')])


def test_anthropic_tool_loop():
    client = FakeAnthropic()
    calls = []
    tool = Tool("lookup_genes", "", {}, lambda a: calls.append(a) or "result")
    out = AnthropicBackend(client=client).complete_json("sys", "user", {}, tools=[tool])
    assert out == {"ok": 1} and calls == [{"genes": ["MS4A1"]}]
    second = client.requests[1]["messages"]
    assert second[1]["role"] == "assistant" and second[2]["content"][0]["tool_use_id"] == "t1"
    assert "tool_choice" not in client.requests[1]


def test_anthropic_tool_rounds_capped():
    class Always(FakeAnthropic):
        def create(self, **kw):
            self.requests.append(kw)
            if kw.get("tool_choice") == {"type": "none"}:
                return NS(stop_reason="end_turn", content=[NS(type="text", text='{"ok": 2}')])
            content = [NS(type="tool_use", id="t", name="lookup_genes", input={"genes": []})]
            return NS(stop_reason="tool_use", content=content)

    client = Always()
    tool = Tool("lookup_genes", "", {}, lambda a: "")
    out = AnthropicBackend(client=client, max_tool_rounds=2).complete_json("s", "u", {}, [tool])
    assert out == {"ok": 2} and len(client.requests) == 3


class FakeOpenAI:
    def __init__(self):
        self.requests = []
        self.chat = NS(completions=self)

    def create(self, **kw):
        self.requests.append(kw)
        if len(self.requests) == 1:
            call = NS(id="c1", function=NS(name="lookup_genes", arguments='{"genes": ["CD3E"]}'))
            msg = NS(tool_calls=[call], content=None, refusal=None,
                     model_dump=lambda **_: {"role": "assistant", "tool_calls": []})
            return NS(choices=[NS(message=msg, finish_reason="tool_calls")])
        msg = NS(tool_calls=None, content='{"ok": 3}', refusal=None)
        return NS(choices=[NS(message=msg, finish_reason="stop")])


def test_openai_tool_loop():
    client = FakeOpenAI()
    tool = Tool("lookup_genes", "", {}, lambda a: "looked up " + ",".join(a["genes"]))
    out = OpenAIBackend("m", client=client).complete_json("s", "u", {}, tools=[tool])
    assert out == {"ok": 3}
    msgs = client.requests[1]["messages"]
    assert msgs[-1] == {"role": "tool", "tool_call_id": "c1", "content": "looked up CD3E"}


class LookupBackend:
    def complete_json(self, system, user, schema, tools=()):
        assert "lookup_genes" in system
        text = tools[0].run({"genes": ["MS4A1", "CD3E"]})
        assert "MS4A1" in text
        return {"description": "", "key_genes": ["CD3E", "MS4A1", "FAKE"], "category": "cell_type",
                "cell_types": [], "confidence": "high", "label": "T cells"}


def test_describe_with_lookup():
    out = describe_metagenes(GENES, make_H(), backend=LookupBackend(), lookup=True, metagenes=[0])
    assert out[0].looked_up == ["MS4A1", "CD3E"]
    assert out[0].key_genes == ["CD3E", "MS4A1"]  # looked-up genes are allowed, unknown dropped
