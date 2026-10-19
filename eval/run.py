"""Label one example dataset and write the results as a TSV.

    uv run --group eval python eval/run.py DATASET OUT.tsv [--no-lookup] [--refine]
        [--backend anthropic|openai] [--model M] [--base-url URL] [--api-key-env VAR]
        [--metagenes 1,2,3]

Prints token usage and wall time (and cost, for known Anthropic models) to stderr.
"""

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))
from datasets import DATASETS, load  # noqa: E402

from metagene_describer import AnthropicBackend, OpenAIBackend, describe_metagenes, to_dataframe

PRICES = {  # $ per million input / output tokens
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
}


class Tally:
    """Wraps an SDK client's create() to total token usage across concurrent calls."""

    def __init__(self, create):
        self._create, self.lock = create, threading.Lock()
        self.calls = self.input = self.output = 0

    def create(self, **kw):
        r = self._create(**kw)
        u = r.usage
        with self.lock:
            self.calls += 1
            self.input += getattr(u, "input_tokens", None) or getattr(u, "prompt_tokens", 0) or 0
            self.output += (
                getattr(u, "output_tokens", None) or getattr(u, "completion_tokens", 0) or 0
            )
        return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=DATASETS)
    p.add_argument("out")
    p.add_argument("--no-lookup", action="store_true")
    p.add_argument("--refine", action="store_true")
    p.add_argument("--backend", default="anthropic", choices=["anthropic", "openai"])
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--api-key-env", help="environment variable holding the API key")
    p.add_argument("--json-mode", default="schema", help="openai backend: schema, object or prompt")
    p.add_argument("--metagenes", help="comma-separated subset of metagene indices")
    p.add_argument("--max-workers", type=int, default=8)
    a = p.parse_args()

    api_key = os.environ[a.api_key_env] if a.api_key_env else None
    if a.backend == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, base_url=a.base_url)
        tally = Tally(client.messages.create)
        client.messages = tally
        model = a.model or "claude-haiku-5-5"
        backend = AnthropicBackend(model, client=client)
    else:
        import openai

        client = openai.OpenAI(api_key=api_key, base_url=a.base_url)
        tally = Tally(client.chat.completions.create)
        client.chat.completions = tally
        model = a.model
        backend = OpenAIBackend(model, client=client, json_mode=a.json_mode)

    genes, H, W = load(a.dataset)
    metagenes = [int(i) for i in a.metagenes.split(",")] if a.metagenes else None
    t0 = time.time()
    results = describe_metagenes(
        genes, H, usage=W, context=DATASETS[a.dataset], metagenes=metagenes, backend=backend,
        lookup=not a.no_lookup, refine=a.refine, max_workers=a.max_workers,
    )
    df = to_dataframe(results)
    df.to_csv(a.out, sep="\t")
    cost = ""
    if model in PRICES:
        pi, po = PRICES[model]
        cost = f", ${(tally.input * pi + tally.output * po) * 1e-6:.4f}"
    print(
        f"{a.dataset} {model}: {tally.calls} calls, {tally.input} in / {tally.output} out tokens"
        f"{cost}, {time.time() - t0:.0f}s, {df.error.notna().sum()} errors",
        file=sys.stderr,
    )
    for i, err in df.error.dropna().items():
        print(f"  metagene {i}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
