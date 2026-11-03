"""Label one example dataset's Leiden clusters, from metagene descriptions made by run.py, and write
the results as a TSV.

    uv run --group eval python eval/run_clusters.py DATASET METAGENES.tsv OUT.tsv
        [--basis expr|metagene] [--res 2.0] [--no-refine] [--no-broad]
        [--backend anthropic|openai] [--model M] [--base-url URL] [--api-key-env VAR]

Prints token usage and wall time (and cost, for known Anthropic models) to stderr.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import cluster_reference  # noqa: E402
from datasets import DATASETS, load  # noqa: E402
from run import PRICES, Tally  # noqa: E402

import pandas as pd  # noqa: E402

from metagene_describer import (  # noqa: E402
    AnthropicBackend,
    MetageneDescription,
    OpenAIBackend,
    describe_clusters,
    to_dataframe,
)


def read_descriptions(path: str) -> list[MetageneDescription]:
    df = pd.read_csv(path, sep="\t", keep_default_na=False)

    def split(s):
        return [x for x in s.split(", ") if x]

    return [
        MetageneDescription(
            int(r["index"]), r["label"], r["description"], r["category"], r["confidence"],
            cell_types=split(r["cell_types"]), key_genes=split(r["key_genes"]),
            top_genes=split(r["top_genes"]),
        )
        for _, r in df.iterrows()
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=DATASETS)
    p.add_argument("metagenes", help="describe_metagenes results for the dataset (TSV)")
    p.add_argument("out")
    p.add_argument("--basis", default="expr", choices=list(cluster_reference.CLUSTERS))
    p.add_argument("--res", default="2.0")
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--no-broad", action="store_true")
    p.add_argument("--backend", default="anthropic", choices=["anthropic", "openai"])
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--api-key-env", help="environment variable holding the API key")
    p.add_argument("--json-mode", default="schema", help="openai backend: schema, object or prompt")
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
    clusters = cluster_reference.clusters(a.dataset, a.basis, a.res)
    t0 = time.time()
    results = describe_clusters(
        clusters, read_descriptions(a.metagenes), genes, H, W, context=DATASETS[a.dataset],
        backend=backend, refine=not a.no_refine, broad=not a.no_broad,
        max_workers=a.max_workers,
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


if __name__ == "__main__":
    main()
