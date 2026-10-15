"""Evaluate image-search rankings against manually labeled query examples.

Manifest format:
[
  {"image": "queries/shoe-1.jpg", "item_id": "1771646202717"},
  {"image": "queries/bag-1.jpg", "item_id": "1771646202727"}
]

Run with: python benchmark_search.py path/to/manifest.json

To compare models, set AIFIND_MODEL_NAME and AIFIND_PRETRAINED, build that
model's separate index, and run the same manifest again. Set
AIFIND_ENABLE_OCR=1 before both index building and evaluation to include OCR.
"""

import argparse
import json
import time
from pathlib import Path


def calculate_metrics(ranks):
    if not ranks:
        raise ValueError("Benchmark manifest must contain at least one query.")

    count = len(ranks)
    return {
        "queries": count,
        "recall_at_1": sum(rank is not None and rank <= 1 for rank in ranks) / count,
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / count,
        "recall_at_10": sum(rank is not None and rank <= 10 for rank in ranks) / count,
        "mean_reciprocal_rank": sum(
            1 / rank if rank is not None else 0 for rank in ranks
        ) / count,
    }


def load_cases(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open(encoding="utf-8") as manifest_file:
        cases = json.load(manifest_file)

    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest must be a non-empty JSON list.")

    for case_number, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Query {case_number} must be a JSON object.")
        if not isinstance(case.get("image"), str) or not case["image"]:
            raise ValueError(f"Query {case_number} must include an image path.")
        if not isinstance(case.get("item_id"), str) or not case["item_id"]:
            raise ValueError(f"Query {case_number} must include an item_id.")

        image_path = Path(case["image"])
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Query image does not exist: {image_path} (query {case_number})"
            )
        case["image"] = str(image_path)

    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="JSON file with query image and expected item_id")
    args = parser.parse_args()

    cases = load_cases(args.manifest)

    from index_manager import get_manager
    from search_engine import search_multi

    manager = get_manager()
    index = manager.index
    if index is None or index.ntotal == 0:
        raise RuntimeError("Search index is empty; build the index before benchmarking.")

    ranks = []
    latencies = []
    for case in cases:
        started = time.perf_counter()
        response = search_multi([case["image"]], page_size=len(manager.item_rows))
        latencies.append((time.perf_counter() - started) * 1000)

        rank = next(
            (
                position
                for position, result in enumerate(response["results"], start=1)
                if result["item_id"] == case["item_id"]
            ),
            None,
        )
        ranks.append(rank)
        print(
            f'{case["image"]}: expected={case["item_id"]}, '
            f"rank={rank if rank is not None else 'not found'}"
        )

    report = calculate_metrics(ranks)
    report["average_latency_ms"] = sum(latencies) / len(latencies)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
