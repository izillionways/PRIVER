from __future__ import annotations

import argparse
import json
from pathlib import Path

from priver.dota import build_class_queries, collect_dota_records
from priver.io import ensure_dir, read_yaml, write_jsonl
from priver.soda import build_class_queries as build_soda_class_queries
from priver.soda import collect_soda_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DOTA dataset/query indexes.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    out_dir = ensure_dir(cfg["experiment"]["output_dir"])

    dataset_name = cfg["dataset"]["name"]
    if dataset_name == "dota_v1_5":
        image_rows, object_rows = collect_dota_records(cfg)
        query_builder = build_class_queries
    elif dataset_name == "soda_a":
        image_rows, object_rows = collect_soda_records(cfg)
        query_builder = build_soda_class_queries
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    query_cfg = cfg.get("queries", {})
    query_rows = query_builder(
        image_rows=image_rows,
        object_rows=object_rows,
        templates=query_cfg.get(
            "templates",
            [
                "Locate image patches containing {class_phrase}.",
                "Find regions that show {class_phrase}.",
                "Retrieve local areas containing {class_phrase}.",
                "Which image regions contain {class_phrase}?",
            ],
        ),
        min_objects_per_query=int(query_cfg.get("min_objects_per_query", 1)),
    )

    write_jsonl(out_dir / "image_index.jsonl", image_rows)
    write_jsonl(out_dir / "object_index.jsonl", object_rows)
    write_jsonl(out_dir / "query_index.jsonl", query_rows)

    summary = {
        "dataset": dataset_name,
        "split": cfg["dataset"]["split"],
        "num_images": len(image_rows),
        "num_objects": len(object_rows),
        "num_queries": len(query_rows),
        "output_dir": str(out_dir),
    }
    (out_dir / "index_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
