from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from priver.io import ensure_dir, read_jsonl, read_yaml
from run_openclip_retrieval import (
    batched,
    cache_key,
    cache_path,
    crop_patch,
    load_cached_features,
    load_model,
    save_cached_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract features for multiple compatible encoders with shared image preprocessing."
    )
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument(
        "--devices",
        nargs="+",
        required=True,
        help="One PyTorch device per config, for example cuda:0 cuda:1.",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.configs) != len(args.devices):
        raise ValueError("--configs and --devices must contain the same number of values")

    configs = [read_yaml(path) for path in args.configs]
    output_dirs = {cfg["experiment"]["output_dir"] for cfg in configs}
    model_names = {cfg["retrieval"].get("model_name", "ViT-B-32") for cfg in configs}
    batch_sizes = {int(cfg["retrieval"].get("batch_size", 32)) for cfg in configs}
    if len(output_dirs) != 1:
        raise ValueError("All configs must use the same indexed dataset output directory")
    if len(model_names) != 1:
        raise ValueError("Shared preprocessing requires the same model_name")
    if len(batch_sizes) != 1:
        raise ValueError("All configs must use the same batch_size")

    output_dir = Path(next(iter(output_dirs)))
    batch_size = next(iter(batch_sizes))
    patch_rows = read_jsonl(output_dir / "patch_index.jsonl")
    query_rows = read_jsonl(output_dir / "query_index.jsonl")
    patch_ids = [row["patch_id"] for row in patch_rows]
    query_ids = [row["query_id"] for row in query_rows]

    models = []
    preprocesses = []
    tokenizers = []
    resolved_checkpoints = []
    for cfg, device in zip(configs, args.devices):
        model, preprocess, tokenizer, checkpoint = load_model(cfg["retrieval"], device)
        models.append(model)
        preprocesses.append(preprocess)
        tokenizers.append(tokenizer)
        resolved_checkpoints.append(checkpoint)
    preprocess_representations = {repr(preprocess) for preprocess in preprocesses}
    if len(preprocess_representations) != 1:
        raise ValueError("Encoder preprocessing pipelines differ and cannot be shared")

    patch_features: list[torch.Tensor | None] = []
    query_features: list[torch.Tensor | None] = []
    patch_paths = []
    query_paths = []
    for cfg in configs:
        retrieval_cfg = cfg["retrieval"]
        cache_dir = ensure_dir(
            output_dir / retrieval_cfg.get("cache_dir", "features")
        )
        patch_path = cache_path(
            cache_dir,
            retrieval_cfg.get(
                "patch_cache_prefix",
                retrieval_cfg.get("cache_prefix", "patch_openclip"),
            ),
            cache_key(
                patch_rows,
                ["patch_id", "image_path", "bbox", "patch_size"],
                retrieval_cfg,
                "patch",
            ),
        )
        query_path = cache_path(
            cache_dir,
            retrieval_cfg.get(
                "query_cache_prefix",
                retrieval_cfg.get("cache_prefix", "query_openclip"),
            ),
            cache_key(query_rows, ["query_id", "text"], retrieval_cfg, "query"),
        )
        patch_paths.append(patch_path)
        query_paths.append(query_path)
        patch_features.append(
            None
            if args.refresh_cache
            else load_cached_features(patch_path, patch_ids)
        )
        query_features.append(
            None
            if args.refresh_cache
            else load_cached_features(query_path, query_ids)
        )

    missing_patch_indices = [
        index for index, features in enumerate(patch_features) if features is None
    ]
    if missing_patch_indices:
        feature_batches: dict[int, list[torch.Tensor]] = {
            index: [] for index in missing_patch_indices
        }
        total = math.ceil(len(patch_rows) / batch_size)
        with torch.inference_mode():
            for batch in tqdm(
                batched(patch_rows, batch_size),
                total=total,
                desc="Encoding shared patch batches",
            ):
                cpu_images = torch.stack(
                    [preprocesses[0](crop_patch(row)) for row in batch]
                )
                for index in missing_patch_indices:
                    embeddings = models[index].encode_image(
                        cpu_images.to(args.devices[index])
                    )
                    embeddings = embeddings / embeddings.norm(
                        dim=-1, keepdim=True
                    )
                    feature_batches[index].append(embeddings.cpu())
        for index in missing_patch_indices:
            patch_features[index] = torch.cat(feature_batches[index], dim=0)

    missing_query_indices = [
        index for index, features in enumerate(query_features) if features is None
    ]
    if missing_query_indices:
        feature_batches = {index: [] for index in missing_query_indices}
        total = math.ceil(len(query_rows) / batch_size)
        with torch.inference_mode():
            for batch in tqdm(
                batched(query_rows, batch_size),
                total=total,
                desc="Encoding shared query batches",
            ):
                texts = [row["text"] for row in batch]
                for index in missing_query_indices:
                    tokens = tokenizers[index](texts).to(args.devices[index])
                    embeddings = models[index].encode_text(tokens)
                    embeddings = embeddings / embeddings.norm(
                        dim=-1, keepdim=True
                    )
                    feature_batches[index].append(embeddings.cpu())
        for index in missing_query_indices:
            query_features[index] = torch.cat(feature_batches[index], dim=0)

    summaries = []
    for index, cfg in enumerate(configs):
        retrieval_cfg = cfg["retrieval"]
        patch_tensor = patch_features[index]
        query_tensor = query_features[index]
        if patch_tensor is None or query_tensor is None:
            raise RuntimeError("Feature extraction did not produce all requested tensors")
        if index in missing_patch_indices:
            save_cached_features(
                patch_paths[index],
                patch_ids,
                patch_tensor,
                {
                    "type": "patch",
                    "retrieval": retrieval_cfg,
                    "num_items": len(patch_ids),
                },
            )
        if index in missing_query_indices:
            save_cached_features(
                query_paths[index],
                query_ids,
                query_tensor,
                {
                    "type": "query",
                    "retrieval": retrieval_cfg,
                    "num_items": len(query_ids),
                },
            )
        summaries.append(
            {
                "config": args.configs[index],
                "device": args.devices[index],
                "resolved_checkpoint": resolved_checkpoints[index],
                "num_patches": len(patch_ids),
                "num_queries": len(query_ids),
                "patch_feature_cache": str(patch_paths[index]),
                "query_feature_cache": str(query_paths[index]),
            }
        )
    for summary in summaries:
        print(summary)
    summary_path = output_dir / "multi_encoder_feature_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "num_patches": len(patch_ids),
                "num_queries": len(query_ids),
                "shared_model_name": next(iter(model_names)),
                "shared_batch_size": batch_size,
                "encoders": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print({"summary": str(summary_path)})


if __name__ == "__main__":
    main()
