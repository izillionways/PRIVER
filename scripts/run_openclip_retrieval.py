from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import open_clip
from priver.io import ensure_dir, read_jsonl, read_yaml, write_jsonl
from priver.metrics import mean, precision_at_k, recall_at_k, topk_hit
from priver.retrieval import rank_queries_within_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenCLIP patch-query retrieval baseline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--max-patches", type=int, default=None, help="Optional patch limit for debugging.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional query limit for debugging.")
    parser.add_argument("--no-cache", action="store_true", help="Disable feature cache read/write.")
    parser.add_argument("--refresh-cache", action="store_true", help="Recompute and overwrite cached features.")
    return parser.parse_args()


def batched(items: list[dict], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


@lru_cache(maxsize=2)
def load_rgb_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def crop_patch(row: dict) -> Image.Image:
    image = load_rgb_image(row["image_path"])
    x1, y1, x2, y2 = row["bbox"]
    return image.crop((x1, y1, x2, y2))


def build_positive_patch_map(patch_rows: list[dict]) -> dict[tuple[str, str], set[str]]:
    positives: dict[tuple[str, str], set[str]] = defaultdict(set)
    for patch in patch_rows:
        for class_name, count in patch.get("class_counts", {}).items():
            if count > 0:
                positives[(patch["image_id"], class_name)].add(patch["patch_id"])
    return positives


def normalize_pretrained(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"", "none", "null"}:
        return None
    return value


def cache_key(rows: list[dict], fields: list[str], retrieval_cfg: dict, prefix: str) -> str:
    payload = {
        "prefix": prefix,
        "model_name": retrieval_cfg.get("model_name", "ViT-B-32"),
        "pretrained": normalize_pretrained(retrieval_cfg.get("pretrained", "openai")),
        "checkpoint_path": retrieval_cfg.get("checkpoint_path"),
        "hf_repo_id": retrieval_cfg.get("hf_repo_id"),
        "hf_filename": retrieval_cfg.get("hf_filename"),
        "cache_prefix": retrieval_cfg.get("cache_prefix"),
        "rows": [{field: row.get(field) for field in fields} for row in rows],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def cache_path(cache_dir: Path, prefix: str, key: str) -> Path:
    return cache_dir / f"{prefix}_{key}.npz"


def load_cached_features(path: Path, expected_ids: list[str]) -> torch.Tensor | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    ids = data["ids"].astype(str).tolist()
    if ids != expected_ids:
        return None
    return torch.from_numpy(data["features"].astype(np.float32))


def save_cached_features(path: Path, ids: list[str], features: torch.Tensor, metadata: dict) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        ids=np.asarray(ids),
        features=features.numpy().astype(np.float32),
        metadata=json.dumps(metadata, ensure_ascii=False),
    )


def resolve_checkpoint(retrieval_cfg: dict) -> Path | None:
    checkpoint_path = retrieval_cfg.get("checkpoint_path")
    if checkpoint_path:
        return Path(checkpoint_path)

    hf_repo_id = retrieval_cfg.get("hf_repo_id")
    hf_filename = retrieval_cfg.get("hf_filename")
    if hf_repo_id and hf_filename:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required for hf_repo_id/hf_filename checkpoints") from exc
        return Path(hf_hub_download(repo_id=hf_repo_id, filename=hf_filename))

    return None


def load_model(retrieval_cfg: dict, device: str):
    model_name = retrieval_cfg.get("model_name", "ViT-B-32")
    pretrained = normalize_pretrained(retrieval_cfg.get("pretrained", "openai"))
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    checkpoint_path = resolve_checkpoint(retrieval_cfg)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and not any(str(key).startswith("visual.") for key in state.keys()):
            state = state.get("state_dict") or state.get("model") or state
        incompatible = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                "Checkpoint loaded with "
                f"{len(incompatible.missing_keys)} missing keys and "
                f"{len(incompatible.unexpected_keys)} unexpected keys."
            )

    tokenizer = open_clip.get_tokenizer(model_name)
    return model.to(device).eval(), preprocess, tokenizer, str(checkpoint_path) if checkpoint_path else None


@torch.inference_mode()
def encode_patches(model, preprocess, patch_rows: list[dict], batch_size: int, device: str) -> torch.Tensor:
    feats: list[torch.Tensor] = []
    total = math.ceil(len(patch_rows) / batch_size)
    for batch in tqdm(batched(patch_rows, batch_size), total=total, desc="Encoding patches"):
        images = torch.stack([preprocess(crop_patch(row)) for row in batch]).to(device)
        emb = model.encode_image(images)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        feats.append(emb.cpu())
    return torch.cat(feats, dim=0)


@torch.inference_mode()
def encode_queries(model, tokenizer, query_rows: list[dict], batch_size: int, device: str) -> torch.Tensor:
    feats: list[torch.Tensor] = []
    total = math.ceil(len(query_rows) / batch_size)
    for batch in tqdm(batched(query_rows, batch_size), total=total, desc="Encoding queries"):
        text = tokenizer([row["text"] for row in batch]).to(device)
        emb = model.encode_text(text)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        feats.append(emb.cpu())
    return torch.cat(feats, dim=0)


def score_uncertainty(scores: np.ndarray) -> dict:
    if scores.size == 0:
        return {"score_entropy": 0.0, "top1_top2_margin": 0.0, "score_std": 0.0}
    shifted = scores - scores.max()
    probs = np.exp(shifted)
    probs = probs / max(probs.sum(), 1e-12)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / math.log(len(probs))) if len(probs) > 1 else 0.0
    margin = float(scores[0] - scores[1]) if len(scores) > 1 else float(scores[0])
    return {
        "score_entropy": entropy,
        "top1_top2_margin": margin,
        "score_std": float(scores.std()),
    }


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    out_dir = ensure_dir(cfg["experiment"]["output_dir"])
    retrieval_cfg = cfg["retrieval"]
    retrieval_name = retrieval_cfg.get("output_name", "retrieval_openclip")
    retrieval_dir = ensure_dir(out_dir / retrieval_name)

    patch_rows = read_jsonl(out_dir / "patch_index.jsonl")
    query_rows = read_jsonl(out_dir / "query_index.jsonl")
    if args.max_patches:
        patch_rows = patch_rows[: args.max_patches]
    if args.max_queries:
        query_rows = query_rows[: args.max_queries]

    device = retrieval_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    batch_size = int(retrieval_cfg.get("batch_size", 32))
    top_k = int(retrieval_cfg.get("top_k", 10))
    candidate_k = int(retrieval_cfg.get("candidate_k", top_k))
    cache_enabled = bool(retrieval_cfg.get("cache_features", True)) and not args.no_cache
    cache_dir = ensure_dir(out_dir / retrieval_cfg.get("cache_dir", "features"))

    model, preprocess, tokenizer, resolved_checkpoint = load_model(retrieval_cfg, device)

    patch_ids = [row["patch_id"] for row in patch_rows]
    query_ids = [row["query_id"] for row in query_rows]
    patch_cache = cache_path(
        cache_dir,
        retrieval_cfg.get("patch_cache_prefix", retrieval_cfg.get("cache_prefix", "patch_openclip")),
        cache_key(
            patch_rows,
            ["patch_id", "image_path", "bbox", "patch_size"],
            retrieval_cfg,
            "patch",
        ),
    )
    query_cache = cache_path(
        cache_dir,
        retrieval_cfg.get("query_cache_prefix", retrieval_cfg.get("cache_prefix", "query_openclip")),
        cache_key(query_rows, ["query_id", "text"], retrieval_cfg, "query"),
    )

    patch_features = None
    query_features = None
    if cache_enabled and not args.refresh_cache:
        patch_features = load_cached_features(patch_cache, patch_ids)
        query_features = load_cached_features(query_cache, query_ids)

    if patch_features is None:
        patch_features = encode_patches(model, preprocess, patch_rows, batch_size, device)
        if cache_enabled:
            save_cached_features(
                patch_cache,
                patch_ids,
                patch_features,
                {"type": "patch", "retrieval": retrieval_cfg, "num_items": len(patch_ids)},
            )
    else:
        print(f"Loaded cached patch features: {patch_cache}")

    if query_features is None:
        query_features = encode_queries(model, tokenizer, query_rows, batch_size, device)
        if cache_enabled:
            save_cached_features(
                query_cache,
                query_ids,
                query_features,
                {"type": "query", "retrieval": retrieval_cfg, "num_items": len(query_ids)},
            )
    else:
        print(f"Loaded cached query features: {query_cache}")

    positives = build_positive_patch_map(patch_rows)

    rankings = rank_queries_within_images(
        query_features.numpy(),
        patch_features.numpy(),
        query_rows,
        patch_rows,
        candidate_k,
    )
    results: list[dict] = []
    metric_rows: list[dict] = []

    for query_idx, query in enumerate(tqdm(query_rows, desc="Retrieving")):
        top_indices, top_scores = rankings[query_idx]
        retrieved = [
            {
                "patch_id": patch_ids[idx],
                "score": float(score),
                "bbox": patch_rows[idx]["bbox"],
                "patch_size": patch_rows[idx]["patch_size"],
                "class_counts": patch_rows[idx].get("class_counts", {}),
            }
            for idx, score in zip(top_indices, top_scores)
        ]
        retrieved_patch_ids = [row["patch_id"] for row in retrieved]
        positive_patch_ids = positives.get((query["image_id"], query["class_name"]), set())
        uncertainty = score_uncertainty(top_scores)
        metrics = {
            "query_id": query["query_id"],
            "image_id": query["image_id"],
            "class_name": query["class_name"],
            "num_positive_patches": len(positive_patch_ids),
            "hit_at_1": topk_hit(retrieved_patch_ids, positive_patch_ids, 1),
            "hit_at_5": topk_hit(retrieved_patch_ids, positive_patch_ids, min(5, top_k)),
            "hit_at_10": topk_hit(retrieved_patch_ids, positive_patch_ids, min(10, top_k)),
            "precision_at_5": precision_at_k(retrieved_patch_ids, positive_patch_ids, min(5, top_k)),
            "precision_at_10": precision_at_k(retrieved_patch_ids, positive_patch_ids, min(10, top_k)),
            "recall_at_5": recall_at_k(retrieved_patch_ids, positive_patch_ids, min(5, top_k)),
            "recall_at_10": recall_at_k(retrieved_patch_ids, positive_patch_ids, min(10, top_k)),
            **uncertainty,
        }
        metric_rows.append(metrics)
        results.append(
            {
                "query": query,
                "metrics": metrics,
                "retrieved": retrieved,
            }
        )

    summary = {
        "encoder": retrieval_cfg,
        "retrieval_name": retrieval_name,
        "resolved_checkpoint": resolved_checkpoint,
        "num_patches": len(patch_rows),
        "num_queries": len(query_rows),
        "candidate_k": candidate_k,
        "top_k": top_k,
        "hit_at_1": mean([row["hit_at_1"] for row in metric_rows]),
        "hit_at_5": mean([row["hit_at_5"] for row in metric_rows]),
        "hit_at_10": mean([row["hit_at_10"] for row in metric_rows]),
        "precision_at_5": mean([row["precision_at_5"] for row in metric_rows]),
        "precision_at_10": mean([row["precision_at_10"] for row in metric_rows]),
        "recall_at_5": mean([row["recall_at_5"] for row in metric_rows]),
        "recall_at_10": mean([row["recall_at_10"] for row in metric_rows]),
        "score_entropy": mean([row["score_entropy"] for row in metric_rows]),
        "top1_top2_margin": mean([row["top1_top2_margin"] for row in metric_rows]),
        "cache_enabled": cache_enabled,
        "patch_feature_cache": str(patch_cache) if cache_enabled else None,
        "query_feature_cache": str(query_cache) if cache_enabled else None,
    }

    write_jsonl(retrieval_dir / "retrieval_results.jsonl", results)
    write_jsonl(retrieval_dir / "retrieval_metrics.jsonl", metric_rows)
    (retrieval_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
