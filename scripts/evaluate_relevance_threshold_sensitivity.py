from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from tqdm import tqdm

from compare_ranked_outputs import METRICS, evaluate_ranking
from priver.geometry import polygon_coverage
from priver.io import read_jsonl
from priver.ranking_evaluation import bootstrap_cluster_means


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild polygon-coverage links and evaluate fixed rankings "
            "under alternative relevance thresholds."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--methods",
        required=True,
        help="Comma-separated name=path entries relative to output-dir.",
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-batch-size", type=int, default=500)
    parser.add_argument(
        "--reference-summary",
        default=None,
        help="Optional long-form 0.3 summary.csv used for exact verification.",
    )
    parser.add_argument(
        "--reuse-coverage-cache",
        action="store_true",
        help="Reuse a previously verified cache in result-dir.",
    )
    return parser.parse_args()


def parse_methods(raw: str) -> dict[str, Path]:
    methods: dict[str, Path] = {}
    for item in raw.split(","):
        name, path = item.split("=", 1)
        methods[name.strip()] = Path(path.strip())
    if not methods:
        raise ValueError("At least one method is required")
    return methods


def iter_jsonl(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boxes_have_positive_area_overlap(
    first: list[float], second: list[float]
) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def build_coverage_cache(
    output_dir: Path,
    cache_path: Path,
    min_threshold: float,
) -> dict:
    object_path = output_dir / "object_index.jsonl"
    patch_path = output_dir / "patch_index.jsonl"
    original_path = output_dir / "patch_object_links.jsonl"
    tiling_summary = json.loads(
        (output_dir / "tiling_summary.json").read_text(encoding="utf-8")
    )
    verification_threshold = float(
        tiling_summary["min_object_coverage"]
    )
    if min_threshold > verification_threshold:
        raise ValueError(
            "Minimum sensitivity threshold must not exceed the source threshold"
        )

    objects_by_image: dict[str, list[dict]] = defaultdict(list)
    for obj in read_jsonl(object_path):
        objects_by_image[obj["image_id"]].append(obj)

    original_iterator = iter_jsonl(original_path)
    original_current = next(original_iterator, None)
    verified_links = 0
    cached_links = 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
        for patch in tqdm(
            iter_jsonl(patch_path),
            total=int(tiling_summary["num_patches"]),
            desc="Rebuilding polygon coverage links",
        ):
            for obj in objects_by_image[patch["image_id"]]:
                if not boxes_have_positive_area_overlap(
                    patch["bbox"], obj["bbox"]
                ):
                    continue
                coverage = polygon_coverage(obj["polygon"], patch["bbox"])
                link = {
                    "patch_id": patch["patch_id"],
                    "image_id": patch["image_id"],
                    "object_id": obj["object_id"],
                    "class_name": obj["class_name"],
                    "coverage": coverage,
                    "coverage_geometry": "polygon",
                }
                if coverage >= verification_threshold:
                    if original_current is None:
                        raise ValueError(
                            "Rebuilt source-threshold links exceed the original"
                        )
                    keys = (
                        "patch_id",
                        "image_id",
                        "object_id",
                        "class_name",
                        "coverage_geometry",
                    )
                    if any(
                        link[key] != original_current[key] for key in keys
                    ) or not np.isclose(
                        coverage,
                        float(original_current["coverage"]),
                        rtol=0.0,
                        atol=1e-12,
                    ):
                        raise ValueError(
                            "Coverage-link reconstruction differs at "
                            f"{patch['patch_id']} / {obj['object_id']}"
                        )
                    verified_links += 1
                    original_current = next(original_iterator, None)
                if coverage >= min_threshold:
                    handle.write(
                        json.dumps(link, ensure_ascii=False) + "\n"
                    )
                    cached_links += 1

    if original_current is not None:
        raise ValueError(
            "Rebuilt source-threshold links did not exhaust the original"
        )
    metadata = {
        "status": "PASS",
        "output_dir": str(output_dir),
        "cache": str(cache_path),
        "minimum_coverage": min_threshold,
        "source_verification_threshold": verification_threshold,
        "cached_links": cached_links,
        "verified_source_links": verified_links,
        "expected_source_links": int(
            tiling_summary["num_patch_object_links"]
        ),
        "input_sha256": {
            "object_index": sha256(object_path),
            "patch_index": sha256(patch_path),
            "patch_object_links": sha256(original_path),
        },
    }
    if verified_links != metadata["expected_source_links"]:
        raise ValueError(
            "Verified source-link count does not match tiling_summary.json"
        )
    return metadata


def accumulate_relevance_maps(
    links: Iterable[dict],
    threshold: float,
) -> tuple[
    dict[tuple[str, str], set[str]],
    dict[str, set[str]],
    int,
]:
    positive_patch_map: dict[tuple[str, str], set[str]] = defaultdict(set)
    patch_objects: dict[str, set[str]] = defaultdict(set)
    count = 0
    for link in links:
        if float(link["coverage"]) < threshold:
            continue
        positive_patch_map[
            (link["image_id"], link["class_name"])
        ].add(link["patch_id"])
        patch_objects[link["patch_id"]].add(link["object_id"])
        count += 1
    return dict(positive_patch_map), dict(patch_objects), count


def load_rankings(
    output_dir: Path,
    method_paths: dict[str, Path],
) -> dict[str, dict[str, dict]]:
    rankings: dict[str, dict[str, dict]] = {}
    for method, relative_path in method_paths.items():
        rows = read_jsonl(output_dir / relative_path)
        rankings[method] = {
            row["query"]["query_id"]: row for row in rows
        }
    query_sets = [set(rows) for rows in rankings.values()]
    if any(query_set != query_sets[0] for query_set in query_sets[1:]):
        raise ValueError("Methods do not contain identical query IDs")
    return rankings


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_threshold(
    threshold: float,
    rankings: dict[str, dict[str, dict]],
    positive_patch_map: dict[tuple[str, str], set[str]],
    patch_objects: dict[str, set[str]],
    baseline: str,
    top_k: int,
    n_bootstrap: int,
    seed: int,
    bootstrap_batch_size: int,
) -> tuple[list[dict], list[dict], dict[str, dict[str, dict]]]:
    query_ids = sorted(next(iter(rankings.values())))
    metrics_by_method: dict[str, dict[str, dict]] = {}
    for method, rows in rankings.items():
        method_metrics: dict[str, dict] = {}
        for query_id in query_ids:
            row = rows[query_id]
            query = row["query"]
            positives = positive_patch_map.get(
                (query["image_id"], query["class_name"]), set()
            )
            method_metrics[query_id] = evaluate_ranking(
                query,
                row["retrieved"],
                positives,
                patch_objects,
                top_k,
            )
        metrics_by_method[method] = method_metrics

    image_ids = [
        metrics_by_method[baseline][query_id]["image_id"]
        for query_id in query_ids
    ]
    unique_images = sorted(set(image_ids))
    image_index = {
        image_id: index for index, image_id in enumerate(unique_images)
    }
    counts = np.zeros(len(unique_images), dtype=float)
    sums = {
        method: np.zeros((len(unique_images), len(METRICS)), dtype=float)
        for method in metrics_by_method
    }
    for query_id, image_id in zip(query_ids, image_ids, strict=True):
        index = image_index[image_id]
        counts[index] += 1.0
        for method, rows in metrics_by_method.items():
            sums[method][index] += np.asarray(
                [float(rows[query_id][metric]) for metric in METRICS],
                dtype=float,
            )
    boot = bootstrap_cluster_means(
        counts,
        sums,
        n_bootstrap=n_bootstrap,
        seed=seed,
        batch_size=bootstrap_batch_size,
    )

    summary_rows: list[dict] = []
    total_queries = len(query_ids)
    for method, method_sums in sums.items():
        means = method_sums.sum(axis=0) / total_queries
        lows, highs = np.percentile(boot[method], [2.5, 97.5], axis=0)
        for index, metric in enumerate(METRICS):
            summary_rows.append(
                {
                    "threshold": threshold,
                    "method": method,
                    "metric": metric,
                    "num_queries": total_queries,
                    "num_images": len(unique_images),
                    "mean": float(means[index]),
                    "ci95_low": float(lows[index]),
                    "ci95_high": float(highs[index]),
                }
            )

    delta_rows: list[dict] = []
    baseline_boot = boot[baseline]
    baseline_rows = metrics_by_method[baseline]
    for method in metrics_by_method:
        if method == baseline:
            continue
        for index, metric in enumerate(METRICS):
            differences = np.asarray(
                [
                    float(metrics_by_method[method][query_id][metric])
                    - float(baseline_rows[query_id][metric])
                    for query_id in query_ids
                ],
                dtype=float,
            )
            delta_samples = boot[method][:, index] - baseline_boot[:, index]
            low, high = np.percentile(delta_samples, [2.5, 97.5])
            delta_rows.append(
                {
                    "threshold": threshold,
                    "method": method,
                    "baseline": baseline,
                    "metric": metric,
                    "delta": float(differences.mean()),
                    "delta_ci95_low": float(low),
                    "delta_ci95_high": float(high),
                    "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
                    "query_wins": int((differences > 0.0).sum()),
                    "query_ties": int((differences == 0.0).sum()),
                    "query_losses": int((differences < 0.0).sum()),
                }
            )
    return summary_rows, delta_rows, metrics_by_method


def verify_reference_summary(
    reference_path: Path,
    rows: list[dict],
    threshold: float,
) -> dict:
    with reference_path.open("r", encoding="utf-8", newline="") as handle:
        reference_rows = list(csv.DictReader(handle))
    expected = {
        (row["method"], row["metric"]): float(row["mean"])
        for row in reference_rows
    }
    checked = 0
    mismatches: list[dict] = []
    for row in rows:
        if not np.isclose(float(row["threshold"]), threshold):
            continue
        key = (row["method"], row["metric"])
        if key not in expected:
            continue
        checked += 1
        if not np.isclose(
            float(row["mean"]), expected[key], rtol=0.0, atol=1e-12
        ):
            mismatches.append(
                {
                    "method": key[0],
                    "metric": key[1],
                    "rebuilt": float(row["mean"]),
                    "reference": expected[key],
                }
            )
    if mismatches:
        raise ValueError(
            f"Threshold-{threshold:g} metrics differ from the reference"
        )
    return {
        "status": "PASS",
        "threshold": threshold,
        "checked_method_metric_pairs": checked,
        "mismatches": mismatches,
        "reference_summary": str(reference_path),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    thresholds = sorted(
        {float(value) for value in args.thresholds.split(",")}
    )
    method_paths = parse_methods(args.methods)
    if args.baseline not in method_paths:
        raise ValueError(f"Unknown baseline: {args.baseline}")

    cache_path = result_dir / (
        f"coverage_links_min_{min(thresholds):g}".replace(".", "p")
        + ".jsonl.gz"
    )
    metadata_path = result_dir / "coverage_cache_metadata.json"
    if args.reuse_coverage_cache:
        if not cache_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Verified coverage cache is unavailable")
        cache_metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        if cache_metadata.get("status") != "PASS":
            raise ValueError("Coverage cache metadata is not verified")
    else:
        cache_metadata = build_coverage_cache(
            output_dir, cache_path, min(thresholds)
        )
        metadata_path.write_text(
            json.dumps(cache_metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    rankings = load_rankings(output_dir, method_paths)
    all_summary_rows: list[dict] = []
    all_delta_rows: list[dict] = []
    link_counts: list[dict] = []
    for threshold in thresholds:
        positive_map, patch_objects, link_count = accumulate_relevance_maps(
            iter_jsonl(cache_path), threshold
        )
        link_counts.append(
            {
                "threshold": threshold,
                "num_links": link_count,
                "num_image_class_groups": len(positive_map),
                "num_positive_patches": len(patch_objects),
            }
        )
        summary_rows, delta_rows, metrics_by_method = evaluate_threshold(
            threshold,
            rankings,
            positive_map,
            patch_objects,
            args.baseline,
            args.top_k,
            args.n_bootstrap,
            args.seed,
            args.bootstrap_batch_size,
        )
        all_summary_rows.extend(summary_rows)
        all_delta_rows.extend(delta_rows)
        per_query_path = result_dir / (
            f"per_query_metrics_threshold_{threshold:g}".replace(".", "p")
            + ".csv.gz"
        )
        with gzip.open(
            per_query_path, "wt", encoding="utf-8", newline=""
        ) as handle:
            first_row = next(
                iter(next(iter(metrics_by_method.values())).values())
            )
            writer = csv.DictWriter(
                handle,
                fieldnames=("threshold", "method") + tuple(first_row),
            )
            writer.writeheader()
            for method, rows in metrics_by_method.items():
                for query_id in sorted(rows):
                    writer.writerow(
                        {
                            "threshold": threshold,
                            "method": method,
                            **rows[query_id],
                        }
                    )

    write_csv(result_dir / "summary.csv", all_summary_rows)
    write_csv(result_dir / "paired_deltas.csv", all_delta_rows)
    write_csv(result_dir / "coverage_counts.csv", link_counts)

    reference_verification = None
    if args.reference_summary:
        source_threshold = float(
            cache_metadata["source_verification_threshold"]
        )
        reference_verification = verify_reference_summary(
            Path(args.reference_summary),
            all_summary_rows,
            source_threshold,
        )
        (result_dir / "reference_metric_verification.json").write_text(
            json.dumps(
                reference_verification, indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )

    run_config = {
        "output_dir": str(output_dir),
        "method_paths": {
            method: str(path) for method, path in method_paths.items()
        },
        "baseline": args.baseline,
        "thresholds": thresholds,
        "top_k": args.top_k,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "resample_unit": "source_image",
        "retuning": False,
        "coverage_cache": cache_metadata,
        "reference_verification": reference_verification,
    }
    (result_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(run_config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
