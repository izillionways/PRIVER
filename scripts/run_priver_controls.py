from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from priver.io import ensure_dir, read_jsonl, write_jsonl
from priver.reranking import (
    InterScaleSupportSpec,
    SameScaleSupportSpec,
    build_candidate_context,
    compute_inter_scale_support,
    compute_same_scale_support,
    stable_rank,
)


@dataclass(frozen=True)
class ControlCondition:
    name: str
    alpha: float
    beta: float
    graph: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate frozen PRIVER mechanism and scale-set controls."
    )
    parser.add_argument("--retrieval-dir", required=True)
    parser.add_argument("--priver-config", default="configs/priver_frozen.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--reference-priver-dir",
        default=None,
        help="Optional frozen PRIVER output used to verify the regenerated ranking.",
    )
    return parser.parse_args()


def load_priver_config(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("method") != "PRIVER":
        raise ValueError(f"{path} is not a PRIVER configuration")
    return document


def build_specs(
    config: dict[str, Any],
) -> tuple[SameScaleSupportSpec, dict[str, InterScaleSupportSpec]]:
    same = config["same_scale_support"]
    inter = config["inter_scale_support"]
    same_spec = SameScaleSupportSpec(
        kernel=str(same["kernel"]),
        aggregation=str(same["aggregation"]),
        top_m=int(same["top_m"]),
        supporter_floor=float(same["supporter_floor"]),
        score_weight=str(same["score_weight"]),
    )
    full = InterScaleSupportSpec(
        kernel=str(inter["kernel"]),
        aggregation=str(inter["aggregation"]),
        top_m=int(inter["top_m"]),
        supporter_floor=float(inter["supporter_floor"]),
        score_weight=str(inter["score_weight"]),
        reciprocal_k=int(inter["reciprocal_k"]),
        row_degree_power=float(inter["row_degree_power"]),
        column_degree_power=float(inter["column_degree_power"]),
        consensus=str(inter["consensus"]),
    )
    graphs = {
        "full": full,
        "all_overlap": replace(
            full,
            reciprocal_k=0,
            row_degree_power=0.0,
            column_degree_power=0.0,
        ),
        "reciprocal_no_degree": replace(
            full,
            row_degree_power=0.0,
            column_degree_power=0.0,
        ),
    }
    return same_spec, graphs


def build_control_conditions(
    config: dict[str, Any],
) -> tuple[ControlCondition, ...]:
    alpha = float(config["same_scale_weight"])
    beta = float(config["inter_scale_weight"])
    return (
        ControlCondition("same_scale_only", alpha, 0.0, "full"),
        ControlCondition("inter_scale_only", 0.0, beta, "full"),
        ControlCondition(
            "all_overlap_smoothing", alpha, beta, "all_overlap"
        ),
        ControlCondition(
            "reciprocal_no_degree",
            alpha,
            beta,
            "reciprocal_no_degree",
        ),
        ControlCondition("priver", alpha, beta, "full"),
    )


def filter_candidates(
    candidates: list[dict],
    candidate_k: int,
    patch_size: int | None,
) -> list[dict]:
    if patch_size is None:
        return candidates[:candidate_k]
    return [
        candidate
        for candidate in candidates
        if int(candidate["patch_size"]) == patch_size
    ][:candidate_k]


def rank_with_support(
    candidates: list[dict],
    same_support: np.ndarray,
    cross_support: np.ndarray,
    alpha: float,
    beta: float,
    top_k: int,
) -> list[dict]:
    context = build_candidate_context(candidates)
    scores = (
        context.base
        + float(alpha) * same_support
        + float(beta) * cross_support
    )
    ranked: list[dict] = []
    for index in stable_rank(scores)[:top_k]:
        item = dict(candidates[int(index)])
        item["base_score_norm"] = float(context.base[index])
        item["same_scale_support"] = float(same_support[index])
        item["inter_scale_support"] = float(cross_support[index])
        item["control_score"] = float(scores[index])
        ranked.append(item)
    return ranked


def score_control_conditions(
    candidates: list[dict],
    same_spec: SameScaleSupportSpec,
    graphs: dict[str, InterScaleSupportSpec],
    conditions: tuple[ControlCondition, ...],
    top_k: int,
) -> dict[str, list[dict]]:
    context = build_candidate_context(candidates)
    same_support = compute_same_scale_support(context, same_spec)
    cross_supports = {
        name: compute_inter_scale_support(context, spec)
        for name, spec in graphs.items()
    }
    outputs = {}
    for condition in conditions:
        outputs[condition.name] = rank_with_support(
            candidates,
            same_support,
            cross_supports[condition.graph],
            condition.alpha,
            condition.beta,
            top_k,
        )
    return outputs


def verify_priver_ranking(
    generated_rows: list[dict],
    reference_dir: Path,
) -> dict[str, Any]:
    reference_rows = read_jsonl(reference_dir / "retrieval_results.jsonl")
    if len(generated_rows) != len(reference_rows):
        raise ValueError("Generated and reference PRIVER query counts differ")
    mismatches: list[str] = []
    for generated, reference in zip(
        generated_rows, reference_rows, strict=True
    ):
        generated_id = generated["query"]["query_id"]
        reference_id = reference["query"]["query_id"]
        if generated_id != reference_id:
            raise ValueError(
                f"PRIVER query order differs: {generated_id} != {reference_id}"
            )
        generated_patches = [
            item["patch_id"] for item in generated["retrieved"]
        ]
        reference_patches = [
            item["patch_id"] for item in reference["retrieved"]
        ]
        if generated_patches != reference_patches:
            mismatches.append(generated_id)
    if mismatches:
        raise ValueError(
            "Regenerated PRIVER ranking differs for "
            f"{len(mismatches)} queries; first={mismatches[0]}"
        )
    return {
        "status": "PASS",
        "num_queries": len(generated_rows),
        "top_k_patch_id_mismatches": 0,
        "reference_dir": str(reference_dir),
    }


def main() -> None:
    args = parse_args()
    retrieval_dir = Path(args.retrieval_dir)
    config_path = Path(args.priver_config)
    output_dir = ensure_dir(args.out_dir)
    config = load_priver_config(config_path)
    same_spec, graphs = build_specs(config)
    conditions = build_control_conditions(config)
    candidate_k = int(config["candidate_k"])
    top_k = int(config["top_k"])

    condition_rows: dict[str, list[dict]] = {
        "prompt": [],
        **{condition.name: [] for condition in conditions},
    }
    scale_rows: dict[str, list[dict]] = {
        "prompt_512": [],
        "priver_512": [],
        "prompt_1024": [],
        "priver_1024": [],
    }

    retrieval_rows = read_jsonl(retrieval_dir / "retrieval_results.jsonl")
    full_condition = next(
        condition for condition in conditions if condition.name == "priver"
    )
    for row in tqdm(retrieval_rows, desc="PRIVER controls"):
        query = row["query"]
        candidates = filter_candidates(
            row["retrieved"], candidate_k, patch_size=None
        )
        condition_rows["prompt"].append(
            {
                "query": query,
                "retrieved": candidates[:top_k],
                "method": "prompt",
            }
        )
        ranked_conditions = score_control_conditions(
            candidates,
            same_spec,
            graphs,
            conditions,
            top_k,
        )
        for condition in conditions:
            condition_rows[condition.name].append(
                {
                    "query": query,
                    "retrieved": ranked_conditions[condition.name],
                    "method": condition.name,
                }
            )

        for patch_size in (512, 1024):
            scale_candidates = filter_candidates(
                row["retrieved"], candidate_k, patch_size
            )
            scale_rows[f"prompt_{patch_size}"].append(
                {
                    "query": query,
                    "retrieved": scale_candidates[:top_k],
                    "method": f"prompt_{patch_size}",
                }
            )
            scale_ranked = score_control_conditions(
                scale_candidates,
                same_spec,
                graphs,
                (full_condition,),
                top_k,
            )["priver"]
            scale_rows[f"priver_{patch_size}"].append(
                {
                    "query": query,
                    "retrieved": scale_ranked,
                    "method": f"priver_{patch_size}",
                }
            )

    controls_dir = ensure_dir(output_dir / "controls")
    for method, rows in condition_rows.items():
        method_dir = ensure_dir(controls_dir / method)
        write_jsonl(method_dir / "retrieval_results.jsonl", rows)

    scales_dir = ensure_dir(output_dir / "scale_sets")
    for method, rows in scale_rows.items():
        method_dir = ensure_dir(scales_dir / method)
        write_jsonl(method_dir / "retrieval_results.jsonl", rows)

    verification = None
    if args.reference_priver_dir:
        verification = verify_priver_ranking(
            condition_rows["priver"], Path(args.reference_priver_dir)
        )
        (output_dir / "reference_verification.json").write_text(
            json.dumps(verification, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    run_config = {
        "source_retrieval_dir": str(retrieval_dir),
        "priver_config": str(config_path),
        "candidate_k": candidate_k,
        "top_k": top_k,
        "num_queries": len(retrieval_rows),
        "retuning": False,
        "conditions": [
            {
                "name": condition.name,
                "alpha": condition.alpha,
                "beta": condition.beta,
                "graph": graphs[condition.graph].to_dict(),
            }
            for condition in conditions
        ],
        "scale_sets": [512, 1024, "multi-scale"],
        "reference_verification": verification,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(run_config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
