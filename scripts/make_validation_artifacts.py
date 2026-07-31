from __future__ import annotations

import argparse
import csv
from pathlib import Path


DATASETS = {
    "DOTA-v1.5": Path("outputs/dota_v15_val_458"),
    "SODA-A": Path("outputs/soda_a_val_holdout_526"),
}
PRIMARY_METRICS = (
    "precision_at_10",
    "ndcg_at_10",
    "object_recall_at_10",
)
METHODS = (
    "prompt",
    "same_scale_only",
    "inter_scale_only",
    "all_overlap_smoothing",
    "reciprocal_no_degree",
    "priver",
)
METHOD_LABELS = {
    "prompt": "Prompt only",
    "same_scale_only": "Same-scale only",
    "inter_scale_only": "Inter-scale only",
    "all_overlap_smoothing": "All-overlap smoothing",
    "reciprocal_no_degree": "Reciprocal, no degree norm.",
    "priver": "PRIVER",
}
METRIC_LABELS = {
    "precision_at_10": "P@10",
    "ndcg_at_10": "nDCG@10",
    "object_recall_at_10": "Object R@10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manuscript validation tables from verified CSVs."
    )
    parser.add_argument(
        "--latex-dir",
        required=True,
        help="Submission LaTeX directory receiving generated table files.",
    )
    parser.add_argument(
        "--paper-table-dir",
        default="paper_tables/priver",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
    return {
        (row["method"], row["metric"]): row
        for row in rows
    }


def matched_control_rows(root: Path) -> list[dict]:
    output: list[dict] = []
    for dataset, dataset_root in DATASETS.items():
        rows = read_csv(
            root
            / dataset_root
            / "priver_paper/validation/control_comparison/summary.csv"
        )
        lookup = metric_lookup(rows)
        for method in METHODS:
            output.append(
                {
                    "dataset": dataset,
                    "method": method,
                    **{
                        metric: float(lookup[(method, metric)]["mean"])
                        for metric in PRIMARY_METRICS
                    },
                }
            )
    return output


def threshold_rows(root: Path) -> list[dict]:
    output: list[dict] = []
    for dataset, dataset_root in DATASETS.items():
        rows = read_csv(
            root
            / dataset_root
            / "priver_paper/validation/threshold_sensitivity/"
            "paired_deltas.csv"
        )
        for row in rows:
            if row["metric"] not in PRIMARY_METRICS:
                continue
            output.append(
                {
                    "dataset": dataset,
                    "threshold": float(row["threshold"]),
                    "metric": row["metric"],
                    "delta": float(row["delta"]),
                    "ci95_low": float(row["delta_ci95_low"]),
                    "ci95_high": float(row["delta_ci95_high"]),
                }
            )
    return output


def scale_rows(root: Path) -> list[dict]:
    output: list[dict] = []
    for dataset, dataset_root in DATASETS.items():
        rows = read_csv(
            root
            / dataset_root
            / "priver_paper/validation/scale_comparison/summary.csv"
        )
        lookup = metric_lookup(rows)
        for scale, prompt_method, priver_method in (
            ("512 only", "prompt_512", "priver_512"),
            ("1024 only", "prompt_1024", "priver_1024"),
            ("512 + 1024", "prompt_multiscale", "priver_multiscale"),
        ):
            output.append(
                {
                    "dataset": dataset,
                    "scale_set": scale,
                    **{
                        metric: (
                            float(lookup[(priver_method, metric)]["mean"])
                            - float(lookup[(prompt_method, metric)]["mean"])
                        )
                        for metric in PRIMARY_METRICS
                    },
                }
            )
    return output


def format_score(value: float, best: bool) -> str:
    text = f"{value:.3f}"
    return rf"\textbf{{{text}}}" if best else text


def format_delta(value: float) -> str:
    return f"{value:+.3f}"


def matched_controls_tex(rows: list[dict]) -> str:
    by_key = {
        (row["dataset"], row["method"]): row
        for row in rows
    }
    best = {
        (dataset, metric): max(
            by_key[(dataset, method)][metric] for method in METHODS
        )
        for dataset in DATASETS
        for metric in PRIMARY_METRICS
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Matched controls under the final frozen protocol. All rows "
        r"use the same prompt-stable candidate pool, \(K_c=100\), and "
        r"\(K=10\); no control is retuned. Best values in each dataset column "
        r"are bold.}",
        r"\label{tab:matched-controls}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{DOTA-v1.5} "
        r"& \multicolumn{3}{c}{SODA-A} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Method & P@10 & nDCG@10 & Object R@10 "
        r"& P@10 & nDCG@10 & Object R@10 \\",
        r"\midrule",
    ]
    for method in METHODS:
        values = []
        for dataset in DATASETS:
            for metric in PRIMARY_METRICS:
                value = by_key[(dataset, method)][metric]
                values.append(
                    format_score(
                        value,
                        abs(value - best[(dataset, metric)]) < 1e-12,
                    )
                )
        lines.append(
            f"{METHOD_LABELS[method]} & " + " & ".join(values) + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def threshold_sensitivity_tex(rows: list[dict]) -> str:
    by_key = {
        (row["dataset"], row["threshold"], row["metric"]): row
        for row in rows
    }
    thresholds = sorted({row["threshold"] for row in rows})
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Relevance-threshold sensitivity. Entries are paired "
        r"PRIVER-minus-Prompt-only gains with 95\% source-image cluster "
        r"bootstrap intervals. Rankings and parameters remain fixed; only "
        r"the oriented-polygon coverage criterion is changed.}",
        r"\label{tab:threshold-sensitivity}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Coverage threshold & \(\Delta\)P@10 "
        r"& \(\Delta\)nDCG@10 & \(\Delta\)Object R@10 \\",
        r"\midrule",
    ]
    for dataset_index, dataset in enumerate(DATASETS):
        for threshold in thresholds:
            cells = []
            for metric in PRIMARY_METRICS:
                row = by_key[(dataset, threshold, metric)]
                cells.append(
                    f"{format_delta(row['delta'])} "
                    f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
                )
            lines.append(
                f"{dataset} & {threshold:.1f} & "
                + " & ".join(cells)
                + r" \\"
            )
        if dataset_index == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def scale_sensitivity_tex(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Scale-set sensitivity with frozen parameters. Entries are "
        r"PRIVER-minus-Prompt-only differences.}",
        r"\label{tab:scale-sensitivity}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Patch scales & \(\Delta\)P@10 & \(\Delta\)nDCG@10 "
        r"& \(\Delta\)Object R@10 \\",
        r"\midrule",
    ]
    for index, row in enumerate(rows):
        if index == 3:
            lines.append(r"\midrule")
        lines.append(
            f"{row['dataset']} & {row['scale_set']} & "
            + " & ".join(
                format_delta(row[metric]) for metric in PRIMARY_METRICS
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(__file__).parents[1]
    table_dir = root / args.paper_table_dir
    latex_table_dir = Path(args.latex_dir) / "tables"
    latex_table_dir.mkdir(parents=True, exist_ok=True)

    controls = matched_control_rows(root)
    thresholds = threshold_rows(root)
    scales = scale_rows(root)
    write_csv(table_dir / "matched_controls.csv", controls)
    write_csv(table_dir / "relevance_threshold_sensitivity.csv", thresholds)
    write_csv(table_dir / "scale_sensitivity.csv", scales)

    (latex_table_dir / "matched_controls.tex").write_text(
        matched_controls_tex(controls), encoding="utf-8"
    )
    (latex_table_dir / "threshold_sensitivity.tex").write_text(
        threshold_sensitivity_tex(thresholds), encoding="utf-8"
    )
    (latex_table_dir / "scale_sensitivity.tex").write_text(
        scale_sensitivity_tex(scales), encoding="utf-8"
    )
    print(
        {
            "controls": len(controls),
            "threshold_rows": len(thresholds),
            "scale_rows": len(scales),
            "latex_table_dir": str(latex_table_dir),
        }
    )


if __name__ == "__main__":
    main()
