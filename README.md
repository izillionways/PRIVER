# PRIVER

PRIVER (Prompt-Stable Reciprocal Inter-scale Visual Evidence Reranking) is
a training-free method for reranking within-image patch candidates in
high-resolution remote-sensing imagery. It combines a prompt-stable
semantic score with same-scale spatial support and reciprocal,
degree-normalized inter-scale support.

This repository contains the implementation and compact result tables
for the accompanying manuscript. It does not distribute DOTA-v1.5,
SODA-A, model checkpoints, or cached image features.

## Revised Manuscript Release

Use [v1.1.2](https://github.com/izillionways/PRIVER/releases/tag/v1.1.2)
for the revised manuscript. Pass `configs/priver_revision_20260916.json`
explicitly: same-scale weight 0.10, inter-scale weight 1.25, and candidate
budget 100. The core algorithm is unchanged. `priver_frozen.json`, CLI
defaults, and `results/tables/` retain the original submission settings
and results; they must not be confused with the revision.

Revision table sources are in `results/revision_20260923/`. The release
asset includes analysis scripts, development records, numeric results,
and saved detector predictions for result inspection and replay.
See [revision reproduction notes](docs/REVISION_20260923.md) for the
available reproduction levels and required external inputs.

## Repository Layout

- `src/priver/`: reusable indexing, geometry, retrieval, evaluation, and
  PRIVER modules.
- `scripts/`: dataset preparation, feature extraction, retrieval,
  reranking, and analysis entry points.
- `configs/`: frozen dataset and PRIVER configurations.
- `splits/`: source-image manifests used in the manuscript.
- `results/tables/`: source CSV files for reported tables and figures.
- `tests/`: unit tests that do not require the research datasets.
- `docker/`: CUDA 12.1 container definition and Compose configuration.

## Quick Start

Python 3.10 or later is required. For the lightweight API and tests:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
pytest -q
python examples/rerank_candidates.py
```

For GPU feature extraction, Docker is recommended:

```bash
cp .env.example .env
# Set RSDATA_ROOT in .env to the parent directory of DOTA-v1.5 and SODA-A.
docker compose --env-file .env -f docker/compose.yaml build
docker compose --env-file .env -f docker/compose.yaml run --rm priver bash
```

Inside the container, build a dataset index and multi-scale patch index:

```bash
python scripts/build_dataset_index.py \
  --config configs/dota_v15_val_458_ck200.yaml
python scripts/tile_patches.py \
  --config configs/dota_v15_val_458_ck200.yaml
python scripts/run_openclip_retrieval.py \
  --config configs/dota_v15_val_458_ck200.yaml
```

Use `build_prompt_stable_candidates.py` with the generated patch and
query feature caches, then run the frozen reranker:

```bash
python scripts/apply_priver.py \
  --dataset-config configs/dota_v15_val_458_ck200.yaml \
  --retrieval-dir outputs/dota_v15_val_458/retrieval_prompt_stable \
  --priver-config configs/priver_revision_20260916.json \
  --out-dir outputs/dota_v15_val_458/priver
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the full
pipeline and dataset layout.

## Manuscript Protocol

The manuscript configuration uses 512- and 1024-pixel patches, polygon
coverage for annotation-defined relevance, four semantically equivalent
query templates, a candidate pool of 100, and a returned list of 10.
Prompt-stable similarities are clipped at zero before candidate-pool
min--max normalization. All method choices were selected on the
DOTA-v1.5 development split and applied unchanged to the DOTA-v1.5
internal test and SODA-A holdout. The revision re-evaluates these previously
used sets and adds a fixed confirmation on 1,067 previously unused SODA-A
images. Inter-scale support supplies the main gain; independent effects
of same-scale support and reciprocity are limited, and large-window
preference remains. The results do not establish free-form query
understanding or missing-class rejection.

## License

The PRIVER source code, configuration files, documentation, and aggregate
result tables in this repository are released under the [MIT License](LICENSE).
DOTA-v1.5, SODA-A, model checkpoints, and other third-party resources remain
subject to their providers' terms and are not redistributed here.
