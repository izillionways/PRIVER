# Reproducibility Guide

## 1. Dataset Layout

Download DOTA-v1.5 and SODA-A from their official distribution channels.
The container expects the following read-only layout:

```text
/data/rsdata/
├── DOTA-v1.5/
│   └── DOTA_V1.5/
└── SODA-A/
```

The repository contains only source-image split manifests. Raw images
and annotations are not redistributed.

## 2. Environment

Copy `.env.example` to `.env`, set `RSDATA_ROOT`, and optionally set the
proxy variables. Build and enter the container:

```bash
docker compose --env-file .env -f docker/compose.yaml build
docker compose --env-file .env -f docker/compose.yaml run --rm priver bash
```

The reference image uses CUDA 12.1, PyTorch 2.3.1, OpenCLIP, and Python
3.10 on Ubuntu 22.04.

## 3. Indexing and Tiling

For each dataset configuration:

```bash
python scripts/build_dataset_index.py --config <dataset-config>
python scripts/tile_patches.py --config <dataset-config>
python scripts/validate_expanded_indexes.py
```

Tiling uses 512- and 1024-pixel windows with 25% overlap. A patch is
relevant when it covers at least 30% of an oriented annotation polygon.
The image-level manifests prevent development/test leakage.

## 4. Features and Semantic Retrieval

Run `scripts/run_openclip_retrieval.py` for OpenCLIP. The GeoRSCLIP and
RemoteCLIP configurations identify their public Hugging Face
checkpoints. Use `scripts/extract_multi_encoder_features.py` when
extracting multiple compatible encoders concurrently.

The final semantic candidate list is built with:

```bash
python scripts/build_prompt_stable_candidates.py \
  --dataset-config <dataset-config> \
  --patch-features <patch-cache.npz> \
  --query-features <query-cache.npz> \
  --priver-config configs/priver_frozen.json \
  --out-dir <prompt-output>
```

This command applies the fixed four-template mean-minus-0.5-standard-
deviation score. It performs no parameter search.

## 5. PRIVER and Evaluation

```bash
python scripts/apply_priver.py \
  --dataset-config <dataset-config> \
  --retrieval-dir <prompt-output> \
  --priver-config configs/priver_frozen.json \
  --out-dir <priver-output>
```

The JSON configuration is the authoritative record of candidate budget,
support kernels, reciprocal-neighbor count, degree normalization, and
support weights. Do not tune it on either test set.

Compact source tables used by the manuscript are under
`results/tables/`. Confidence intervals use source-image cluster
bootstrap resampling rather than treating four prompt variants as
independent samples.

## 6. Verification

Run:

```bash
pytest -q
python examples/rerank_candidates.py
```

Before comparing new results with the paper, record the Git commit,
dataset checksums or release versions, encoder checkpoint hashes, GPU,
and command lines.
