# Release Checklist

## Required Before Public Release

- [ ] Confirm all coauthors approve public code release.
- [ ] Confirm Zhejiang University intellectual-property requirements.
- [x] Add the MIT open-source license.
- [x] Create the GitHub repository and add its URL to `README.md`.
- [x] Run the full test and secret audit documented below.
- [ ] Add article citation metadata after publication.

## Optional Archival Steps

- [ ] Tag an immutable release, for example `v0.1.0`.
- [ ] Archive the tagged release with Zenodo and add the DOI when available.

## Verification Commands

```bash
pytest -q
python examples/rerank_candidates.py
rg -n '/home/|100\.100\.|192\.168\.|API_KEY|TOKEN|PASSWORD' .
git status --short
```

Expected audit matches are limited to this checklist's search command
and explanatory placeholder text. Never commit `.env`, credentials,
raw datasets, model checkpoints, feature caches, or generated outputs.
