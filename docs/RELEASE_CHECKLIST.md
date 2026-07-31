# Release Checklist

## Required Before Public Release

- [ ] Confirm all coauthors approve public code release.
- [ ] Confirm Zhejiang University intellectual-property requirements.
- [ ] Select and add an open-source license, if release is approved.
- [x] Create the GitHub repository and add its URL to `README.md` and
  `CITATION.cff`.
- [ ] Run the full test and secret audit documented below.
- [ ] Tag an immutable release, for example `v0.1.0`.
- [ ] Archive the tagged release with Zenodo and add the DOI.
- [ ] Replace the manuscript's anonymized code statement after peer
  review permits repository disclosure.

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
