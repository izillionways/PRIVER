# Verification Record

Verification was repeated on 2026-08-06 before public release.

## Automated Checks

- Unit tests: 32 passed.
- Static syntax check: 49 Python files parsed successfully.
- Docker Compose: configuration resolved successfully with a placeholder
  dataset root.
- Synthetic smoke example: completed successfully.

## Implementation Equivalence

The public `priver.reranking` implementation was compared with the
original final experiment implementation on 100 seeded, randomized
within-image candidate lists containing 1 to 150 patches.

For every candidate list, the following matched:

- complete patch ranking;
- normalized semantic score;
- same-scale support;
- reciprocal inter-scale support;
- final PRIVER score.

This check confirms that the release refactor changes package structure
and field names, not the frozen scoring behavior.

## Release Audit

The tracked source tree must contain no credentials, private proxy
addresses, machine-specific absolute paths, raw datasets, model
weights, or feature caches. Repeat the commands in
`RELEASE_CHECKLIST.md` before each tagged release.
