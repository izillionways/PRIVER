# Contributing

Keep changes reproducible and scoped. Use Python 3.10 or later, 4-space
indentation, `snake_case` names, and type hints for public functions.
Run `pytest -q` before opening a pull request.

Do not commit datasets, checkpoints, feature caches, generated outputs,
credentials, proxy addresses, or machine-specific absolute paths.
Place reusable code in `src/priver/`, command-line workflows in
`scripts/`, and dataset-independent tests in `tests/`.

Pull requests should explain the scientific or engineering reason for
the change, list the commands used for verification, and identify any
reported metric or manuscript claim affected. Changes to the frozen
configuration or split manifests require a separate rationale and must
not silently replace published results.
