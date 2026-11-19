# Installation and supported commands

These instructions are for the author or separately authorized reviewers. The All Rights Reserved notice does not automatically authorize local execution, copying or installation beyond applicable law and platform terms. Repository access and these instructions do not authorize public release or expand the rights granted by the notice. A public-visibility transition requires a separate, explicit author decision.

## Recorded supported environment

The installed Core demonstration was validated on macOS arm64, CPython 3.11.5. P4-D2B used a fresh isolated environment with 16 reviewed external Core wheels plus the original APF wheel, normal dependency checks, offline hash enforcement and successful pip check. It did not install tracking, pytest, LightGBM or PostgreSQL support. This is a specific observed installation, not general release-installation certification.

No new v5 wheel has been built. Documentation changes do not change the 50 scientific package/example/resource files, but the changed packaged README would change a rebuilt wheel's metadata and hash. Never reuse the original wheel hash for a changed build.

## Installed Core quick start

With the reviewed environment already installed and active, change to a working directory outside the checkout. Use a new output directory:

```bash
python -I -B -m examples.synthetic_research --output /tmp/apf-demo-new
```

The package also declares `apf-synthetic = examples.synthetic_research:main`. Its arguments are required `--output` and optional `--track-local`; no research-only, provider or remote-tracking flag exists. The unchanged demo includes synthetic execution comparisons. Its outputs include summary.json, source inputs, an immutable research archive and separate policy ledger/fill CSVs. Existing output directories are rejected.

The CLI blocks networking during the demonstration and permits only its newly owned SQLite output when tracking is requested. Those API guards are not a sandbox for hostile code. No historical database or market data is needed.

## Optional tracking and full tests

The `tracking` extra provides optional MLflow-related packages. Only after that separate reviewed installation, use another new root:

```bash
python -I -B -m examples.synthetic_research --output /tmp/apf-demo-tracked-new --track-local
```

This creates a fresh local SQLite store; it does not start a service or use a production destination. Core plus the `test` extra alone is insufficient for every checkout test: the full research tests also use tracking and its reviewed transitives. From the project root, with that already prepared development environment:

```bash
python -m pytest tests -k 'not native_linux'
```

Run it as one pytest session. On macOS it deselects eight native Linux cases. Neither the command nor this documentation claims a new full v5 run. Existing workflows and CI drivers reject incomplete locks before dependency installation; do not bypass their gates.

## Preparing a new installation

No universal one-line install is offered while dependency evidence remains incomplete. `pyproject.toml` specifies Core pins and separate `tracking` / `test` extras; requirements/portfolio.txt is a direct constraint list, not a complete hashed lock. See [dependency locking](dependency_locking.md). macOS profiles remain LOCK_PARTIAL and Linux profiles LOCK_UNAVAILABLE. A new environment, artifact retrieval, install or build requires a separately reviewed plan and authorization; the documentation is not such authorization.

Source builds use declared setuptools 68.0.0 and wheel 0.38.4. Do not change versions, use --no-deps, conceal source imports or silently access indexes. Verify installed module origins outside the checkout. A new build must receive its own hash and test evidence. No clean Linux installation was performed; required native artifacts and conditional metadata remain missing.

## Unsupported paths

Legacy LightGBM and database support are not part of the Core example. Do not execute scripts/test_database.py or invoke database loaders for this demo; pytest discovery is confined to tests. No production credentials, private package directories or real financial datasets belong in the environment.
