# Batch Effects in Raman Spectroscopy of Yeast

This repository contains both a Python package for analyzing batch effects in Raman spectroscopy data and the draft of a scientific publication describing the findings.

## Repository Structure

```text
/pub/                    # Publication draft, figures, and markdown tooling
/src/                    # Python package source
/tests/                  # Python package tests
/artifacts/              # Reference materials (read-only)
```

## Publication

The publication draft is in `/pub/main-text.md`.

## Python Package

The `raman-batch-effects` package provides tools for analyzing batch effects in Raman spectroscopy data.

### Installation

```bash
uv sync
```

### Development

**Publication linting and formatting:**

```bash
cd pub
npm install
npm run lint:all
npm run format
```

**Python linting and formatting:**

```bash
make format
make lint
```

## License

TODO: Add license.
