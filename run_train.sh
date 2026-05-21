#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
uv sync

# Run training
uv run python train.py
