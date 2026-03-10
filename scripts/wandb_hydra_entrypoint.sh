#!/usr/bin/env bash
set -euo pipefail

# W&B sweep agents pass hyperparameters as `--key=value`.
# Hydra overrides for `python -m topobench` must be positional `key=value`.
# This wrapper strips the leading `--` so sweeps work with Hydra.

fixed_args=()
for arg in "$@"; do
  if [[ "$arg" == --*=* ]]; then
    fixed_args+=("${arg:2}")
  else
    fixed_args+=("$arg")
  fi
done

exec python -m topobench "${fixed_args[@]}"

