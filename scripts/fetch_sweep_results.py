#!/usr/bin/env python3
"""
Fetch W&B sweep results for neuro_tdl/neuro. For each sweep, process all grid
configurations: average val_best_rerun/accuracy only over runs with
dataset.split_params.data_seed in [0, 1, 2], then write a report sorted by
that metric (descending) to sweep_results.txt.

Usage:
  python scripts/fetch_sweep_results.py
  python scripts/fetch_sweep_results.py oyjcr9n8 j5nv0hqh n1ccfebd
  python scripts/fetch_sweep_results.py -o my_report.txt
"""

import argparse
import sys
from collections import defaultdict

try:
    from wandb.apis.public import Api
except ImportError:
    raise SystemExit("Install wandb: pip install wandb")

ENTITY = "neuro_tdl"
PROJECT = "neuro"
METRIC = "val_best_rerun/accuracy"
SEED_KEY = "dataset.split_params.data_seed"
VALID_SEEDS = {0, 1, 2}
DEFAULT_OUTPUT = "sweep_results.txt"
# Parameter keys we ignore when building config groups (not model/dataset hyperparams)
IGNORED_PARAM_KEYS = {"trainer.devices"}


def get_metric_from_run(run, metric_key: str):
    """Get metric from run summary."""
    summary = run.summary._json_dict if hasattr(run.summary, "_json_dict") else dict(run.summary)
    if metric_key in summary:
        return summary[metric_key]
    alt = metric_key.replace("/", ".")
    if alt in summary:
        return summary[alt]
    for k, v in summary.items():
        if "val_best_rerun" in k and "accuracy" in k.lower():
            return v
    return None


def get_config_value(run, key: str):
    """Get config value; support dot-separated nested keys and flat keys."""
    cfg = run.config
    if hasattr(cfg, "_config"):
        cfg = cfg._config
    if not isinstance(cfg, dict):
        cfg = dict(cfg) if hasattr(cfg, "items") else {}
    # Try direct key (W&B sometimes flattens with dots)
    if key in cfg:
        return cfg[key]
    # Walk nested
    d = cfg
    for part in key.split("."):
        if isinstance(d, dict) and part in d:
            d = d[part]
        else:
            d = None
            break
    if d is not None:
        return d
    # Try flattened style (e.g. "dataset.split_params.data_seed" as key in a flat dict)
    for k, v in cfg.items():
        if k == key or (isinstance(k, str) and k.endswith("." + key.split(".")[-1])):
            return v
    return None


def get_swept_param_keys(sweep):
    """Return list of sweep parameter keys to group by (excluding seed and ignored)."""
    config = getattr(sweep, "config", None) or getattr(sweep, "_config", {})
    if not isinstance(config, dict):
        config = dict(config) if hasattr(config, "items") else {}
    params = config.get("parameters") or config.get("config", {}).get("parameters") or {}
    if not isinstance(params, dict):
        params = {}
    keys = [
        k for k in params
        if k != SEED_KEY and k not in IGNORED_PARAM_KEYS
    ]
    return sorted(keys)


def _hashable_val(v):
    """Return a value suitable for grouping (hashable)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return tuple(_hashable_val(x) for x in v)
    return str(v)


def run_key_from_run(run, param_keys):
    """Build a tuple of (param_key, value) pairs for this run for grouping."""
    return tuple((k, _hashable_val(get_config_value(run, k))) for k in param_keys)


def main():
    parser = argparse.ArgumentParser(
        description="Average sweep accuracy over data seeds and write report (descending by accuracy)."
    )
    parser.add_argument(
        "sweep_ids",
        nargs="*",
        default=[],
        help="Sweep IDs to process (if empty, script may use a default list)",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        default=DEFAULT_OUTPUT,
        help=f"Output report file (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    sweep_ids = args.sweep_ids if args.sweep_ids else [
        "oyjcr9n8", "j5nv0hqh", "n1ccfebd", "yqsnesm2"
    ]

    api = Api()
    lines = []

    def out(s=""):
        line = s if isinstance(s, str) else str(s)
        lines.append(line)
        print(line)

    for sweep_id in sweep_ids:
        path = f"{ENTITY}/{PROJECT}/{sweep_id}"
        try:
            sweep = api.sweep(path)
        except Exception as e:
            out(f"\n=== Sweep {sweep_id} ({path}) ===")
            out(f"  Error: {e}")
            continue

        param_keys = get_swept_param_keys(sweep)
        if not param_keys:
            # Fallback: try common sweep param names that appear in run configs
            common = [
                "model", "model.feature_encoder.out_channels",
                "model.backbone.n_layers", "model.backbone.num_layers",
                "optimizer.parameters.lr",
            ]
            for run in sweep.runs:
                if run.state != "finished":
                    break
                param_keys = [k for k in common if get_config_value(run, k) is not None]
                param_keys = sorted(set(param_keys))
                break

        # Group runs by config (all params except seed); only runs with seed in [0,1,2]
        groups = defaultdict(list)  # run_key -> [acc, ...]

        for run in sweep.runs:
            if run.state != "finished":
                continue
            seed = get_config_value(run, SEED_KEY)
            try:
                seed_int = int(seed) if seed is not None else None
            except (TypeError, ValueError):
                seed_int = None
            # Only average over runs with data_seed in [0, 1, 2]; include runs with no seed in config (may be stored elsewhere)
            if seed_int is not None and seed_int not in VALID_SEEDS:
                continue
            acc = get_metric_from_run(run, METRIC)
            if acc is None:
                continue
            run_key = run_key_from_run(run, param_keys)
            groups[run_key].append(acc)

        # Average over seeds per config
        results = []
        for run_key, accs in groups.items():
            n = len(accs)
            mean_acc = sum(accs) / n if n else 0.0
            results.append((run_key, mean_acc, n, accs))

        # Sort by mean val_best_rerun/accuracy descending
        results.sort(key=lambda x: (-x[1], str(x[0])))

        # Report section for this sweep
        out(f"\n{'='*60}")
        out(f"Sweep: {sweep_id}  ({path})")
        out(f"Metric: {METRIC} (averaged over data_seed in {sorted(VALID_SEEDS)})")
        out(f"Grid parameters: {param_keys}")
        out(f"{'='*60}")

        if not results:
            out("  No completed runs with valid seeds found.")
            continue

        # Header: param names + mean_accuracy, n_seeds
        param_names = [k for k, _ in results[0][0]]
        header = "  " + " | ".join(f"{k}" for k in param_names) + f" | mean_{METRIC.replace('/', '_')} | n_seeds"
        out(header)
        out("  " + "-" * (len(header) - 2))

        for run_key, mean_acc, n, accs in results:
            vals = [str(v) for _, v in run_key]
            row = "  " + " | ".join(vals) + f" | {mean_acc:.4f} | {n}"
            out(row)

        out("")

    # Write to file
    report_path = args.output
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    out(f"\nReport written to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
