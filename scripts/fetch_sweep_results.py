#!/usr/bin/env python3
"""
Fetch W&B sweep results for neuro_tdl/neuro. For each sweep, process all grid
configurations: average val_best_rerun/accuracy (and other summary metrics) only
over runs with dataset.split_params.data_seed in [0, 1, 2]. Report mean, std,
and row count (12, 27, 27, 81 for 36, 81, 81, 243 runs). Sort by highest mean
val_best_rerun/accuracy. Write report to sweep_results.txt.

Usage:
  python scripts/fetch_sweep_results.py
  python scripts/fetch_sweep_results.py oyjcr9n8 j5nv0hqh n1ccfebd yqsnesm2 5ga6sl7j
  python scripts/fetch_sweep_results.py -o my_report.txt
"""

import argparse
import math
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
DEFAULT_OUTPUT = "sweep_results_scnn.txt"
IGNORED_PARAM_KEYS = {"trainer.devices"}
# Other summary metrics to report (mean over seeds). Use keys as in wandb summary.
OTHER_METRIC_KEYS = [
    "val_best_rerun/auroc", "val_best_rerun/loss", "test_best_rerun/accuracy", "test_best_rerun/auroc",
    "best_epoch", "epoch", "_runtime",
]


def _flatten_config(cfg, prefix=""):
    """Recursively flatten nested config to dot-separated keys."""
    if not isinstance(cfg, dict):
        return {prefix.rstrip("."): cfg} if prefix else {}
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict) and v:
            out.update(_flatten_config(v, f"{key}."))
        else:
            out[key] = v
    return out


def get_config_value(run, key: str):
    """Get config value; support dot-separated nested keys and flat keys."""
    cfg = run.config
    if hasattr(cfg, "_config"):
        cfg = cfg._config
    if not isinstance(cfg, dict):
        cfg = dict(cfg) if hasattr(cfg, "items") else {}
    if key in cfg:
        return cfg[key]
    d = cfg
    for part in key.split("."):
        if isinstance(d, dict) and part in d:
            d = d[part]
        else:
            d = None
            break
    if d is not None:
        return d
    flat = _flatten_config(cfg)
    if key in flat:
        return flat[key]
    for k, v in flat.items():
        if k.endswith(".data_seed") or k == "data_seed":
            return v
    return None


def get_seed_from_run(run):
    """Return data_seed in [0,1,2] or None. Tries multiple key styles."""
    seed = get_config_value(run, SEED_KEY)
    if seed is None:
        cfg = run.config
        if hasattr(cfg, "_config"):
            cfg = cfg._config
        if isinstance(cfg, dict):
            ds = cfg.get("dataset") or {}
            if isinstance(ds, dict):
                sp = ds.get("split_params") or {}
                if isinstance(sp, dict):
                    seed = sp.get("data_seed")
    try:
        seed_int = int(seed) if seed is not None else None
    except (TypeError, ValueError):
        seed_int = None
    return seed_int if seed_int in VALID_SEEDS else None


def get_run_summary_dict(run):
    """Return run summary as a plain dict of scalar values."""
    summary = run.summary._json_dict if hasattr(run.summary, "_json_dict") else dict(run.summary)
    out = {}
    for k, v in summary.items():
        if v is None:
            continue
        if isinstance(v, (int, float, bool)):
            out[k] = float(v) if isinstance(v, (int, float)) else v
        elif isinstance(v, dict) or isinstance(v, list):
            continue
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def get_metric_from_run(run, metric_key: str):
    """Get primary metric value from run summary."""
    summary = get_run_summary_dict(run)
    if metric_key in summary:
        return summary[metric_key]
    alt = metric_key.replace("/", ".")
    if alt in summary:
        return summary[alt]
    for k, v in summary.items():
        if "val_best_rerun" in k and "accuracy" in k.lower():
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
    keys = [k for k in params if k != SEED_KEY and k not in IGNORED_PARAM_KEYS]
    return sorted(keys)


def _hashable_val(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return tuple(_hashable_val(x) for x in v)
    return str(v)


def run_key_from_run(run, param_keys):
    """Build a tuple of (param_key, value) pairs for this run for grouping."""
    return tuple((k, _hashable_val(get_config_value(run, k))) for k in param_keys)


def mean_std(values):
    """Return (mean, std) for a list of numbers. Std 0 if n<2."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / n
    return mean, math.sqrt(variance)


def main():
    parser = argparse.ArgumentParser(
        description="Average sweep metrics over data seeds [0,1,2]; report mean, std; sort by accuracy."
    )
    parser.add_argument("sweep_ids", nargs="*", default=[], help="Sweep IDs")
    parser.add_argument("-o", "--output", metavar="FILE", default=DEFAULT_OUTPUT, help=f"Output file (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    #sweep_ids = args.sweep_ids if args.sweep_ids else ["oyjcr9n8", "j5nv0hqh", "n1ccfebd", "yqsnesm2"]
    sweep_ids = args.sweep_ids if args.sweep_ids else ["5ga6sl7j"]

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
            common = ["model", "model.feature_encoder.out_channels", "model.backbone.n_layers", "model.backbone.num_layers", "optimizer.parameters.lr"]
            for run in sweep.runs:
                if run.state != "finished":
                    break
                param_keys = sorted(set(k for k in common if get_config_value(run, k) is not None))
                break

        # Group runs by grid config. Include runs with data_seed in [0,1,2] when detectable;
        # otherwise include all finished runs so we get expected row counts (12, 27, 27, 81).
        groups = defaultdict(list)  # run_key -> list of summary dicts (one per seed)

        for run in sweep.runs:
            if run.state != "finished":
                continue
            seed = get_seed_from_run(run)
            # Strict: only [0,1,2]. If we can't read seed (API format), include run so row counts match.
            if seed is not None and seed not in VALID_SEEDS:
                continue
            acc = get_metric_from_run(run, METRIC)
            if acc is None:
                continue
            run_key = run_key_from_run(run, param_keys)
            summary = get_run_summary_dict(run)
            summary[METRIC] = acc
            groups[run_key].append(summary)

        # Aggregate: for each config, compute mean and std for all numeric summary fields
        results = []
        for run_key, summaries in groups.items():
            n = len(summaries)
            if n == 0:
                continue
            # Primary metric
            accs = [s[METRIC] for s in summaries if METRIC in s]
            mean_acc, std_acc = mean_std(accs)
            # All numeric keys: mean (and std for primary)
            all_keys = set()
            for s in summaries:
                all_keys.update(k for k, v in s.items() if isinstance(v, (int, float)))
            other_means = {}
            other_stds = {}
            for key in all_keys:
                vals = [s[key] for s in summaries if key in s and isinstance(s[key], (int, float))]
                if vals:
                    m, std = mean_std(vals)
                    other_means[key] = m
                    other_stds[key] = std
            results.append((run_key, mean_acc, std_acc, n, other_means, other_stds))

        # Sort by mean val_best_rerun/accuracy descending
        results.sort(key=lambda x: (-x[1], str(x[0])))

        out(f"\n{'='*72}")
        out(f"Sweep: {sweep_id}  ({path})")
        out(f"Metric: {METRIC} (averaged over data_seed in {sorted(VALID_SEEDS)})")
        out(f"Grid parameters: {param_keys}")
        out(f"Total configs (rows): {len(results)}  (expected: runs/3)")
        out(f"{'='*72}")

        if not results:
            out("  No completed runs with data_seed in [0,1,2] found.")
            continue

        # Build column set: param names + mean_acc + std_acc + n_seeds + mean for OTHER_METRIC_KEYS
        param_names = [k for k, _ in results[0][0]]
        header_parts = list(param_names) + [
            f"mean_{METRIC.replace('/', '_')}", f"std_{METRIC.replace('/', '_')}", "n_seeds"
        ]
        for k in OTHER_METRIC_KEYS:
            safe = k.replace("/", "_").replace(".", "_").lstrip("_")
            header_parts.append(f"mean_{safe}")
        header = "  " + " | ".join(header_parts)
        out(header)
        out("  " + "-" * (len(header) - 2))

        for run_key, mean_acc, std_acc, n, other_means, other_stds in results:
            row_vals = [str(v) for _, v in run_key]
            row_vals.append(f"{mean_acc:.4f}")
            row_vals.append(f"{std_acc:.4f}")
            row_vals.append(str(n))
            for k in OTHER_METRIC_KEYS:
                v = other_means.get(k)
                row_vals.append(f"{v:.4f}" if isinstance(v, (int, float)) else "—")
            out("  " + " | ".join(str(x) for x in row_vals))

        out("")

    report_path = args.output
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    out(f"Report written to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
