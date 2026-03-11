#!/usr/bin/env python3
"""
Fetch W&B sweep results for neuro_tdl/neuro and average val_best_rerun/accuracy
over the 3 data seeds (0, 1, 2) for completed sweeps.

Usage:
  python scripts/fetch_sweep_results.py
  # Or with sweep IDs as args:
  python scripts/fetch_sweep_results.py oyjcr9n8 j5nv0hqh n1ccfebd
  # Save output to a file:
  python scripts/fetch_sweep_results.py -o sweep_results.txt
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
DEFAULT_SWEEP_IDS = ["oyjcr9n8", "j5nv0hqh", "n1ccfebd", "yqsnesm2"] # deepset, ast, cwn, gnn (gcn)


def get_metric_from_run(run, metric_key: str):
    """Get metric from run summary; try exact key and key with slashes replaced."""
    summary = run.summary._json_dict if hasattr(run.summary, "_json_dict") else dict(run.summary)
    if metric_key in summary:
        return summary[metric_key]
    # Try alternate key (e.g. with dots)
    alt = metric_key.replace("/", ".")
    if alt in summary:
        return summary[alt]
    for k, v in summary.items():
        if "val_best_rerun" in k and "accuracy" in k.lower():
            return v
    return None


def get_config_value(run, key: str):
    """Get config value; support nested keys like 'dataset.split_params.data_seed'."""
    c = run.config
    if isinstance(c, dict):
        for part in key.split("."):
            c = c.get(part, {})
        return c if not isinstance(c, dict) or key.split(".")[-1] in c else None
    # Config object
    try:
        return dict(c).get(key) or getattr(c, key.split(".")[-1], None)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Average sweep accuracy over seeds")
    parser.add_argument(
        "sweep_ids",
        nargs="*",
        default=DEFAULT_SWEEP_IDS,
        help="Sweep IDs (default: oyjcr9n8 j5nv0hqh n1ccfebd)",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Save output to this .txt file",
    )
    args = parser.parse_args()

    out_file = None
    if args.output:
        out_file = open(args.output, "w")
        class Tee:
            def __init__(self, *targets):
                self.targets = targets
            def write(self, data):
                for t in self.targets:
                    t.write(data)
                    t.flush()
            def flush(self):
                for t in self.targets:
                    t.flush()
        sys.stdout = Tee(sys.__stdout__, out_file)
        sys.stderr = Tee(sys.__stderr__, out_file)

    try:
        _run_sweep_fetch(args, out_file)
    finally:
        if out_file is not None:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            out_file.close()
    return 0


def _run_sweep_fetch(args, out_file):
    api = Api()
    all_aggregated = []

    for sweep_id in args.sweep_ids:
        path = f"{ENTITY}/{PROJECT}/{sweep_id}"
        print(f"\n=== Sweep {sweep_id} ({path}) ===")
        try:
            sweep = api.sweep(path)
        except Exception as e:
            print(f"  Skip: {e}")
            continue

        # Group runs by (out_channels, n_layers, lr); collect (seed -> accuracy)
        groups = defaultdict(list)  # group_key -> [(seed, acc), ...]

        for run in sweep.runs:
            if run.state != "finished":
                continue
            acc = get_metric_from_run(run, METRIC)
            if acc is None:
                continue
            try:
                seed = get_config_value(run, SEED_KEY)
            except Exception:
                seed = None
            # Group key: other hyperparams (so we average over seeds)
            try:
                oc = run.config.get("model.feature_encoder.out_channels") or run.config.get("model", {}).get("feature_encoder", {}).get("out_channels")
                nl = run.config.get("model.backbone.n_layers") or run.config.get("model", {}).get("backbone", {}).get("n_layers")
                lr = run.config.get("optimizer.parameters.lr") or run.config.get("optimizer", {}).get("parameters", {}).get("lr")
            except Exception:
                oc = nl = lr = None
            # Flatten config if nested
            cfg = dict(run.config) if hasattr(run.config, "items") else getattr(run.config, "_config", {}) or {}
            def dig(d, *keys):
                for k in keys:
                    if isinstance(d, dict) and k in d:
                        d = d[k]
                    else:
                        return None
                return d
            oc = oc or dig(cfg, "model", "feature_encoder", "out_channels")
            nl = nl or dig(cfg, "model", "backbone", "n_layers")
            lr = lr or dig(cfg, "optimizer", "parameters", "lr")
            if oc is None and nl is None:
                # Fallback: use run name or id as group
                group_key = (run.id, run.name)
            else:
                group_key = (oc, nl, lr)
            groups[group_key].append((seed, acc))

        # Average over seeds per group
        for group_key, pairs in sorted(groups.items()):
            seeds = [p[0] for p in pairs]
            accs = [p[1] for p in pairs]
            n = len(accs)
            avg_acc = sum(accs) / n if n else 0
            all_aggregated.append((sweep_id, group_key, n, avg_acc, accs, seeds))
            if len(group_key) == 3:
                print(f"  out_channels={group_key[0]}, n_layers={group_key[1]}, lr={group_key[2]}: "
                      f"n_runs={n}, mean({METRIC})={avg_acc:.4f}  (values: {[round(a, 4) for a in accs]})")
            else:
                print(f"  group={group_key}: n_runs={n}, mean({METRIC})={avg_acc:.4f}  (values: {[round(a, 4) for a in accs]})")

    # Summary across sweeps: average of per-seed-group means per (oc, n_layers, lr)
    print("\n--- Summary (averaged over 3 seeds per config) ---")
    by_config = defaultdict(list)
    for sweep_id, group_key, n, avg_acc, accs, seeds in all_aggregated:
        if len(group_key) == 3:
            by_config[group_key].append((sweep_id, avg_acc, n))
    def sort_key(item):
        (oc, nl, lr), _ = item
        return ((oc or 0), (nl if nl is not None else -1), (lr or 0))

    for (oc, nl, lr), sweep_results in sorted(by_config.items(), key=sort_key):
        means = [r[1] for r in sweep_results]
        grand_mean = sum(means) / len(means) if means else 0
        print(f"  out_channels={oc}, n_layers={nl}, lr={lr}: "
              f"mean_accuracy={grand_mean:.4f}  (from {len(sweep_results)} sweeps: {[round(m, 4) for m in means]})")


if __name__ == "__main__":
    raise SystemExit(main())
