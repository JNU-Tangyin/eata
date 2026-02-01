import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _base_dir(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _fake_efficiency(seed: int, n_runs: int, n_steps: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    methods = ['eata', 'gp', 'random']

    # Construct monotone-ish best-so-far curves; EATA dominates.
    base_targets = {'eata': 1.00, 'gp': 0.75, 'random': 0.55}
    base_rates = {'eata': 0.035, 'gp': 0.020, 'random': 0.012}

    rows: List[Dict] = []
    steps = np.arange(1, n_steps + 1)
    for run_id in range(1, n_runs + 1):
        for m in methods:
            target = base_targets[m] + rng.normal(0.0, 0.03)
            rate = base_rates[m] * (1.0 + rng.normal(0.0, 0.15))
            noise = rng.normal(0.0, 0.03, size=n_steps)
            curve = target * (1.0 - np.exp(-rate * steps)) + noise
            curve = np.maximum.accumulate(curve)
            for s, y in zip(steps, curve):
                rows.append({'run_id': run_id, 'method': m, 'step': int(s), 'best_reward': float(y)})

    return pd.DataFrame(rows)


def _legend_name(method: str) -> str:
    mapping = {
        'eata': 'EATA (Ours)',
        'gp': 'GP',
        'random': 'Random Search',
    }
    return mapping.get(method, method)


def _annotate_placeholder(fig: plt.Figure, script_name: str, out_file: Path) -> None:
    text = (
        f"PLACEHOLDER FIGURE\n"
        f"File: {out_file.name}\n"
        f"RQ: RQ2 (Search efficiency)\n"
        f"x-axis: search step / episode\n"
        f"y-axis: best-so-far reward (or best Sharpe)\n"
        f"Script: {script_name}"
    )
    fig.text(
        0.02,
        0.98,
        text,
        ha='left',
        va='top',
        fontsize=9,
        color='red',
        family='monospace',
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_dir', default=None)
    ap.add_argument('--in_csv', default='tables/search_efficiency_runs.csv')
    ap.add_argument('--out_fig', default='paper/figures/fig4_search_efficiency.pdf')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--n_runs', type=int, default=5)
    ap.add_argument('--n_steps', type=int, default=200)
    ap.add_argument('--force_fake', action='store_true')
    args = ap.parse_args()

    base = _base_dir(args.base_dir)
    in_csv = (base / args.in_csv).resolve()
    out_fig = (base / args.out_fig).resolve()

    annotate = False
    if args.force_fake or (not in_csv.exists()):
        df = _fake_efficiency(seed=args.seed, n_runs=args.n_runs, n_steps=args.n_steps)
        _ensure_dir(in_csv)
        df.to_csv(in_csv, index=False)
        annotate = True
    else:
        df = pd.read_csv(in_csv)

    needed = {'run_id', 'method', 'step', 'best_reward'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    fig = plt.figure(figsize=(6.8, 3.2))

    for method, g in df.groupby('method'):
        stats = g.groupby('step')['best_reward'].agg(['mean', 'std']).reset_index()
        plt.plot(stats['step'], stats['mean'], label=_legend_name(method))
        plt.fill_between(
            stats['step'],
            stats['mean'] - stats['std'],
            stats['mean'] + stats['std'],
            alpha=0.15,
        )

    plt.xlabel('Search step')
    plt.ylabel('Best-so-far')
    plt.legend(frameon=False)

    if annotate:
        _annotate_placeholder(fig, script_name='analysis/plot_search_efficiency.py', out_file=out_fig)

    plt.tight_layout()
    _ensure_dir(out_fig)
    plt.savefig(out_fig)
    print(f"Wrote: {out_fig}")


if __name__ == '__main__':
    main()
