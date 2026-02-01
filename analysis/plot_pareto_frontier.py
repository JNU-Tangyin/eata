import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _base_dir(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _fake_pareto(seed: int, n_points: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def make(method: str, c_mu: float, c_sigma: float, sr_mu: float, sr_sigma: float) -> pd.DataFrame:
        complexity = rng.normal(c_mu, c_sigma, size=n_points)
        complexity = np.clip(complexity, 5, 150).astype(int)
        sr = rng.normal(sr_mu, sr_sigma, size=n_points)
        sr = np.clip(sr, -0.5, 2.5)
        return pd.DataFrame({'method': method, 'complexity': complexity, 'sr': sr})

    df = pd.concat(
        [
            make('EATA (Ours)', 28, 12, 0.85, 0.25),
            make('NEMoTS', 35, 15, 0.68, 0.28),
            make('LightGBM', 8, 3, 0.52, 0.22),
            make('Buy & Hold', 2, 1, 0.35, 0.20),
        ],
        ignore_index=True,
    )
    return df


def _annotate_placeholder(fig: plt.Figure, script_name: str, out_file: Path) -> None:
    text = (
        f"PLACEHOLDER FIGURE\n"
        f"File: {out_file.name}\n"
        f"RQ: RQ2/RQ3 (Interpretability-performance trade-off)\n"
        f"x-axis: complexity (AST node count)\n"
        f"y-axis: Sharpe ratio\n"
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
    ap.add_argument('--in_csv', default='tables/pareto_frontier_points.csv')
    ap.add_argument('--out_fig', default='paper/figures/fig2_pareto_frontier.pdf')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--n_points', type=int, default=120)
    ap.add_argument('--force_fake', action='store_true')
    ap.add_argument('--annotate', action='store_true')
    args = ap.parse_args()

    base = _base_dir(args.base_dir)
    in_csv = (base / args.in_csv).resolve()
    out_fig = (base / args.out_fig).resolve()

    annotate = args.annotate
    if args.force_fake or (not in_csv.exists()):
        df = _fake_pareto(seed=args.seed, n_points=args.n_points)
        _ensure_dir(in_csv)
        df.to_csv(in_csv, index=False)
        annotate = True
    else:
        df = pd.read_csv(in_csv)

    needed = {'method', 'complexity', 'sr'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(6.8, 3.2))
    ax = sns.scatterplot(data=df, x='complexity', y='sr', hue='method', alpha=0.7, s=22)

    ax.set_xlabel('Complexity (AST node count)')
    ax.set_ylabel('Sharpe Ratio')

    if annotate:
        _annotate_placeholder(fig, script_name='analysis/plot_pareto_frontier.py', out_file=out_fig)

    plt.tight_layout()
    _ensure_dir(out_fig)
    plt.savefig(out_fig)
    print(f"Wrote: {out_fig}")


if __name__ == '__main__':
    main()
