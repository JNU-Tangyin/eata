import argparse
from pathlib import Path
from typing import Dict, List, Optional

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


def _paper_name(method: str) -> str:
    m = method.strip()
    mapping = {
        'eata': 'EATA (Ours)',
        'EATA (Ours)': 'EATA (Ours)',
        'lightgbm': 'LightGBM',
        'LightGBM': 'LightGBM',
        'buy_and_hold': 'Buy & Hold',
        'Buy & Hold': 'Buy & Hold',
        'nemots': 'NEMoTS',
        'NEMoTS': 'NEMoTS',
    }
    return mapping.get(m, m)


def _method_order() -> List[str]:
    return ['Buy & Hold', 'LightGBM', 'NEMoTS', 'EATA (Ours)']


def _fake_sp500_sr(n_tickers: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = [f"S{i:04d}" for i in range(1, n_tickers + 1)]

    params = {
        'Buy & Hold': (0.35, 0.25),
        'LightGBM': (0.50, 0.28),
        'NEMoTS': (0.62, 0.30),
        'EATA (Ours)': (0.80, 0.30),
    }

    rows: List[Dict] = []
    for method, (mu, sigma) in params.items():
        sr = rng.normal(mu, sigma, size=n_tickers)
        sr = np.clip(sr, -0.5, 2.5)
        for i, t in enumerate(tickers):
            rows.append({'ticker': t, 'method': method, 'sr': float(sr[i])})

    return pd.DataFrame(rows)


def _annotate_placeholder(fig: plt.Figure, script_name: str, out_file: Path) -> None:
    text = (
        f"PLACEHOLDER FIGURE\n"
        f"File: {out_file.name}\n"
        f"RQ: RQ3 (Full S&P 500 generalization)\n"
        f"x-axis: method (Buy & Hold, LightGBM, NEMoTS, EATA (Ours))\n"
        f"y-axis: per-stock Sharpe ratio\n"
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
    ap.add_argument('--in_csv', default='tables/sp500_full_metrics.csv')
    ap.add_argument('--out_fig', default='paper/figures/sp500_full_sharpe_boxplot.pdf')
    ap.add_argument('--n_tickers', type=int, default=500)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--force_fake', action='store_true')
    ap.add_argument('--annotate', action='store_true')
    args = ap.parse_args()

    base = _base_dir(args.base_dir)
    in_csv = (base / args.in_csv).resolve()
    out_fig = (base / args.out_fig).resolve()

    annotate = args.annotate

    if args.force_fake or (not in_csv.exists()):
        df = _fake_sp500_sr(n_tickers=args.n_tickers, seed=args.seed)
        annotate = True
    else:
        df = pd.read_csv(in_csv)
        if 'method' not in df.columns:
            raise ValueError('Missing column: method')
        if 'sr' not in df.columns:
            if 'sharpe_ratio' in df.columns:
                df = df.rename(columns={'sharpe_ratio': 'sr'})
            else:
                raise ValueError('Missing column: sr (or sharpe_ratio)')
        df['method'] = df['method'].map(_paper_name)
        df = df[['ticker', 'method', 'sr']].copy()

    df = df.dropna(subset=['sr'])
    df['method'] = pd.Categorical(df['method'], categories=_method_order(), ordered=True)

    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(8.6, 3.2))
    ax = sns.boxplot(data=df, x='method', y='sr', showfliers=True)

    n_tickers = int(df['ticker'].nunique())
    ax.set_title(f"Per-stock Sharpe distribution over S&P 500 (N={n_tickers})")
    ax.set_xlabel('Method')
    ax.set_ylabel('Sharpe Ratio')
    plt.xticks(rotation=20, ha='right')

    if annotate:
        _annotate_placeholder(fig, script_name='analysis/plot_sp500_full_sharpe_boxplot.py', out_file=out_fig)

    plt.tight_layout()
    _ensure_dir(out_fig)
    plt.savefig(out_fig)
    print(f"Wrote: {out_fig}")


if __name__ == '__main__':
    main()
