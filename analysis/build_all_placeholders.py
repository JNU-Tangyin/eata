import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _base_dir(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _run(cmd: List[str], cwd: Path) -> None:
    p = subprocess.run(cmd, cwd=str(cwd), text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_dir', default=None)
    ap.add_argument('--force_fake', action='store_true')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--n_tickers', type=int, default=500)
    ap.add_argument('--n_steps', type=int, default=200)
    ap.add_argument('--n_runs', type=int, default=5)
    ap.add_argument('--n_points', type=int, default=120)
    args = ap.parse_args()

    base = _base_dir(args.base_dir)

    py = sys.executable
    force_flag = ['--force_fake'] if args.force_fake else []

    _run(
        [
            py,
            'analysis/build_sp500_full_table.py',
            '--base_dir',
            str(base),
            '--seed',
            str(args.seed),
            '--n_tickers',
            str(args.n_tickers),
            *force_flag,
        ],
        cwd=base,
    )

    _run(
        [
            py,
            'analysis/plot_sp500_full_sharpe_boxplot.py',
            '--base_dir',
            str(base),
            '--seed',
            str(args.seed),
            '--n_tickers',
            str(args.n_tickers),
            *(['--annotate'] if args.force_fake else []),
            *force_flag,
        ],
        cwd=base,
    )

    _run(
        [
            py,
            'analysis/plot_search_efficiency.py',
            '--base_dir',
            str(base),
            '--seed',
            str(args.seed),
            '--n_steps',
            str(args.n_steps),
            '--n_runs',
            str(args.n_runs),
            *force_flag,
        ],
        cwd=base,
    )

    _run(
        [
            py,
            'analysis/plot_pareto_frontier.py',
            '--base_dir',
            str(base),
            '--seed',
            str(args.seed),
            '--n_points',
            str(args.n_points),
            *force_flag,
        ],
        cwd=base,
    )

    print('All placeholder tables/figures built successfully.')


if __name__ == '__main__':
    main()
