"""Render one market's notebook from notebooks/template.ipynb.

uv run python -m prediction_markets_forecasting.render aaa_gas_us
uv run python -m prediction_markets_forecasting.render --all
uv run python -m prediction_markets_forecasting.render high_temp_miami --fast   # pipeline check
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .specs import REGISTRY, get_spec

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"


def render(key: str, fast: bool = False) -> Path:
    import papermill as pm

    spec = get_spec(key)
    out = NOTEBOOKS / (f"_fast_{spec.key}.ipynb" if fast else f"{spec.number}_{spec.key}.ipynb")
    t0 = time.time()
    pm.execute_notebook(
        NOTEBOOKS / "template.ipynb",
        out,
        parameters={"MARKET": key, "FAST": fast},
        cwd=str(NOTEBOOKS),
        kernel_name="python3",
        progress_bar=False,
    )
    print(f"rendered {out.name} in {time.time() - t0:.0f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("market", nargs="?", help=f"one of {sorted(REGISTRY)}")
    ap.add_argument("--all", action="store_true", help="render every market in the registry")
    ap.add_argument("--fast", action="store_true", help="3 CV folds (pipeline check, not results)")
    args = ap.parse_args()
    keys = sorted(REGISTRY) if args.all else [args.market]
    if not keys or keys == [None]:
        ap.error("give a market key or --all")
    for k in keys:
        render(k, fast=args.fast)


if __name__ == "__main__":
    main()
