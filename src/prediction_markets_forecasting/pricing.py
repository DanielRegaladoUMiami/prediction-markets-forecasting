"""Part 6: forecast distribution → probability of each Kalshi bracket → edge after fees."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .kalshi import bracket_bounds, fee, prob_yes

EDGE_MIN = 0.03  # minimum edge (in $ per $1 contract) after fees to flag a candidate


def model_distribution(errors: np.ndarray, point: float) -> tuple[float, float]:
    """X ~ N(point + mean CV error, std of CV errors)."""
    return float(point + np.mean(errors)), float(np.std(errors, ddof=1))


def implied_distribution(g: pd.DataFrame, integer: bool) -> tuple[float, float]:
    """Mean and σ of the underlying implied by one event's ladder of mid prices.

    - Only "greater" markets (a survival curve, e.g. gas): the drop in P(X > K) between strikes is
      the probability of landing between them.
    - Mixed "less"/"between"/"greater" markets (a partition, e.g. temperatures): each mid is the
      probability of its own bracket; mids are normalized to sum to 1.
    """
    if (g["strike_type"] == "greater").all():
        g = g.sort_values("floor")
        k = g["floor"].to_numpy()
        if len(k) < 2:
            return np.nan, np.nan
        p_above = np.minimum.accumulate(np.clip(g["mid"].to_numpy(), 0, 1))
        edges = np.concatenate([[k[0] - (k[1] - k[0])], k, [k[-1] + (k[-1] - k[-2])]])
        pmf = -np.diff(np.concatenate([[1.0], p_above, [0.0]]))
        centers = (edges[:-1] + edges[1:]) / 2
    else:
        bounds = np.array([bracket_bounds(r, integer) for _, r in g.iterrows()])
        widths = bounds[:, 1] - bounds[:, 0]
        w = np.nanmedian(widths[np.isfinite(widths)]) if np.isfinite(widths).any() else 1.0
        lo = np.where(np.isfinite(bounds[:, 0]), bounds[:, 0], bounds[:, 1] - w)
        hi = np.where(np.isfinite(bounds[:, 1]), bounds[:, 1], bounds[:, 0] + w)
        centers, pmf = (lo + hi) / 2, g["mid"].to_numpy()
    pmf = np.clip(pmf, 0, None)
    if pmf.sum() == 0:
        return np.nan, np.nan
    pmf = pmf / pmf.sum()
    mu = float((centers * pmf).sum())
    return mu, float(np.sqrt(((centers - mu) ** 2 * pmf).sum()))


def edge_table(
    g: pd.DataFrame,
    integer: bool,
    model: tuple[float, float] | None,
    nowcast: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Per market: model / nowcast probabilities, edge of buying YES or NO, and the signal.

    A signal requires the model's edge > EDGE_MIN; when a nowcast exists it must agree.
    """
    g = g.copy()
    for name, dist in (("model", model), ("now", nowcast)):
        if dist is None or not np.all(np.isfinite(dist)):
            g[f"p_{name}"] = np.nan
        else:
            g[f"p_{name}"] = [prob_yes(r, dist[0], dist[1], integer) for _, r in g.iterrows()]
        g[f"edge_yes_{name}"] = g[f"p_{name}"] - g["yes_ask"] - fee(g["yes_ask"])
        g[f"edge_no_{name}"] = (1 - g[f"p_{name}"]) - g["no_ask"] - fee(g["no_ask"])
    has_now = g["p_now"].notna()
    yes = (g["edge_yes_model"] > EDGE_MIN) & (~has_now | (g["edge_yes_now"] > EDGE_MIN))
    no = (g["edge_no_model"] > EDGE_MIN) & (~has_now | (g["edge_no_now"] > EDGE_MIN))
    g["signal"] = np.select([yes, no], ["BUY YES", "BUY NO"], "")
    return g


def bracket_label(r: pd.Series) -> str:
    if r["strike_type"] == "greater":
        return f"> {r['floor']:g}"
    if r["strike_type"] == "less":
        return f"< {r['cap']:g}"
    return f"{r['floor']:g}–{r['cap']:g}"
