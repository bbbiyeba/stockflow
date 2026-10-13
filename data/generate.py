"""Generate a reproducible synthetic dataset for a multi-branch distributor.

The generator simulates *true* daily demand for every branch x SKU series, then runs an
(s, S) replenishment simulation with deliberately imperfect planner parameters. The
simulation produces sales (demand censored by stock), DC receipts, ad-hoc historical
transfers, stock adjustments and weekly on-hand snapshots, so the inventory ledger is
internally consistent with sales. Data-quality problems are then injected into the
extracts (missing days, outliers, returns, duplicates, bad adjustments, missing
snapshots) so that the cleaning pipeline in ``src/prep.py`` has real work to do.

Scale: SKUs are processed in chunks of roughly ``data.chunk_series`` series. A chunk always
holds complete SKUs (every branch carrying them), so within-SKU transfers stay inside a
chunk and memory is bounded by the chunk, not the network. ``data.raw_format: parquet``
writes ``sales_daily/`` and ``inventory_txn/`` as Parquet datasets (one file per chunk);
``csv`` writes single CSV files.

Outputs (``paths.raw_dir`` in ``config.yaml``)::

    branches.csv, skus.csv, promotions.csv, transfer_costs.csv, open_purchase_orders.csv,
    sales_daily.(csv|/), inventory_txn.(csv|/), _truth.npz (small runs only; validation only)

Usage::

    python data/generate.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import holidays as hol
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import get_paths, load_config  # noqa: E402
from src.runlog import get_logger  # noqa: E402

log = get_logger("generate")

# (city, region, lat, lon) - branch locations of the fictional distributor
CITIES: list[tuple[str, str, float, float]] = [
    ("Boston", "Northeast", 42.36, -71.06),
    ("Hartford", "Northeast", 41.76, -72.67),
    ("Newark", "Northeast", 40.74, -74.17),
    ("Philadelphia", "Northeast", 39.95, -75.17),
    ("Baltimore", "Northeast", 39.29, -76.61),
    ("Pittsburgh", "Northeast", 40.44, -79.99),
    ("Charlotte", "Southeast", 35.23, -80.84),
    ("Atlanta", "Southeast", 33.75, -84.39),
    ("Nashville", "Southeast", 36.16, -86.78),
    ("Orlando", "Southeast", 28.54, -81.38),
    ("Tampa", "Southeast", 27.95, -82.46),
    ("Raleigh", "Southeast", 35.78, -78.64),
    ("Chicago", "Midwest", 41.88, -87.63),
    ("Columbus", "Midwest", 39.96, -83.00),
    ("Indianapolis", "Midwest", 39.77, -86.16),
    ("Detroit", "Midwest", 42.33, -83.05),
    ("Milwaukee", "Midwest", 43.04, -87.91),
    ("St. Louis", "Midwest", 38.63, -90.20),
    ("Dallas", "Southwest", 32.78, -96.80),
    ("Houston", "Southwest", 29.76, -95.37),
    ("Austin", "Southwest", 30.27, -97.74),
    ("Phoenix", "Southwest", 33.45, -112.07),
    ("Denver", "West", 39.74, -104.99),
    ("Salt Lake City", "West", 40.76, -111.89),
    ("Sacramento", "West", 38.58, -121.49),
]

# category -> (share of SKUs, median unit cost, annual seasonality amplitude,
#              peak day-of-year, base obsolescence risk, has shelf life)
CATEGORIES: dict[str, tuple[float, float, float, int, float, bool]] = {
    "Electrical": (0.18, 22.0, 0.08, 120, 0.04, False),
    "Plumbing": (0.16, 18.0, 0.18, 25, 0.04, False),
    "HVAC": (0.14, 55.0, 0.45, 200, 0.06, False),
    "Fasteners": (0.12, 4.0, 0.05, 150, 0.02, False),
    "Safety": (0.10, 15.0, 0.12, 110, 0.05, False),
    "Tools": (0.12, 75.0, 0.10, 340, 0.05, False),
    "Adhesives & Chemicals": (0.10, 12.0, 0.22, 190, 0.12, True),
    "Janitorial": (0.08, 9.0, 0.06, 60, 0.08, True),
}

# Monday..Sunday demand profile of a B2B branch (mostly closed on Sunday)
WEEKDAY_PROFILE = np.array([1.15, 1.20, 1.15, 1.10, 1.00, 0.35, 0.05])
WEEKDAY_PROFILE = WEEKDAY_PROFILE / WEEKDAY_PROFILE.mean()


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in km (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def make_branches(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Branch master (capacity is filled in after the inventory simulation).

    Beyond the 25 base cities, extra branches are placed near a base city (satellite
    branches in the same metro region) so the network can be scaled up for load tests.
    """
    rows = [CITIES[i % len(CITIES)] for i in range(n)]
    jitter = np.where(np.arange(n) < len(CITIES), 0.0, 1.0)
    df = pd.DataFrame(rows, columns=["branch_name", "region", "lat", "lon"])
    df["branch_name"] = [
        nm if i < len(CITIES) else f"{nm} {i // len(CITIES) + 1}" for i, nm in enumerate(df["branch_name"])
    ]
    df["lat"] = (df["lat"] + jitter * rng.normal(0, 0.6, n)).round(3)
    df["lon"] = (df["lon"] + jitter * rng.normal(0, 0.8, n)).round(3)
    df.insert(0, "branch_id", [f"B{i + 1:02d}" if n < 100 else f"B{i + 1:03d}" for i in range(n)])
    tiers = rng.choice(["Large", "Medium", "Small"], size=n, p=[0.25, 0.45, 0.30])
    tier_mult = {"Large": 1.7, "Medium": 1.0, "Small": 0.55}
    df["size_tier"] = tiers
    df["demand_multiplier"] = [round(tier_mult[t] * rng.lognormal(0, 0.15), 3) for t in tiers]
    return df


def make_skus(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """SKU master without ABC class (assigned from simulated revenue later)."""
    cats = list(CATEGORIES)
    shares = np.array([CATEGORIES[c][0] for c in cats])
    cat = rng.choice(cats, size=n, p=shares / shares.sum())
    unit_cost = np.array([CATEGORIES[c][1] for c in cat]) * rng.lognormal(0, 0.8, n)
    unit_cost = np.clip(unit_cost, 0.5, 900).round(2)

    # demand archetype drives the network-level daily rate per branch
    archetype = rng.choice(["fast", "medium", "slow", "lumpy"], size=n, p=[0.30, 0.30, 0.30, 0.10])
    rate = np.where(
        archetype == "fast",
        rng.lognormal(np.log(6.0), 0.7, n),
        np.where(
            archetype == "medium",
            rng.lognormal(np.log(1.2), 0.5, n),
            np.where(archetype == "slow", rng.lognormal(np.log(0.12), 0.6, n), rng.lognormal(np.log(0.04), 0.4, n)),
        ),
    )
    # cheaper items sell in higher volumes
    rate = rate * np.clip((20.0 / unit_cost) ** 0.25, 0.4, 2.5)

    case_pack = np.where(
        unit_cost < 5, rng.choice([12, 24, 50], n), np.where(unit_cost < 25, rng.choice([1, 6, 12], n), 1)
    )
    shelf = np.array([CATEGORIES[c][5] for c in cat])
    shelf_life = np.where(shelf, rng.choice([365, 540, 730, 1095], n), np.nan)
    obs = np.array([CATEGORIES[c][4] for c in cat]) * rng.lognormal(0, 0.4, n)

    return pd.DataFrame(
        {
            "sku_id": [f"S{i + 1:04d}" for i in range(n)],
            "category": cat,
            "unit_cost": unit_cost,
            "list_price": (unit_cost * rng.uniform(1.3, 1.8, n)).round(2),
            "holding_cost_rate": rng.uniform(0.18, 0.30, n).round(3),
            "lead_time_days": rng.choice([3, 5, 7, 10, 14, 21], n, p=[0.15, 0.25, 0.25, 0.15, 0.12, 0.08]),
            "case_pack": case_pack,
            "shelf_life_days": shelf_life,
            "obsolescence_risk": np.clip(obs, 0.005, 0.5).round(3),
            "_archetype": archetype,
            "_rate": rate,
        }
    )


def make_promotions(
    rng: np.random.Generator, skus: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Company-wide promotions; the calendar extends past the history so it is known for the forecast."""
    rows = []
    promo_skus = skus.loc[rng.random(len(skus)) < 0.35, "sku_id"]
    span = (end - start).days
    for sku in promo_skus:
        n_promos = int(rng.integers(5, 11))  # over ~3.2 years
        for st in np.sort(rng.choice(span - 20, n_promos, replace=False)):
            length = int(rng.integers(7, 15))
            rows.append(
                (
                    sku,
                    start + pd.Timedelta(days=int(st)),
                    start + pd.Timedelta(days=int(st) + length - 1),
                    round(float(rng.uniform(0.10, 0.25)), 2),
                )
            )
    df = pd.DataFrame(rows, columns=["sku_id", "start_date", "end_date", "discount_pct"])
    # drop overlapping promos for the same SKU
    df = df.sort_values(["sku_id", "start_date"])
    prev_end = df.groupby("sku_id")["end_date"].shift()
    return df[prev_end.isna() | (df["start_date"] > prev_end)].reset_index(drop=True)


def holiday_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """Branches are closed on major US holidays and quiet over the year-end break."""
    us = hol.US(years=sorted(set(dates.year)))
    closed = {
        "New Year's Day",
        "Memorial Day",
        "Independence Day",
        "Labor Day",
        "Thanksgiving",
        "Thanksgiving Day",
        "Christmas Day",
    }
    mult = np.ones(len(dates))
    for i, d in enumerate(dates):
        name = us.get(d.date())
        if name and any(c in name for c in closed) and "observed" not in name.lower():
            mult[i] = 0.03
        elif (d.month == 12 and d.day >= 26) or (d.month == 1 and d.day == 2):
            mult[i] = 0.6
        elif name:  # minor federal holidays: slightly quieter
            mult[i] = 0.85
    return mult


def make_assortment(
    rng: np.random.Generator, skus: pd.DataFrame, n_branches: int, coverage: tuple[float, float]
) -> np.ndarray:
    """Boolean [sku, branch] matrix: faster SKUs are carried in more branches (>= 3 each)."""
    rank = skus["_rate"].rank(pct=True).to_numpy()
    p_carry = coverage[0] + (coverage[1] - coverage[0]) * rank
    carried = rng.random((len(skus), n_branches)) < p_carry[:, None]
    for i in np.nonzero(carried.sum(axis=1) < 3)[0]:
        carried[i, rng.choice(n_branches, 3, replace=False)] = True
    return carried


def build_demand(
    rng: np.random.Generator,
    branches: pd.DataFrame,
    skus: pd.DataFrame,
    promos: pd.DataFrame,
    dates: pd.DatetimeIndex,
    carried: np.ndarray,
    holiday: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Series index and an int32 demand matrix [series, day] for the SKUs in ``skus``.

    ``carried`` is the assortment for exactly these SKUs (rows aligned with ``skus``).
    Multiplicative components are applied in place in float32 to bound memory.
    """
    s_idx, b_idx = np.nonzero(carried)
    series = pd.DataFrame(
        {
            "branch_id": branches["branch_id"].to_numpy()[b_idx],
            "sku_id": skus["sku_id"].to_numpy()[s_idx],
            "_b": b_idx,
            "_s": s_idx,
        }
    )
    n, T = len(series), len(dates)
    t_years = (np.arange(T) / 365.25).astype(np.float32)

    base = (
        skus["_rate"].to_numpy()[s_idx] * branches["demand_multiplier"].to_numpy()[b_idx] * rng.lognormal(0, 0.35, n)
    ).astype(np.float32)
    growth = (rng.normal(0.03, 0.08, len(skus))[s_idx] + rng.normal(0, 0.04, n)).astype(np.float32)
    mean = np.exp(growth[:, None] * t_years[None, :])  # trend
    mean *= base[:, None]

    cat = skus["category"].to_numpy()[s_idx]
    amp = (np.array([CATEGORIES[c][2] for c in cat]) * rng.uniform(0.7, 1.3, n)).astype(np.float32)
    peak = (np.array([CATEGORIES[c][3] for c in cat]) + rng.normal(0, 10, n)).astype(np.float32)
    doy = dates.dayofyear.to_numpy().astype(np.float32)
    mean *= 1 + amp[:, None] * np.cos(2 * np.pi * (doy[None, :] - peak[:, None]) / 365.25)
    mean *= (WEEKDAY_PROFILE[dates.dayofweek.to_numpy()] * holiday).astype(np.float32)[None, :]

    # promotions: uplift during the window, small pull-forward dip afterwards (SKU level)
    promo = np.ones((len(skus), T), dtype=np.float32)
    sku_pos = {s: i for i, s in enumerate(skus["sku_id"])}
    elasticity = rng.uniform(2.0, 4.0, len(skus))
    for r in promos.itertuples():
        i = sku_pos[r.sku_id]
        a, b = (r.start_date - dates[0]).days, (r.end_date - dates[0]).days
        if b < 0 or a >= T:
            continue
        promo[i, max(a, 0) : min(b + 1, T)] *= (1 - r.discount_pct) ** (-elasticity[i])
        promo[i, min(b + 1, T) : min(b + 8, T)] *= 0.85
    mean *= promo[s_idx]
    del promo

    # lifecycle events that create stagnant stock
    for i in np.nonzero(rng.random(len(skus)) < 0.05)[0]:  # SKU phase-out
        st = T - int(rng.integers(150, 330))
        ramp = np.clip(1 - (np.arange(T) - st) / 90, 0.0, 1.0).astype(np.float32)
        mean[s_idx == i] *= ramp[None, :]
    for j in np.nonzero(rng.random(n) < 0.04)[0]:  # a branch loses its customer
        mean[j, T - int(rng.integers(100, 220)) :] *= 0.02

    archetype = skus["_archetype"].to_numpy()[s_idx]
    shape = np.where(archetype == "fast", 4.0, 2.0).astype(np.float32)[:, None]
    noise = rng.gamma(shape, 1 / shape, (n, T)).astype(np.float32)
    noise *= mean
    demand = rng.poisson(noise).astype(np.int32)
    del noise
    lumpy = archetype == "lumpy"
    if lumpy.any():
        occ = rng.random((int(lumpy.sum()), T)) < np.clip(mean[lumpy] / 6.0, 0, 1)
        demand[lumpy] = occ * rng.geometric(1 / 6.0, (int(lumpy.sum()), T))
    return series, demand


def simulate_inventory(
    rng: np.random.Generator,
    series: pd.DataFrame,
    skus: pd.DataFrame,
    review_dow_b: np.ndarray,
    branch_bias_b: np.ndarray,
    demand: np.ndarray,
    dates: pd.DatetimeIndex,
    transit: np.ndarray,
) -> dict[str, Any]:
    """(s, S) replenishment with stale, biased planner parameters for one chunk of SKUs.

    Returns sales (censored demand), the end-of-day on-hand matrix and transaction lists.
    """
    n, T = demand.shape
    s_idx = series["_s"].to_numpy()
    b_idx = series["_b"].to_numpy()
    lead = skus["lead_time_days"].to_numpy()[s_idx]
    pack = skus["case_pack"].to_numpy()[s_idx]
    review_dow = review_dow_b[b_idx]  # each branch orders on one weekday
    misest = rng.lognormal(0, 0.45, n) * branch_bias_b[b_idx]  # planner error + branch hoarding bias

    def policy(rate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cover = lead + 7
        s_lvl = rate * cover + 1.65 * np.sqrt(rate * cover) * 1.3
        S_lvl = s_lvl + rate * 14 + 1
        return np.floor(s_lvl), np.ceil(S_lvl)

    planner_rate = demand[:, :90].mean(axis=1) * misest
    s_lvl, S_lvl = policy(planner_rate)
    on_hand = S_lvl.copy()
    pipeline = np.zeros((n, T + 40), dtype=np.float32)
    on_order = np.zeros(n)  # running total of the pipeline
    sales = np.zeros((n, T), dtype=np.int32)
    eod = np.zeros((n, T), dtype=np.int32)

    receipts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    adjustments: list[tuple[np.ndarray, int, np.ndarray]] = []
    transfers: list[tuple[int, int, int, int, int]] = []  # (src_series, dst_series, day_out, day_in, qty)

    bulk = np.nonzero(rng.random(n) < 0.05)[0]  # end-of-quarter bulk buys -> excess
    bulk_day = T - rng.integers(30, 150, len(bulk))
    series_by_sku = series.groupby("_s").indices
    sku_keys = np.array(list(series_by_sku))
    dows = dates.dayofweek.to_numpy()
    n_transfer_skus = max(1, round(12 * len(skus) / 300))

    def schedule(idx: np.ndarray, day: np.ndarray, q: np.ndarray) -> None:
        np.add.at(pipeline, (idx, day), q)
        np.add.at(on_order, idx, q)

    for t in range(T):
        if t > 0 and t % 91 == 0:  # quarterly planner refresh on trailing sales
            planner_rate = sales[:, max(0, t - 180) : t].mean(axis=1) * misest
            s_lvl, S_lvl = policy(planner_rate)

        arriving = pipeline[:, t].astype(np.float64)
        on_hand += arriving
        on_order -= arriving
        sold = np.minimum(on_hand, demand[:, t])
        sales[:, t] = sold
        on_hand -= sold

        shrink = (rng.random(n) < 0.002) & (on_hand > 0)
        if shrink.any():
            q = -np.minimum(rng.integers(1, 4, int(shrink.sum())), on_hand[shrink])
            on_hand[shrink] += q
            adjustments.append((np.nonzero(shrink)[0], t, q))
        found = rng.random(n) < 0.0004
        if found.any():
            q = rng.integers(1, 3, int(found.sum()))
            on_hand[found] += q
            adjustments.append((np.nonzero(found)[0], t, q))

        hit = bulk[bulk_day == t]
        if len(hit):
            q = np.ceil((planner_rate[hit] * rng.uniform(45, 90, len(hit)) + 2) / pack[hit]) * pack[hit]
            schedule(hit, np.full(len(hit), t + 1), q)
            receipts.append((hit, np.full(len(hit), t + 1), q))

        order = (review_dow == dows[t]) & (on_hand + on_order <= s_lvl) & (S_lvl > 0)
        if order.any():
            idx = np.nonzero(order)[0]
            position = on_hand[idx] + on_order[idx]
            q = np.maximum(np.ceil((S_lvl[idx] - position) / pack[idx]) * pack[idx], pack[idx])
            arrive = np.minimum(t + lead[idx] + rng.integers(-1, 4, len(idx)), T + 39)
            schedule(idx, arrive, q)
            receipts.append((idx, arrive, q))

        if dows[t] == 2:  # ad-hoc historical transfers between branches
            for sku in rng.choice(sku_keys, min(n_transfer_skus, len(sku_keys)), replace=False):
                members = series_by_sku[sku]
                if len(members) < 2:
                    continue
                over = members[on_hand[members] > 1.5 * S_lvl[members] + 2]
                under = members[on_hand[members] < s_lvl[members]]
                if len(over) and len(under):
                    src, dst = int(rng.choice(over)), int(rng.choice(under))
                    qty = int(min(on_hand[src] - S_lvl[src], S_lvl[dst] - on_hand[dst]))
                    if qty > 0:
                        days = int(transit[b_idx[src], b_idx[dst]])
                        on_hand[src] -= qty
                        if t + days < T + 40:
                            schedule(np.array([dst]), np.array([t + days]), np.array([qty]))
                        transfers.append((src, dst, t, t + days, qty))
        eod[:, t] = on_hand

    return {"sales": sales, "eod": eod, "receipts": receipts, "adjustments": adjustments, "transfers": transfers}


def assign_abc(skus: pd.DataFrame, value_by_sku: np.ndarray) -> np.ndarray:
    """ABC by cost value of the last 365 days of sales (80% / 15% / 5%)."""
    order = np.argsort(-value_by_sku)
    cum = np.cumsum(value_by_sku[order]) / max(value_by_sku.sum(), 1e-9)
    cls = np.empty(len(skus), dtype=object)
    cls[order] = np.where(cum <= 0.80, "A", np.where(cum <= 0.95, "B", "C"))
    return cls


def sku_chunks(carried: np.ndarray, target_series: int) -> list[np.ndarray]:
    """Split SKU indices into consecutive chunks of about ``target_series`` series each."""
    per_sku = carried.sum(axis=1)
    bounds = np.searchsorted(np.cumsum(per_sku), np.arange(target_series, per_sku.sum(), target_series))
    return [c for c in np.split(np.arange(len(per_sku)), np.unique(bounds + 1)) if len(c)]


class RawWriter:
    """Appends chunk outputs to CSV files or per-chunk Parquet files."""

    def __init__(self, raw: Path, fmt: str) -> None:
        if fmt not in ("csv", "parquet"):
            raise ValueError(f"data.raw_format must be 'csv' or 'parquet', got {fmt!r}")
        self.raw, self.fmt = raw, fmt
        self.started: set[str] = set()
        for name in ("sales_daily", "inventory_txn"):
            (raw / f"{name}.csv").unlink(missing_ok=True)
            d = raw / name
            if d.is_dir():
                for f in d.glob("*.parquet"):
                    f.unlink()

    def write(self, name: str, df: pd.DataFrame, chunk: int) -> None:
        if self.fmt == "csv":
            df.to_csv(
                self.raw / f"{name}.csv", mode="a", header=name not in self.started, index=False, date_format="%Y-%m-%d"
            )
        else:
            (self.raw / name).mkdir(exist_ok=True)
            df.to_parquet(self.raw / name / f"part-{chunk:04d}.parquet", index=False)
        self.started.add(name)


def main(config_path: str | None = None) -> None:
    """Generate every synthetic extract (and the ground truth) described in the config."""
    cfg = load_config(config_path)
    dc = cfg["data"]
    dq = dc["quality_issues"]
    paths = get_paths(cfg)
    rng = np.random.default_rng(dc["seed"])
    t0 = time.time()

    dates = pd.date_range(dc["start_date"], dc["end_date"], freq="D")
    T = len(dates)
    branches = make_branches(rng, dc["n_branches"])
    skus = make_skus(rng, dc["n_skus"])
    promos = make_promotions(rng, skus, dates[0], dates[-1] + pd.Timedelta(days=120))
    carried = make_assortment(rng, skus, len(branches), tuple(dc["assortment_coverage"]))
    holiday = holiday_multiplier(dates)
    review_dow_b = rng.integers(0, 5, len(branches))
    branch_bias_b = rng.lognormal(0.05, 0.25, len(branches))
    outage = rng.random((len(branches), T)) < dq["outage_branch_day_rate"]

    lat, lon = branches["lat"].to_numpy(), branches["lon"].to_numpy()
    km = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    transit = 1 + np.floor(km / 550).astype(int)

    chunks = sku_chunks(carried, int(dc.get("chunk_series", 60_000)))
    n_total = int(carried.sum())
    write_truth = bool(dc.get("write_truth", True)) and n_total <= 20_000
    log.info(
        "%s series (%s branches x %s SKUs) x %s days in %s chunk(s)",
        f"{n_total:,}",
        len(branches),
        len(skus),
        T,
        len(chunks),
    )

    writer = RawWriter(paths.raw, dc.get("raw_format", "csv"))
    snap_days = np.unique(np.concatenate([np.nonzero(dates.dayofweek == 6)[0], [T - 1]]))
    branch_eod = np.zeros((len(branches), T))
    value_by_sku = np.zeros(len(skus))
    truth: dict[str, list[np.ndarray]] = {k: [] for k in ("demand", "sales", "outlier", "branch_id", "sku_id")}
    open_po_parts, totals = [], {"sales_rows": 0, "txn_rows": 0, "demand": 0, "sold": 0}
    txn_counter = 0

    for c, sku_rows in enumerate(chunks):
        sk = skus.iloc[sku_rows].reset_index(drop=True)
        series, demand = build_demand(
            rng, branches, sk, promos[promos["sku_id"].isin(sk["sku_id"])], dates, carried[sku_rows], holiday
        )
        sim = simulate_inventory(rng, series, sk, review_dow_b, branch_bias_b, demand, dates, transit)
        sales, eod = sim["sales"], sim["eod"]
        n = len(series)
        b_of = series["_b"].to_numpy()
        np.add.at(branch_eod, b_of, eod)
        np.add.at(
            value_by_sku,
            sku_rows[series["_s"].to_numpy()],
            sales[:, -365:].sum(axis=1) * sk["unit_cost"].to_numpy()[series["_s"].to_numpy()],
        )
        totals["demand"] += int(demand.sum())
        totals["sold"] += int(sales.sum())

        # ---- sales extract with data-quality issues -------------------------------------
        sb, ss = series["branch_id"].to_numpy(), series["sku_id"].to_numpy()
        cat_b = pd.Categorical(sb)  # categorical ids keep chunk frames small at network scale
        cat_s = pd.Categorical(ss)
        qty = sales.astype(np.int64).ravel()
        row_series = np.repeat(np.arange(n), T)
        row_day = np.tile(np.arange(T), n)
        keep = ~outage[b_of[row_series], row_day]
        keep &= rng.random(len(qty)) >= dq["random_row_drop_rate"]
        pos = np.nonzero(qty > 0)[0]
        out_rows = rng.choice(pos, min(len(pos), int(dq["outlier_rate"] * len(qty))), replace=False)
        qty[out_rows] *= rng.integers(10, 51, len(out_rows))
        zero = np.nonzero(qty == 0)[0]
        ret_rows = rng.choice(zero, min(len(zero), int(dq["return_rate"] * len(qty))), replace=False)
        qty[ret_rows] = -rng.integers(1, 4, len(ret_rows))
        sel = np.nonzero(keep)[0]
        dup = rng.choice(sel, int(dq["duplicate_rate"] * len(sel)), replace=False)
        sel = np.sort(np.concatenate([sel, dup]))
        writer.write(
            "sales_daily",
            pd.DataFrame(
                {
                    "date": dates.to_numpy()[row_day[sel]],
                    "branch_id": cat_b[row_series[sel]],
                    "sku_id": cat_s[row_series[sel]],
                    "qty": qty[sel],
                }
            ),
            c,
        )
        totals["sales_rows"] += len(sel)
        if write_truth:
            mask = np.zeros(n * T, dtype=bool)
            mask[out_rows] = True
            for k, v in (
                ("demand", demand),
                ("sales", sales),
                ("outlier", mask.reshape(n, T)),
                ("branch_id", sb),
                ("sku_id", ss),
            ):
                truth[k].append(v)
        del qty, row_series, row_day, keep

        # ---- inventory transactions ------------------------------------------------------
        def frame(
            idx: np.ndarray,
            day: np.ndarray,
            q: np.ndarray,
            kind: str,
            ref: Any = "",
            cat_b: pd.Categorical = cat_b,
            cat_s: pd.Categorical = cat_s,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "day": day,
                    "branch_id": cat_b[idx],
                    "sku_id": cat_s[idx],
                    "txn_type": pd.Categorical([kind] * len(idx)) if len(idx) else kind,
                    "qty": np.asarray(q).astype(np.int64),
                    "ref": ref,
                }
            )

        parts = []
        r_idx = np.concatenate([r[0] for r in sim["receipts"]]) if sim["receipts"] else np.array([], int)
        r_day = np.concatenate([r[1] for r in sim["receipts"]]) if sim["receipts"] else np.array([], int)
        r_qty = np.concatenate([r[2] for r in sim["receipts"]]) if sim["receipts"] else np.array([])
        in_hist = r_day < T  # receipts after the cut-off are open purchase orders
        parts.append(frame(r_idx[in_hist], r_day[in_hist], r_qty[in_hist], "receipt", "DC"))
        if sim["adjustments"]:
            parts.append(
                frame(
                    np.concatenate([a[0] for a in sim["adjustments"]]),
                    np.concatenate([np.full(len(a[0]), a[1]) for a in sim["adjustments"]]),
                    np.concatenate([a[2] for a in sim["adjustments"]]),
                    "adjustment",
                    "cycle_count",
                )
            )
        tr = np.array(sim["transfers"], dtype=int).reshape(-1, 5)
        if len(tr):
            refs = np.array([f"TR{c:03d}{i:05d}" for i in range(len(tr))])
            parts.append(frame(tr[:, 0], tr[:, 2], -tr[:, 4], "transfer_out", refs))
            ok = tr[:, 3] < T
            parts.append(frame(tr[ok, 1], tr[ok, 3], tr[ok, 4], "transfer_in", refs[ok]))
        n_bad = max(1, int(dq["bad_adjustment_rate"] * n * T))  # keying errors with no physical effect
        parts.append(
            frame(
                rng.integers(0, n, n_bad),
                rng.integers(0, T, n_bad),
                -rng.integers(50, 500, n_bad),
                "adjustment",
                "manual",
            )
        )
        sn_series = np.repeat(np.arange(n), len(snap_days))
        sn_day = np.tile(snap_days, n)
        lost = (rng.random(len(sn_day)) < dq["missing_snapshot_rate"]) & (sn_day != T - 1)
        sn_series, sn_day = sn_series[~lost], sn_day[~lost]
        parts.append(frame(sn_series, sn_day, eod[sn_series, sn_day], "snapshot", "count"))

        po = ~in_hist
        open_po_parts.append(
            pd.DataFrame(
                {
                    "branch_id": sb[r_idx[po]],
                    "sku_id": ss[r_idx[po]],
                    "due_date": (dates[0] + pd.to_timedelta(r_day[po], unit="D")).strftime("%Y-%m-%d"),
                    "qty": r_qty[po].astype(int),
                }
            )
        )

        txn = pd.concat(parts, ignore_index=True)
        for col in ("branch_id", "sku_id", "txn_type", "ref"):
            txn[col] = txn[col].astype("category")
        txn = txn.sort_values(["day", "branch_id", "sku_id"], kind="stable")
        txn.insert(0, "date", dates.to_numpy()[txn.pop("day").to_numpy()])
        txn.insert(0, "txn_id", np.arange(txn_counter, txn_counter + len(txn), dtype=np.int64))
        txn_counter += len(txn)
        writer.write("inventory_txn", txn, c)
        totals["txn_rows"] += len(txn)
        log.info("chunk %d/%d: %s series written (%.0fs)", c + 1, len(chunks), f"{n:,}", time.time() - t0)
        del sim, sales, eod, demand, txn, parts

    # ---- masters (need all chunks) -----------------------------------------------------------
    skus["abc_class"] = assign_abc(skus, value_by_sku)
    cap_rng = np.random.default_rng([dc["seed"], 7])
    branches["storage_capacity_units"] = (branch_eod.max(axis=1) * cap_rng.uniform(1.03, 1.30, len(branches))).astype(
        int
    )
    branches.drop(columns=["demand_multiplier"]).to_csv(paths.raw / "branches.csv", index=False)
    skus.drop(columns=["_archetype", "_rate"]).to_csv(paths.raw / "skus.csv", index=False)
    promos.to_csv(paths.raw / "promotions.csv", index=False, date_format="%Y-%m-%d")
    pd.concat(open_po_parts, ignore_index=True).to_csv(paths.raw / "open_purchase_orders.csv", index=False)
    bid = branches["branch_id"].to_numpy()
    pairs = pd.DataFrame(
        {
            "from_branch": np.repeat(bid, len(bid)),
            "to_branch": np.tile(bid, len(bid)),
            "distance_km": km.ravel().round(1),
            "cost_per_unit": (0.35 + 0.0012 * km.ravel()).round(3),
            "transit_days": transit.ravel(),
        }
    )
    pairs[pairs["from_branch"] != pairs["to_branch"]].to_csv(paths.raw / "transfer_costs.csv", index=False)

    truth_path = paths.raw / "_truth.npz"
    truth_path.unlink(missing_ok=True)
    if write_truth:  # ground truth for validating the cleaning pipeline (never read by the pipeline)
        np.savez_compressed(
            truth_path,
            demand=np.concatenate(truth["demand"]),
            sales=np.concatenate(truth["sales"]),
            outlier=np.concatenate(truth["outlier"]),
            branch_id=np.concatenate(truth["branch_id"]).astype("U8"),
            sku_id=np.concatenate(truth["sku_id"]).astype("U8"),
        )
    log.info(
        "done: sales rows=%s txn rows=%s fill rate=%.3f lost units=%s (%.0fs)",
        f"{totals['sales_rows']:,}",
        f"{totals['txn_rows']:,}",
        totals["sold"] / max(totals["demand"], 1),
        f"{totals['demand'] - totals['sold']:,}",
        time.time() - t0,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--config", default=None)
    main(ap.parse_args().config)
