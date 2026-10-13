"""Inventory rebalancing: min-cost transfer optimisation (MILP) and a greedy baseline.

Model (per SKU s, sender i, receiver j):

    min  sum c_ij x_ij                         transfer cost (+ cross-region penalty)
       + sum_i e_i * excess_after_i            holding + obsolescence of remaining excess
       + sum_j p1_j (tier1_j - cov1_j)         expected lost sales within the lead time
       + sum_j p2_j (tier2_j - cov2_j)         unprotected safety-stock units

    s.t. sum_j x_ij <= sendable_i              sender never drops below its floor (>= safety stock)
         sum_i x_ij <= room_j                  receiver never pushed above target_max
         cov1_j + cov2_j <= sum_i x_ij,  cov1_j <= tier1_j,  cov2_j <= tier2_j
         excess_after_i >= excess_i - sum_j x_ij,  excess_after_i >= 0
         sum_{s,i} x_ij <= free_capacity_j     branch storage capacity (units)
         m_s y_ij <= x_ij <= U_ij y_ij         minimum transfer quantity (y binary)
         x_ij integer, only lanes with transit_days <= max (and same region if required)

Senders (excess or stagnant stock) and receivers (below target with live demand) are
disjoint per SKU, so circular A->B->A moves cannot occur.

Scale: each receiver keeps only its ``max_senders_per_receiver`` cheapest senders and each
sender its ``max_receivers_per_sender`` cheapest receivers. SKUs are then solved in batches
of about ``batch_arcs`` arcs, most valuable first; branch capacity used by earlier batches
is deducted before the next. A batch that hits the time limit keeps its best feasible
solution; a batch with no solution (or a solver error) falls back to the greedy heuristic.

Nothing here executes a transfer: the output is a list of recommendations for planners.
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pulp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import get_paths, load_config  # noqa: E402
from src.policy import expected_lost_units, stockout_probability  # noqa: E402
from src.runlog import get_logger, run_id  # noqa: E402
from src.validation import TRANSFERS, validate  # noqa: E402

log = get_logger("optimizer")

REASON_EXCESS_SHORTFALL = "excess->shortfall"
REASON_STAGNANT_DEMAND = "stagnant->demand"
REASON_EXCESS_DEMAND = "excess->demand"


@dataclass
class Problem:
    """Pre-processed optimisation inputs."""

    nodes: pd.DataFrame  # one row per branch x SKU with sender / receiver attributes
    pairs: pd.DataFrame  # eligible (sku, sender, receiver) arcs
    free_capacity: pd.Series  # branch_id -> spare storage units


def _nodes(policy: pd.DataFrame, oc: dict[str, Any]) -> pd.DataFrame:
    """Sender / receiver quantities and unit economics per branch x SKU."""
    d = policy.copy()
    floor = d["reorder_point"] if oc["sender_floor"] == "rop" else d["safety_stock"]
    d["floor"] = np.maximum(floor, d["safety_stock"])
    is_sender = ((d["excess_units"] >= 1) | d["is_stagnant"]) & (d["on_hand"] > 0)
    d["sendable"] = np.where(is_sender, np.floor(np.minimum(d["on_hand"], d["inv_position"] - d["floor"])), 0)
    d["sendable"] = d["sendable"].clip(lower=0)
    is_receiver = (
        (~is_sender)
        & (~d["is_stagnant"])
        & (d["forecast_mean_daily"] > 0.01)
        & (d["inv_position"] < d["target_max"] - 0.5)
    )
    d["room"] = np.where(is_receiver, np.floor(d["target_max"] - d["inv_position"]), 0).clip(min=0)
    d["tier1"] = np.where(is_receiver, np.maximum(d["fc_lead"] - d["inv_position"], 0), 0.0)
    d["tier2"] = np.where(is_receiver, np.maximum(d["reorder_point"] - d["inv_position"], 0) - d["tier1"], 0.0)
    d["tier1"] = np.minimum(d["tier1"], d["room"])
    d["tier2"] = np.clip(np.minimum(d["tier2"], d["room"] - d["tier1"]), 0, None)

    d["p1"] = oc["stockout_cost_multiplier"] * d["unit_cost"]
    d["p2"] = oc["safety_stock_tier_weight"] * d["p1"]
    sit_days = np.where(
        d["forecast_mean_daily"] > 0.01,
        d["excess_units"] / np.maximum(d["forecast_mean_daily"], 1e-9),
        oc["excess_cost_max_days"],
    )
    sit_days = np.clip(sit_days, oc["excess_cost_min_days"], oc["excess_cost_max_days"])
    obs_mult = np.where(d["is_stagnant"], oc["obsolescence_multiplier_stagnant"], 1.0)
    d["e_cost"] = d["unit_cost"] * (d["holding_cost_rate"] + d["obsolescence_risk"] * obs_mult) * sit_days / 365.0
    d["min_qty"] = np.maximum(oc["min_transfer_units"], np.ceil(oc["min_transfer_value"] / d["unit_cost"]))
    d["node"] = np.arange(len(d))
    return d


def _prune(pairs: pd.DataFrame, k_send: int, k_recv: int) -> pd.DataFrame:
    """Keep an arc if it is among the receiver's k cheapest senders or the sender's k cheapest receivers."""
    r_rank = pairs.groupby("node_r")["unit_cost_move"].rank(method="first")
    s_rank = pairs.groupby("node_s")["unit_cost_move"].rank(method="first")
    return pairs[(r_rank <= k_send) | (s_rank <= k_recv)]


def build_problem(policy: pd.DataFrame, branches: pd.DataFrame, costs: pd.DataFrame, cfg: dict[str, Any]) -> Problem:
    """Derive sender/receiver quantities, unit economics and eligible, pruned arcs."""
    oc = cfg["optimizer"]
    d = _nodes(policy, oc)
    reg = branches.set_index("branch_id")["region"]
    lanes = costs.rename(columns={"from_branch": "branch_id_s", "to_branch": "branch_id_r"})
    lanes = lanes[lanes["transit_days"] <= oc["max_transit_days"]].copy()
    lanes["cross_region"] = lanes["branch_id_s"].map(reg) != lanes["branch_id_r"].map(reg)
    if oc["same_region_only"]:
        lanes = lanes[~lanes["cross_region"]]
    lanes["unit_cost_move"] = lanes["cost_per_unit"] + np.where(
        lanes["cross_region"] & (not oc["same_region_only"]), oc["cross_region_penalty"], 0.0
    )
    lanes = lanes[["branch_id_s", "branch_id_r", "cost_per_unit", "transit_days", "cross_region", "unit_cost_move"]]

    snd = d[d["sendable"] > 0][["node", "sku_id", "branch_id", "sendable", "min_qty", "e_cost"]]
    rcv = d[d["room"] > 0][["node", "sku_id", "branch_id", "room", "p1"]]
    k_send = int(oc.get("max_senders_per_receiver", 8))
    k_recv = int(oc.get("max_receivers_per_sender", 8))
    skus = np.intersect1d(snd["sku_id"].unique(), rcv["sku_id"].unique())
    parts = []
    for chunk in np.array_split(skus, max(1, len(skus) // 200)):  # bounded memory at network scale
        p = snd[snd["sku_id"].isin(chunk)].merge(rcv[rcv["sku_id"].isin(chunk)], on="sku_id", suffixes=("_s", "_r"))
        p = p.merge(lanes, on=["branch_id_s", "branch_id_r"], how="inner")
        p["upper"] = np.minimum(p["sendable"], p["room"])
        # an arc is only worth having if a unit can create more value than it costs to move
        p = p[(p["upper"] >= p["min_qty"]) & (p["p1"] + p["e_cost"] > p["unit_cost_move"])]
        parts.append(_prune(p, k_send, k_recv))
    cols = [
        "node_s",
        "sku_id",
        "branch_id_s",
        "sendable",
        "min_qty",
        "node_r",
        "branch_id_r",
        "room",
        "cost_per_unit",
        "transit_days",
        "cross_region",
        "unit_cost_move",
        "upper",
    ]
    pairs = pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)

    on_hand_total = d.groupby("branch_id")["on_hand"].sum()
    cap = branches.set_index("branch_id")["storage_capacity_units"]
    free = (cap - on_hand_total.reindex(cap.index).fillna(0)).clip(lower=0)
    return Problem(d, pairs.reset_index(drop=True), free)


def objective_terms(nodes: pd.DataFrame, plan: pd.DataFrame) -> dict[str, float]:
    """Evaluate the optimisation objective (and its parts) for any transfer plan.

    ``plan`` has columns node_s, node_r, qty, unit_cost_move.
    """
    out_q = plan.groupby("node_s")["qty"].sum().reindex(nodes["node"], fill_value=0).to_numpy()
    in_q = plan.groupby("node_r")["qty"].sum().reindex(nodes["node"], fill_value=0).to_numpy()
    cov1 = np.minimum(in_q, nodes["tier1"].to_numpy())
    cov2 = np.minimum(in_q - cov1, nodes["tier2"].to_numpy())
    excess_after = np.maximum(nodes["excess_units"].to_numpy() - out_q, 0)
    parts = {
        "transfer_cost": float((plan["qty"] * plan["unit_cost_move"]).sum()) if len(plan) else 0.0,
        "excess_cost": float((nodes["e_cost"] * excess_after).sum()),
        "stockout_cost": float(
            (nodes["p1"] * (nodes["tier1"] - cov1)).sum() + (nodes["p2"] * (nodes["tier2"] - cov2)).sum()
        ),
    }
    parts["total"] = sum(parts.values())
    return parts


def solve_milp(
    prob: Problem,
    cfg: dict[str, Any],
    pairs: pd.DataFrame | None = None,
    capacity: Mapping[Hashable, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Solve the MILP for ``pairs`` (default: all arcs) with HiGHS; returns the plan and solver info."""
    oc = cfg["optimizer"]
    d = prob.nodes.set_index("node")
    P = prob.pairs if pairs is None else pairs
    cap = prob.free_capacity.to_dict() if capacity is None else capacity
    if P.empty:
        return P.assign(qty=[]), {"status": "Empty", "seconds": 0.0, "arcs": 0}
    m = pulp.LpProblem("rebalance", pulp.LpMinimize)
    ids = P.index.to_numpy()
    upper = P["upper"].to_numpy(dtype=float)
    minq = P["min_qty"].to_numpy(dtype=float)
    ucost = P["unit_cost_move"].to_numpy(dtype=float)
    x = {
        k: m.add_variable(f"x_{k}", lowBound=0, upBound=float(u), cat="Integer")
        for k, u in zip(ids, upper, strict=True)
    }
    y = {k: m.add_variable(f"y_{k}", cat="Binary") for k in ids}
    senders = P["node_s"].unique()
    receivers = P["node_r"].unique()
    ex = {i: m.add_variable(f"ex_{i}", lowBound=0) for i in senders}
    col = {
        c: d[c].astype(float).to_dict()
        for c in ("tier1", "tier2", "e_cost", "p1", "p2", "sendable", "excess_units", "room")
    }
    c1 = {j: m.add_variable(f"c1_{j}", lowBound=0, upBound=float(col["tier1"][j])) for j in receivers}
    c2 = {j: m.add_variable(f"c2_{j}", lowBound=0, upBound=float(col["tier2"][j])) for j in receivers}

    m += (
        pulp.lpSum(float(c) * x[k] for k, c in zip(ids, ucost, strict=True))
        + pulp.lpSum(float(col["e_cost"][i]) * ex[i] for i in senders)
        - pulp.lpSum(float(col["p1"][j]) * c1[j] + float(col["p2"][j]) * c2[j] for j in receivers)
    )
    for i, ks in P.groupby("node_s").groups.items():
        m += pulp.lpSum(x[k] for k in ks) <= float(col["sendable"][i])
        m += ex[i] >= float(col["excess_units"][i]) - pulp.lpSum(x[k] for k in ks)
    for j, ks in P.groupby("node_r").groups.items():
        m += pulp.lpSum(x[k] for k in ks) <= float(col["room"][j])
        m += c1[j] + c2[j] <= pulp.lpSum(x[k] for k in ks)
    for b, ks in P.groupby("branch_id_r").groups.items():
        m += pulp.lpSum(x[k] for k in ks) <= float(cap.get(b, 0))
    for k, u, q in zip(ids, upper, minq, strict=True):
        m += x[k] >= float(q) * y[k]
        m += x[k] <= float(u) * y[k]

    solver = pulp.HiGHS(msg=False, timeLimit=oc["solver_time_limit_s"], gapRel=oc["solver_gap"])
    t0 = time.time()
    stats = m.solve(solver)
    info = {
        "status": stats.status.name,
        "has_solution": bool(getattr(stats, "has_solution", True)),
        "seconds": round(time.time() - t0, 1),
        "arcs": len(ids),
        "variables": len(m.variables()),
        "constraints": len(m.constraints()),
    }
    qty = np.array([round(x[k].value() or 0) for k in ids]) if info["has_solution"] else np.zeros(len(ids))
    plan = P.assign(qty=qty)
    return plan[plan["qty"] > 0], info


def solve_greedy(
    prob: Problem, pairs: pd.DataFrame | None = None, capacity: Mapping[Hashable, float] | None = None
) -> pd.DataFrame:
    """Baseline: serve the most urgent receivers first from the cheapest eligible sender.

    Pass 1 covers shortfalls (highest lost-sales value first); pass 2 moves remaining excess
    into receivers' spare room when holding/obsolescence saved exceeds the move cost.
    """
    d = prob.nodes.set_index("node")
    P = (prob.pairs if pairs is None else pairs).copy()
    if P.empty:
        return P.assign(qty=[])
    left_send = d["sendable"].to_dict()
    left_room = d["room"].to_dict()
    left_cap = dict(prob.free_capacity.to_dict() if capacity is None else capacity)
    need = (d["tier1"] + d["tier2"]).to_dict()
    tier1, p1, p2 = d["tier1"].to_numpy(), d["p1"].to_numpy(), d["p2"].to_numpy()
    tier2, ecost = d["tier2"].to_numpy(), d["e_cost"].to_numpy()
    r, s_ = P["node_r"].to_numpy(), P["node_s"].to_numpy()
    P["urgency"] = tier1[r] * p1[r] + tier2[r] * p2[r]
    P["gain2"] = ecost[s_] - P["unit_cost_move"]
    qty = np.zeros(len(P))
    pos = {k: i for i, k in enumerate(P.index)}
    for pass_no in (1, 2):
        order = (
            P.sort_values(["urgency", "unit_cost_move"], ascending=[False, True])
            if pass_no == 1
            else P[P["gain2"] > 0].sort_values("gain2", ascending=False)
        )
        for k, i, j, b, mq, uc in zip(
            order.index,
            order["node_s"].to_numpy(),
            order["node_r"].to_numpy(),
            order["branch_id_r"].to_numpy(),
            order["min_qty"].to_numpy(),
            order["unit_cost_move"].to_numpy(),
            strict=True,
        ):
            if pass_no == 1 and (need[j] <= 0 or uc >= p1[j]):
                continue
            want = np.ceil(need[j]) if pass_no == 1 else left_room[j]
            q = np.floor(min(left_send[i], left_room[j], want, left_cap.get(b, 0)))
            if q < mq or q <= 0:
                continue
            qty[pos[k]] += q
            left_send[i] -= q
            left_room[j] -= q
            left_cap[b] = left_cap.get(b, 0) - q
            need[j] = max(need[j] - q, 0)
    plan = P.drop(columns=["urgency", "gain2"]).assign(qty=qty)
    return plan[plan["qty"] > 0]


_SHARED: dict[str, Any] = {}  # per-process copy of the problem (set directly, or by the pool initializer)


def _init_worker(prob: Problem, cfg: dict[str, Any]) -> None:
    _SHARED.update(prob=prob, cfg=cfg)


def _solve_batch(job: tuple[int, list[str], dict[Hashable, float]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Solve one SKU batch with its capacity allocation; greedy fallback on solver failure."""
    bi, skus, cap = job
    prob, cfg = _SHARED["prob"], _SHARED["cfg"]
    sub = prob.pairs[prob.pairs["sku_id"].isin(skus)]
    try:
        plan, info = solve_milp(prob, cfg, sub, cap)
        method = "milp"
        if not info.get("has_solution", True) or info["status"] in (
            "Infeasible",
            "Unbounded",
            "NotSolved",
            "Undefined",
            "NumericalError",
        ):
            raise RuntimeError(info["status"])
    except Exception as e:  # noqa: BLE001 - any solver failure falls back to greedy
        plan = solve_greedy(prob, sub, cap)
        info, method = {"status": f"fallback: {e}"[:80], "arcs": len(sub)}, "greedy_fallback"
    return plan, {"batch": bi + 1, "skus": len(skus), "method": method, **info}


def allocate_capacity(prob: Problem, batches: list[list[str]]) -> list[dict[Hashable, float]]:
    """Split each branch's free capacity across batches in proportion to the room they could fill.

    Where a branch can absorb everything every batch could send it, each batch gets its full need,
    so the split never binds; otherwise capacity is shared pro rata. This makes batches independent.
    """
    d = prob.nodes.set_index("node")
    rec = prob.pairs.drop_duplicates("node_r")[["node_r", "sku_id", "branch_id_r"]]
    rec = rec.assign(room=d.loc[rec["node_r"], "room"].to_numpy())
    sku_batch = {s: i for i, skus in enumerate(batches) for s in skus}
    rec["batch"] = rec["sku_id"].map(sku_batch)
    need = rec.groupby(["batch", "branch_id_r"])["room"].sum().unstack(fill_value=0)
    free = prob.free_capacity.reindex(need.columns).fillna(0.0).to_numpy(dtype=float)
    n = need.to_numpy(dtype=float)
    total = n.sum(axis=0)
    binding = total > free  # branches that cannot absorb every batch's full need
    alloc = n.copy()
    alloc[:, binding] = n[:, binding] / total[binding] * free[binding]
    out = pd.DataFrame(alloc, index=need.index, columns=need.columns)
    return [out.loc[i].to_dict() if i in out.index else {} for i in range(len(batches))]


def solve_decomposed(prob: Problem, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Solve SKU batches in parallel, each with its own share of branch capacity."""
    oc = cfg["optimizer"]
    P = prob.pairs
    if P.empty:
        return P.assign(qty=[]), pd.DataFrame()
    d = prob.nodes.set_index("node")
    val = d["tier1"] * d["p1"] + d["tier2"] * d["p2"]
    sku_value = (
        P.drop_duplicates("node_r").assign(v=lambda x: val.loc[x["node_r"]].to_numpy()).groupby("sku_id")["v"].sum()
    )
    arcs = P.groupby("sku_id").size().reindex(sku_value.index)
    order = sku_value.sort_values(ascending=False).index
    batch_arcs = int(oc.get("batch_arcs", 20_000))
    batches: list[list[str]] = []
    cur: list[str] = []
    n = 0
    for sku in order:
        cur.append(str(sku))
        n += int(arcs[sku])
        if n >= batch_arcs:
            batches.append(cur)
            cur, n = [], 0
    if cur:
        batches.append(cur)

    caps = allocate_capacity(prob, batches)
    jobs = [(i, skus, caps[i]) for i, skus in enumerate(batches)]
    workers = int(oc.get("workers", 0)) or (os.cpu_count() or 1)
    workers = max(1, min(workers, len(jobs)))
    if workers == 1:
        _init_worker(prob, cfg)
        try:
            results = [_solve_batch(j) for j in jobs]
        finally:
            _SHARED.clear()
    else:
        # "spawn", not "fork": forking a process whose solver/OpenMP thread pools are already running
        # can deadlock the children. Each worker receives the problem once via the initializer.
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(prob, cfg)) as pool:
            results = pool.map(_solve_batch, jobs, chunksize=1)
    plans = [r[0] for r in results]
    info_df = pd.DataFrame([r[1] for r in results])
    log.info(
        "solved %d batch(es) on %d worker(s): %s",
        len(batches),
        workers,
        ", ".join(f"{k}={v}" for k, v in info_df["status"].value_counts().items()),
    )
    return pd.concat(plans, ignore_index=True), info_df


REASON_TEXT = {
    REASON_EXCESS_SHORTFALL: (
        "{to} is short of {sku}: on-hand covers {to_dos} of demand and stock-out risk is "
        "{risk} before replenishment arrives. {frm} holds {frm_dos} of supply, above its "
        "target; moving {qty} units costs {cost} and avoids an estimated {sav} in lost sales "
        "and carrying cost."
    ),
    REASON_STAGNANT_DEMAND: (
        "{sku} has not sold at {frm} for {days} days. {to} still sells it ({to_dos} of supply "
        "on hand), so moving {qty} units puts dead stock back to work; net benefit {sav}."
    ),
    REASON_EXCESS_DEMAND: (
        "{frm} holds {frm_dos} of {sku}, above target. {to} has room under its target and live "
        "demand; moving {qty} units cuts holding and obsolescence cost by a net {sav}."
    ),
}


def _days(x: float) -> str:
    return "over a year" if x >= 365 else f"{x:.0f} days"


def describe_plan(prob: Problem, plan: pd.DataFrame) -> pd.DataFrame:
    """Attach id, cost, projected savings, reason code and a plain-language reason to each transfer."""
    d = prob.nodes.set_index("node")
    cols = [
        "transfer_id",
        "run_id",
        "from_branch",
        "to_branch",
        "sku_id",
        "category",
        "qty",
        "unit_cost",
        "value_moved",
        "transit_days",
        "transfer_cost",
        "stockout_cost_avoided",
        "excess_cost_avoided",
        "projected_savings",
        "reason_code",
        "reason",
        "from_days_of_supply",
        "to_days_of_supply",
        "to_p_stockout_before",
        "cross_region",
    ]
    if plan.empty:
        return pd.DataFrame(columns=cols)
    p = plan.copy()
    in_q = p.groupby("node_r")["qty"].transform("sum")
    out_q = p.groupby("node_s")["qty"].transform("sum")
    t1 = d.loc[p["node_r"], "tier1"].to_numpy()
    t2 = d.loc[p["node_r"], "tier2"].to_numpy()
    cov1 = np.minimum(in_q, t1)
    cov2 = np.minimum(in_q - cov1, t2)
    share_r = p["qty"] / in_q
    exc = d.loc[p["node_s"], "excess_units"].to_numpy()
    freed = np.minimum(out_q, exc) * p["qty"] / out_q
    p["transfer_cost"] = p["qty"] * p["unit_cost_move"]
    p["stockout_cost_avoided"] = share_r * (
        cov1 * d.loc[p["node_r"], "p1"].to_numpy() + cov2 * d.loc[p["node_r"], "p2"].to_numpy()
    )
    p["excess_cost_avoided"] = freed * d.loc[p["node_s"], "e_cost"].to_numpy()
    p["projected_savings"] = p["stockout_cost_avoided"] + p["excess_cost_avoided"] - p["transfer_cost"]
    stagnant = d.loc[p["node_s"], "is_stagnant"].to_numpy()
    p["reason_code"] = np.where(
        stagnant,
        REASON_STAGNANT_DEMAND,
        np.where(share_r * (cov1 + cov2) > 0, REASON_EXCESS_SHORTFALL, REASON_EXCESS_DEMAND),
    )
    p["unit_cost"] = d.loc[p["node_s"], "unit_cost"].to_numpy()
    p["value_moved"] = p["qty"] * p["unit_cost"]
    p["category"] = d.loc[p["node_s"], "category"].to_numpy()
    p["from_days_of_supply"] = d.loc[p["node_s"], "days_of_supply"].to_numpy()
    p["to_days_of_supply"] = d.loc[p["node_r"], "days_of_supply"].to_numpy()
    p["to_p_stockout_before"] = d.loc[p["node_r"], "p_stockout"].to_numpy()
    since = d.loc[p["node_s"], "days_since_last_sale"].to_numpy() if "days_since_last_sale" in d else np.zeros(len(p))
    out = p.rename(columns={"branch_id_s": "from_branch", "branch_id_r": "to_branch"})
    rid = run_id()
    out["run_id"] = rid
    out["transfer_id"] = [
        hashlib.sha1(f"{rid}|{s}|{a}|{b}".encode(), usedforsecurity=False).hexdigest()[:12]
        for s, a, b in zip(out["sku_id"], out["from_branch"], out["to_branch"], strict=True)
    ]
    out["reason"] = [
        REASON_TEXT[r].format(
            frm=f,
            to=t,
            sku=s,
            qty=f"{q:,.0f}",
            cost=f"${c:,.0f}",
            sav=f"${v:,.0f}",
            frm_dos=_days(fd),
            to_dos=_days(td),
            risk=f"{pr:.0%}",
            days=int(sd),
        )
        for r, f, t, s, q, c, v, fd, td, pr, sd in zip(
            out["reason_code"],
            out["from_branch"],
            out["to_branch"],
            out["sku_id"],
            out["qty"],
            out["transfer_cost"],
            out["projected_savings"],
            out["from_days_of_supply"],
            out["to_days_of_supply"],
            out["to_p_stockout_before"],
            since,
            strict=True,
        )
    ]
    return out[cols].sort_values("projected_savings", ascending=False).reset_index(drop=True)


def apply_plan(nodes: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """Return node-level state after executing ``plan`` (on hand and position adjusted)."""
    d = nodes.copy()
    out_q = plan.groupby("node_s")["qty"].sum().reindex(d["node"], fill_value=0).to_numpy()
    in_q = plan.groupby("node_r")["qty"].sum().reindex(d["node"], fill_value=0).to_numpy()
    d["transfer_out"], d["transfer_in"] = out_q, in_q
    d["on_hand_after"] = d["on_hand"] - out_q + in_q
    d["inv_position_after"] = d["inv_position"] - out_q + in_q
    return d


def impact(nodes: pd.DataFrame, plan: pd.DataFrame, risk_threshold: float = 0.2) -> dict[str, float]:
    """Business impact of a plan versus doing nothing.

    Inventory turns use projected inventory after one replenishment cycle: positions below
    target are replenished up to target either way, so units moved out of excess are units
    the DC does not have to buy.
    """
    a = apply_plan(nodes, plan)
    cost = a["unit_cost"]
    exc_before = a["excess_units"]
    exc_after = np.maximum(a["inv_position_after"] - a["target_max"], 0)
    lost_before = expected_lost_units(a["inv_position"], a["fc_lead"], a["sigma_lead"])
    lost_after = expected_lost_units(a["inv_position_after"], a["fc_lead"], a["sigma_lead"])
    stag = a["is_stagnant"]
    risk_after = stockout_probability(a["inv_position_after"], a["fc_lead"], a["sigma_lead"])
    cogs = (a["forecast_mean_daily"] * 365 * cost).sum()
    inv_before = (np.maximum(a["inv_position"], a["target_max"]) * cost).sum()
    inv_after = (np.maximum(a["inv_position_after"], a["target_max"]) * cost).sum()
    t = objective_terms(nodes, plan)
    base = objective_terms(nodes, plan.iloc[0:0])
    unit_cost = nodes.set_index("node")["unit_cost"]
    return {
        "transfers": len(plan),
        "units_moved": float(plan["qty"].sum()),
        "value_moved": float((plan["qty"] * unit_cost.loc[plan["node_s"]].to_numpy()).sum()) if len(plan) else 0.0,
        "transfer_cost": t["transfer_cost"],
        "excess_value_before": float((exc_before * cost).sum()),
        "excess_value_after": float((exc_after * cost).sum()),
        "excess_value_freed": float(((exc_before - exc_after) * cost).sum()),
        "stagnant_value_before": float((a["on_hand"] * cost)[stag].sum()),
        "stagnant_value_after": float((a["on_hand_after"] * cost)[stag].sum()),
        "annual_carrying_cost_excess_before": float((exc_before * cost * a["holding_cost_rate"]).sum()),
        "annual_carrying_cost_excess_after": float((exc_after * cost * a["holding_cost_rate"]).sum()),
        "expected_stockout_units_before": float(lost_before.sum()),
        "expected_stockout_units_after": float(lost_after.sum()),
        "stockout_units_avoided": float(lost_before.sum() - lost_after.sum()),
        "series_at_risk_before": int((a["p_stockout"] > risk_threshold).sum()),
        "series_at_risk_after": int((risk_after > risk_threshold).sum()),
        "inventory_turns_before": float(cogs / inv_before) if inv_before else np.nan,
        "inventory_turns_after": float(cogs / inv_after) if inv_after else np.nan,
        "objective_before": base["total"],
        "objective_after": t["total"],
        "net_savings": base["total"] - t["total"],
    }


def check_plan(prob: Problem, plan: pd.DataFrame) -> dict[str, int]:
    """Sanity checks that must all be zero for a plan to be published."""
    a = apply_plan(prob.nodes, plan)
    senders = a[a["transfer_out"] > 0]
    receivers = a[a["transfer_in"] > 0]
    arcs = set(zip(plan["sku_id"], plan["branch_id_s"], plan["branch_id_r"], strict=True))
    inbound = plan.groupby("branch_id_r")["qty"].sum()
    return {
        "unit_conservation_errors": int(abs(a["transfer_out"].sum() - a["transfer_in"].sum()) > 1e-6),
        "senders_below_safety_stock": int((senders["inv_position_after"] < senders["safety_stock"] - 1e-6).sum()),
        "senders_below_floor": int((senders["inv_position_after"] < senders["floor"] - 1e-6).sum()),
        "receivers_above_target": int((receivers["inv_position_after"] > receivers["target_max"] + 1e-6).sum()),
        "circular_transfers": sum((s, b, a_) in arcs for s, a_, b in arcs),
        "capacity_violations": int((inbound > prob.free_capacity.reindex(inbound.index).fillna(0) + 1e-6).sum()),
        "below_minimum_qty": int((plan["qty"] < plan["min_qty"]).sum()),
    }


def run(config_path: str | None = None) -> None:
    """Optimise transfers, compare with greedy, and write recommendations + impact."""
    t0 = time.time()
    cfg = load_config(config_path)
    paths = get_paths(cfg)
    policy = pd.read_parquet(paths.output / "inventory_policy.parquet")
    branches = pd.read_csv(paths.raw / "branches.csv")
    costs = pd.read_csv(paths.raw / "transfer_costs.csv")
    prob = build_problem(policy, branches, costs, cfg)
    log.info(
        "senders=%s receivers=%s arcs=%s (after pruning)",
        f"{int((prob.nodes['sendable'] > 0).sum()):,}",
        f"{int((prob.nodes['room'] > 0).sum()):,}",
        f"{len(prob.pairs):,}",
    )

    t1 = time.time()
    plan, batch_info = solve_decomposed(prob, cfg)
    opt_sec = time.time() - t1
    t_g = time.time()
    greedy = solve_greedy(prob)
    g_sec = time.time() - t_g
    checks = check_plan(prob, plan)
    if any(checks.values()):
        raise RuntimeError(f"transfer plan failed sanity checks: {checks}")

    recs = describe_plan(prob, plan)
    validate(recs, TRANSFERS)
    recs.to_csv(paths.output / "transfer_recommendations.csv", index=False)
    recs.to_parquet(paths.output / "transfer_recommendations.parquet", index=False)
    describe_plan(prob, greedy).to_csv(paths.output / "transfer_recommendations_greedy.csv", index=False)
    batch_info.to_csv(paths.reports / "optimizer_batches.csv", index=False)
    thr = cfg["policy"]["stockout_risk_threshold"]
    imp = pd.DataFrame({"optimizer": impact(prob.nodes, plan, thr), "greedy": impact(prob.nodes, greedy, thr)})
    imp.loc["solve_seconds"] = [round(opt_sec, 1), round(g_sec, 1)]
    imp.loc["rows_with_non_positive_savings"] = [int((recs["projected_savings"] <= 0).sum()), np.nan]
    imp.index.name = "metric"
    imp.to_csv(paths.reports / "optimizer_impact.csv")
    pd.Series(checks, name="violations").rename_axis("check").to_csv(paths.reports / "optimizer_checks.csv")
    apply_plan(prob.nodes, plan).to_parquet(paths.output / "inventory_after_transfers.parquet", index=False)
    o, g = imp["optimizer"], imp["greedy"]
    log.info(
        "optimizer: %d transfers, %s units, net savings $%s (greedy $%s); stock-out units avoided %s; "
        "all sanity checks passed (%.0fs)",
        int(o["transfers"]),
        f"{o['units_moved']:,.0f}",
        f"{o['net_savings']:,.0f}",
        f"{g['net_savings']:,.0f}",
        f"{o['stockout_units_avoided']:,.0f}",
        time.time() - t0,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Optimise inventory transfers.")
    ap.add_argument("--config", default=None)
    run(ap.parse_args().config)
