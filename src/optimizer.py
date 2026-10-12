"""Inventory rebalancing: min-cost transfer optimisation (MILP) and a greedy baseline.

Model (per SKU s, sender i, receiver j; all SKUs solved jointly because receiving
capacity is shared across SKUs):

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
         x_ij integer, only pairs with transit_days <= max (and same region if required)

Senders (excess or stagnant stock) and receivers (below target with live demand) are
disjoint per SKU, so circular A->B->A moves cannot occur.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pulp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import get_paths, load_config  # noqa: E402
from src.policy import expected_lost_units, stockout_probability  # noqa: E402

REASON_EXCESS_SHORTFALL = "excess->shortfall"
REASON_STAGNANT_DEMAND = "stagnant->demand"
REASON_EXCESS_DEMAND = "excess->demand"


@dataclass
class Problem:
    """Pre-processed optimisation inputs."""

    nodes: pd.DataFrame   # one row per branch x SKU with sender / receiver attributes
    pairs: pd.DataFrame   # eligible (sku, sender, receiver) arcs
    free_capacity: pd.Series  # branch_id -> spare storage units


def build_problem(policy: pd.DataFrame, branches: pd.DataFrame, costs: pd.DataFrame,
                  cfg: dict[str, Any]) -> Problem:
    """Derive sender/receiver quantities, unit economics and eligible arcs."""
    oc = cfg["optimizer"]
    d = policy.copy()
    floor = d["reorder_point"] if oc["sender_floor"] == "rop" else d["safety_stock"]
    d["floor"] = np.maximum(floor, d["safety_stock"])
    is_sender = ((d["excess_units"] >= 1) | d["is_stagnant"]) & (d["on_hand"] > 0)
    d["sendable"] = np.where(is_sender, np.floor(np.minimum(d["on_hand"], d["inv_position"] - d["floor"])), 0)
    d["sendable"] = d["sendable"].clip(lower=0)
    is_receiver = (~is_sender) & (~d["is_stagnant"]) & (d["forecast_mean_daily"] > 0.01) \
        & (d["inv_position"] < d["target_max"] - 0.5)
    d["room"] = np.where(is_receiver, np.floor(d["target_max"] - d["inv_position"]), 0).clip(min=0)
    d["tier1"] = np.where(is_receiver, np.maximum(d["fc_lead"] - d["inv_position"], 0), 0.0)
    d["tier2"] = np.where(is_receiver, np.maximum(d["reorder_point"] - d["inv_position"], 0) - d["tier1"], 0.0)
    d["tier1"] = np.minimum(d["tier1"], d["room"])
    d["tier2"] = np.clip(np.minimum(d["tier2"], d["room"] - d["tier1"]), 0, None)

    # unit economics
    d["p1"] = oc["stockout_cost_multiplier"] * d["unit_cost"]
    d["p2"] = oc["safety_stock_tier_weight"] * d["p1"]
    sit_days = np.where(d["forecast_mean_daily"] > 0.01, d["excess_units"] / np.maximum(d["forecast_mean_daily"], 1e-9),
                        oc["excess_cost_max_days"])
    sit_days = np.clip(sit_days, oc["excess_cost_min_days"], oc["excess_cost_max_days"])
    obs_mult = np.where(d["is_stagnant"], oc["obsolescence_multiplier_stagnant"], 1.0)
    d["e_cost"] = d["unit_cost"] * (d["holding_cost_rate"] + d["obsolescence_risk"] * obs_mult) * sit_days / 365.0
    d["min_qty"] = np.maximum(oc["min_transfer_units"], np.ceil(oc["min_transfer_value"] / d["unit_cost"]))
    d["node"] = np.arange(len(d))

    snd = d[d["sendable"] > 0][["node", "sku_id", "branch_id", "sendable", "min_qty"]]
    rcv = d[d["room"] > 0][["node", "sku_id", "branch_id", "room"]]
    pairs = snd.merge(rcv, on="sku_id", suffixes=("_s", "_r"))
    pairs = pairs.merge(costs.rename(columns={"from_branch": "branch_id_s", "to_branch": "branch_id_r"}),
                        on=["branch_id_s", "branch_id_r"], how="left")
    reg = branches.set_index("branch_id")["region"]
    pairs["cross_region"] = pairs["branch_id_s"].map(reg) != pairs["branch_id_r"].map(reg)
    pairs = pairs[pairs["transit_days"] <= oc["max_transit_days"]]
    if oc["same_region_only"]:
        pairs = pairs[~pairs["cross_region"]]
    pairs["unit_cost_move"] = pairs["cost_per_unit"] + np.where(
        pairs["cross_region"] & (not oc["same_region_only"]), oc["cross_region_penalty"], 0.0)
    pairs["upper"] = np.minimum(pairs["sendable"], pairs["room"])
    pairs = pairs[pairs["upper"] >= pairs["min_qty"]]
    # an arc is only worth considering if a unit can create more value than it costs to move
    gain = d.set_index("node")
    best_gain = gain.loc[pairs["node_r"], "p1"].to_numpy() + gain.loc[pairs["node_s"], "e_cost"].to_numpy()
    pairs = pairs[best_gain > pairs["unit_cost_move"].to_numpy()].reset_index(drop=True)

    on_hand_total = d.groupby("branch_id")["on_hand"].sum()
    cap = branches.set_index("branch_id")["storage_capacity_units"]
    free = (cap - on_hand_total.reindex(cap.index).fillna(0)).clip(lower=0)
    return Problem(d, pairs, free)


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
        "stockout_cost": float((nodes["p1"] * (nodes["tier1"] - cov1)).sum()
                               + (nodes["p2"] * (nodes["tier2"] - cov2)).sum()),
    }
    parts["total"] = sum(parts.values())
    return parts


def solve_milp(prob: Problem, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Solve the MILP with HiGHS; returns the plan and solver info."""
    oc = cfg["optimizer"]
    d, P = prob.nodes.set_index("node"), prob.pairs
    m = pulp.LpProblem("rebalance", pulp.LpMinimize)
    ids = P.index.to_numpy()
    x = {k: m.add_variable(f"x_{k}", lowBound=0, upBound=float(P.at[k, "upper"]), cat="Integer") for k in ids}
    y = {k: m.add_variable(f"y_{k}", cat="Binary") for k in ids}
    senders = P["node_s"].unique()
    receivers = P["node_r"].unique()
    ex = {i: m.add_variable(f"ex_{i}", lowBound=0) for i in senders}
    c1 = {j: m.add_variable(f"c1_{j}", lowBound=0, upBound=float(d.at[j, "tier1"])) for j in receivers}
    c2 = {j: m.add_variable(f"c2_{j}", lowBound=0, upBound=float(d.at[j, "tier2"])) for j in receivers}

    by_s = P.groupby("node_s").groups
    by_r = P.groupby("node_r").groups
    m += (pulp.lpSum(float(P.at[k, "unit_cost_move"]) * x[k] for k in ids)
          + pulp.lpSum(float(d.at[i, "e_cost"]) * ex[i] for i in senders)
          - pulp.lpSum(float(d.at[j, "p1"]) * c1[j] + float(d.at[j, "p2"]) * c2[j] for j in receivers))
    for i, ks in by_s.items():
        m += pulp.lpSum(x[k] for k in ks) <= float(d.at[i, "sendable"])
        m += ex[i] >= float(d.at[i, "excess_units"]) - pulp.lpSum(x[k] for k in ks)
    for j, ks in by_r.items():
        m += pulp.lpSum(x[k] for k in ks) <= float(d.at[j, "room"])
        m += c1[j] + c2[j] <= pulp.lpSum(x[k] for k in ks)
    for b, ks in P.groupby("branch_id_r").groups.items():
        m += pulp.lpSum(x[k] for k in ks) <= float(prob.free_capacity.get(b, 0))
    for k in ids:
        m += x[k] >= float(P.at[k, "min_qty"]) * y[k]
        m += x[k] <= float(P.at[k, "upper"]) * y[k]

    solver = pulp.HiGHS(msg=False, timeLimit=oc["solver_time_limit_s"], gapRel=oc["solver_gap"])
    t0 = time.time()
    status = m.solve(solver)
    info = {"status": status.status.name, "seconds": round(time.time() - t0, 1),
            "arcs": len(ids), "variables": len(m.variables()), "constraints": len(m.constraints())}
    qty = np.array([round(x[k].value() or 0) for k in ids])
    plan = P.assign(qty=qty)
    return plan[plan["qty"] > 0].reset_index(drop=True), info


def solve_greedy(prob: Problem) -> pd.DataFrame:
    """Baseline: serve the most urgent receivers first from the cheapest eligible sender."""
    d = prob.nodes.set_index("node")
    P = prob.pairs.copy()
    left_send = d["sendable"].to_dict()
    left_room = d["room"].to_dict()
    left_cap = prob.free_capacity.to_dict()
    chosen: list[tuple[int, float]] = []
    # pass 1: shortfalls, most valuable lost units first; pass 2: excess into remaining room
    P["urgency"] = d.loc[P["node_r"], "tier1"].to_numpy() * d.loc[P["node_r"], "p1"].to_numpy() \
        + d.loc[P["node_r"], "tier2"].to_numpy() * d.loc[P["node_r"], "p2"].to_numpy()
    need = (d["tier1"] + d["tier2"]).to_dict()
    for pass_no in (1, 2):
        order = P.sort_values(["urgency", "unit_cost_move"], ascending=[False, True]) if pass_no == 1 \
            else P.assign(g=d.loc[P["node_s"], "e_cost"].to_numpy() - P["unit_cost_move"]) \
            .sort_values("g", ascending=False)
        for k, r in order.iterrows():
            i, j, b = r["node_s"], r["node_r"], r["branch_id_r"]
            want = np.ceil(need[j]) if pass_no == 1 else left_room[j]
            if pass_no == 2 and d.at[i, "e_cost"] <= r["unit_cost_move"]:
                continue
            q = np.floor(min(left_send[i], left_room[j], want, left_cap.get(b, 0)))
            if q < r["min_qty"] or q <= 0:
                continue
            if pass_no == 1 and r["unit_cost_move"] >= d.at[j, "p1"]:
                continue
            chosen.append((k, q))
            left_send[i] -= q
            left_room[j] -= q
            left_cap[b] = left_cap.get(b, 0) - q
            need[j] = max(need[j] - q, 0)
    if not chosen:
        return P.iloc[0:0].assign(qty=[])
    idx, q = zip(*chosen)
    plan = P.loc[list(idx)].assign(qty=list(q))
    return plan.groupby(["node_s", "node_r"], as_index=False).agg(
        {**{c: "first" for c in P.columns if c not in ("node_s", "node_r")}, "qty": "sum"})


def describe_plan(prob: Problem, plan: pd.DataFrame) -> pd.DataFrame:
    """Attach cost, projected savings and reason code to each recommended transfer."""
    d = prob.nodes.set_index("node")
    if plan.empty:
        return pd.DataFrame(columns=["from_branch", "to_branch", "sku_id", "qty", "transfer_cost",
                                     "projected_savings", "reason_code"])
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
    p["stockout_cost_avoided"] = share_r * (cov1 * d.loc[p["node_r"], "p1"].to_numpy()
                                            + cov2 * d.loc[p["node_r"], "p2"].to_numpy())
    p["excess_cost_avoided"] = freed * d.loc[p["node_s"], "e_cost"].to_numpy()
    p["projected_savings"] = p["stockout_cost_avoided"] + p["excess_cost_avoided"] - p["transfer_cost"]
    stagnant = d.loc[p["node_s"], "is_stagnant"].to_numpy()
    p["reason_code"] = np.where(stagnant, REASON_STAGNANT_DEMAND,
                                np.where(share_r * (cov1 + cov2) > 0, REASON_EXCESS_SHORTFALL, REASON_EXCESS_DEMAND))
    p["unit_cost"] = d.loc[p["node_s"], "unit_cost"].to_numpy()
    p["value_moved"] = p["qty"] * p["unit_cost"]
    p["category"] = d.loc[p["node_s"], "category"].to_numpy()
    p["from_days_of_supply"] = d.loc[p["node_s"], "days_of_supply"].to_numpy()
    p["to_days_of_supply"] = d.loc[p["node_r"], "days_of_supply"].to_numpy()
    p["to_p_stockout_before"] = d.loc[p["node_r"], "p_stockout"].to_numpy()
    out = p.rename(columns={"branch_id_s": "from_branch", "branch_id_r": "to_branch"})
    cols = ["from_branch", "to_branch", "sku_id", "category", "qty", "unit_cost", "value_moved",
            "transit_days", "transfer_cost", "stockout_cost_avoided", "excess_cost_avoided",
            "projected_savings", "reason_code", "from_days_of_supply", "to_days_of_supply",
            "to_p_stockout_before", "cross_region"]
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


def impact(nodes: pd.DataFrame, plan: pd.DataFrame) -> dict[str, float]:
    """Business impact of a plan versus doing nothing."""
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
    return {
        "transfers": len(plan), "units_moved": float(plan["qty"].sum()),
        "value_moved": float((plan["qty"] * nodes.set_index("node").loc[plan["node_s"], "unit_cost"].to_numpy()).sum())
        if len(plan) else 0.0,
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
        "series_at_risk_before": int((a["p_stockout"] > 0.2).sum()),
        "series_at_risk_after": int((risk_after > 0.2).sum()),
        "inventory_turns_before": float(cogs / inv_before) if inv_before else np.nan,
        "inventory_turns_after": float(cogs / inv_after) if inv_after else np.nan,
        "objective_before": base["total"], "objective_after": t["total"],
        "net_savings": base["total"] - t["total"],
    }


def run(config_path: str | None = None) -> None:
    """Optimise transfers, compare with greedy, and write recommendations + impact."""
    t0 = time.time()
    cfg = load_config(config_path)
    paths = get_paths(cfg)
    policy = pd.read_csv(paths.output / "inventory_policy.csv")
    branches = pd.read_csv(paths.raw / "branches.csv")
    costs = pd.read_csv(paths.raw / "transfer_costs.csv")
    prob = build_problem(policy, branches, costs, cfg)
    print(f"[optimizer] senders={int((prob.nodes['sendable'] > 0).sum()):,} "
          f"receivers={int((prob.nodes['room'] > 0).sum()):,} arcs={len(prob.pairs):,}")

    plan, info = solve_milp(prob, cfg)
    print(f"[optimizer] MILP {info}")
    t_g = time.time()
    greedy = solve_greedy(prob)
    g_sec = time.time() - t_g

    recs = describe_plan(prob, plan)
    recs.to_csv(paths.output / "transfer_recommendations.csv", index=False)
    describe_plan(prob, greedy).to_csv(paths.output / "transfer_recommendations_greedy.csv", index=False)
    imp = pd.DataFrame({"optimizer": impact(prob.nodes, plan), "greedy": impact(prob.nodes, greedy)})
    imp.loc["solve_seconds"] = [info["seconds"], round(g_sec, 1)]
    imp.index.name = "metric"
    imp.to_csv(paths.reports / "optimizer_impact.csv")
    apply_plan(prob.nodes, plan).to_csv(paths.output / "inventory_after_transfers.csv", index=False)
    print(imp.round(2).to_string())
    print(f"[optimizer] done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Optimise inventory transfers.")
    ap.add_argument("--config", default=None)
    run(ap.parse_args().config)
