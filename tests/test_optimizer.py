"""Constraint checks for the rebalancing optimiser on a small random network."""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.optimizer import apply_plan, build_problem, describe_plan, objective_terms, solve_greedy, solve_milp


def _network(seed: int = 0, n_branch: int = 6, n_sku: int = 8):
    rng = np.random.default_rng(seed)
    branches = pd.DataFrame(
        {
            "branch_id": [f"B{i}" for i in range(n_branch)],
            "region": ["East", "East", "East", "West", "West", "West"][:n_branch],
            "storage_capacity_units": rng.integers(2_000, 4_000, n_branch),
        }
    )
    rows = []
    for s in range(n_sku):
        cost = float(rng.uniform(5, 80))
        for b in branches["branch_id"]:
            mu = float(rng.uniform(0.2, 6))
            ss = 1.65 * np.sqrt(mu * 7)
            fc_lead = mu * 7
            rop = fc_lead + ss
            tmax = mu * 21 + ss
            on_hand = float(np.round(rng.choice([0.2, 0.6, 1.0, 2.5, 4.0]) * tmax))
            stagnant = bool(rng.random() < 0.1)
            if stagnant:
                mu = 0.0
            rows.append(
                dict(
                    branch_id=b,
                    sku_id=f"S{s}",
                    category="Tools",
                    unit_cost=cost,
                    holding_cost_rate=0.25,
                    obsolescence_risk=0.05,
                    is_stagnant=stagnant,
                    forecast_mean_daily=mu,
                    fc_lead=fc_lead if mu else 0.0,
                    safety_stock=ss if mu else 0.0,
                    reorder_point=rop if mu else 0.0,
                    target_max=tmax if mu else 0.0,
                    sigma_lead=np.sqrt(mu * 7) + 0.1,
                    on_hand=on_hand,
                    inv_position=on_hand,
                    days_of_supply=on_hand / mu if mu else 999.0,
                )
            )
    pol = pd.DataFrame(rows)
    pol["excess_units"] = np.maximum(pol["inv_position"] - pol["target_max"], 0)
    pol["p_stockout"] = 0.5
    ids = branches["branch_id"]
    costs = pd.DataFrame(
        [
            (a, b, 1.0 + abs(int(a[1]) - int(b[1])) * 0.4, 1 + abs(int(a[1]) - int(b[1])) // 2)
            for a in ids
            for b in ids
            if a != b
        ],
        columns=["from_branch", "to_branch", "cost_per_unit", "transit_days"],
    )
    return pol, branches, costs


@pytest.fixture(scope="module")
def solved():
    cfg = load_config()
    pol, branches, costs = _network()
    prob = build_problem(pol, branches, costs, cfg)
    plan, info = solve_milp(prob, cfg)
    return cfg, prob, plan, info


def test_solver_finds_transfers(solved):
    _, _, plan, info = solved
    assert info["status"] == "Optimal"
    assert len(plan) > 0


def test_units_are_conserved(solved):
    _, prob, plan, _ = solved
    after = apply_plan(prob.nodes, plan)
    by_sku = after.groupby("sku_id")[["on_hand", "on_hand_after"]].sum()
    np.testing.assert_allclose(by_sku["on_hand"], by_sku["on_hand_after"])
    assert after["transfer_out"].sum() == after["transfer_in"].sum() == plan["qty"].sum()


def test_no_sender_below_safety_stock_or_floor(solved):
    _, prob, plan, _ = solved
    after = apply_plan(prob.nodes, plan)
    senders = after[after["transfer_out"] > 0]
    assert (senders["inv_position_after"] >= senders["safety_stock"] - 1e-6).all()
    assert (senders["inv_position_after"] >= senders["floor"] - 1e-6).all()
    assert (after["on_hand_after"] >= 0).all()


def test_receivers_not_pushed_above_target_and_capacity_respected(solved):
    _, prob, plan, _ = solved
    after = apply_plan(prob.nodes, plan)
    rec = after[after["transfer_in"] > 0]
    assert (rec["inv_position_after"] <= rec["target_max"] + 1e-6).all()
    inbound = plan.groupby("branch_id_r")["qty"].sum()
    assert (inbound <= prob.free_capacity.reindex(inbound.index) + 1e-6).all()


def test_minimum_quantity_and_transit_limit(solved):
    cfg, _, plan, _ = solved
    assert (plan["qty"] >= plan["min_qty"]).all()
    assert (plan["transit_days"] <= cfg["optimizer"]["max_transit_days"]).all()


def test_no_circular_transfers(solved):
    _, _, plan, _ = solved
    arcs = set(zip(plan["sku_id"], plan["branch_id_s"], plan["branch_id_r"], strict=True))
    assert not any((s, b, a) in arcs for s, a, b in arcs)
    # a branch never both sends and receives the same SKU
    assert not set(plan["node_s"]) & set(plan["node_r"])


def test_milp_at_least_as_good_as_greedy_and_beats_doing_nothing(solved):
    _, prob, plan, _ = solved
    greedy = solve_greedy(prob)
    opt = objective_terms(prob.nodes, plan)["total"]
    nothing = objective_terms(prob.nodes, plan.iloc[0:0])["total"]
    assert opt <= objective_terms(prob.nodes, greedy)["total"] + 1e-6
    assert opt < nothing


def test_recommendations_have_reasons_and_positive_value(solved):
    _, prob, plan, _ = solved
    recs = describe_plan(prob, plan)
    assert recs["reason_code"].isin(["excess->shortfall", "stagnant->demand", "excess->demand"]).all()
    assert recs["projected_savings"].sum() > 0


def test_same_region_only_is_enforced():
    cfg = load_config()
    cfg["optimizer"]["same_region_only"] = True
    pol, branches, costs = _network(seed=3)
    prob = build_problem(pol, branches, costs, cfg)
    plan, _ = solve_milp(prob, cfg)
    reg = branches.set_index("branch_id")["region"]
    assert (plan["branch_id_s"].map(reg) == plan["branch_id_r"].map(reg)).all()


def test_capacity_split_never_exceeds_free_capacity():
    from src.optimizer import allocate_capacity

    cfg = load_config()
    pol, branches, costs = _network(seed=5)
    prob = build_problem(pol, branches, costs, cfg)
    skus = sorted(prob.pairs["sku_id"].unique())
    batches = [skus[: len(skus) // 2], skus[len(skus) // 2 :]]
    # scarce capacity: every branch can take only a few units
    prob.free_capacity[:] = 5.0
    caps = allocate_capacity(prob, batches)
    for b in prob.free_capacity.index:
        assert sum(c.get(b, 0) for c in caps) <= 5.0 + 1e-9
    # ample capacity: each batch gets its full need, i.e. the split never binds
    prob.free_capacity[:] = 1e9
    caps = allocate_capacity(prob, batches)
    room = prob.nodes.set_index("node")["room"]
    for i, skus_i in enumerate(batches):
        sub = prob.pairs[prob.pairs["sku_id"].isin(skus_i)].drop_duplicates("node_r")
        need = sub.assign(room=room.loc[sub["node_r"]].to_numpy()).groupby("branch_id_r")["room"].sum()
        for b, v in need.items():
            assert caps[i][b] == pytest.approx(v)


def test_parallel_batches_respect_all_constraints():
    cfg = load_config()
    cfg["optimizer"].update(batch_arcs=5, workers=2)
    pol, branches, costs = _network(seed=7)
    from src.optimizer import check_plan, solve_decomposed

    prob = build_problem(pol, branches, costs, cfg)
    plan, info = solve_decomposed(prob, cfg)
    assert len(info) > 1
    assert not any(check_plan(prob, plan).values())
