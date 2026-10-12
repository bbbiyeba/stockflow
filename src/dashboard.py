"""Dashboard pages and helpers (the Streamlit entry point is ``app.py``).

Read-only over the batch pipeline's outputs; the only thing it writes is the planners'
approve / reject log. Every loader is cached per config file, so one process never mixes
outputs from different configurations.
"""

from __future__ import annotations

import contextlib
import html
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import access
from src import decisions as dlog
from src.config import DEFAULT_CONFIG, get_paths, load_config
from src.ui_theme import (
    ACCENT,
    CSS,
    GRID,
    INK,
    MUTED,
    SEQUENTIAL,
    STATUS,
    SUBTLE,
    SURFACE,
    money,
    pct,
    register_plotly_template,
    units,
)

# map outlines are served from ./static so the app never calls cdn.plot.ly
PLOT_CFG: dict[str, Any] = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "topojsonURL": "app/static/topojson/",
}
PAGE_SIZE = 200
HEALTH_LABEL = {"stockout_risk": "Stock-out risk", "excess": "Excess", "stagnant": "Stagnant", "healthy": "Healthy"}
REASON_LABEL = {
    "excess->shortfall": "Excess to shortfall",
    "stagnant->demand": "Stagnant to demand",
    "excess->demand": "Excess to live demand",
}
MODEL_LABEL = {
    "lightgbm": "LightGBM (global)",
    "ses": "Exp. smoothing",
    "tsb": "TSB",
    "croston_sba": "Croston (SBA)",
    "moving_average": "28-day average",
    "seasonal_naive": "Seasonal naive",
    "selected": "Selected per segment",
    "cold_start": "Cold start (peer-based)",
}
FEATURE_LABEL = {
    "dow_mean_4w": "same weekday, last 4 weeks",
    "rmean_28": "average of the last 28 days",
    "rmean_7": "average of the last 7 days",
    "rmean_365_lag28": "average over the past year",
    "rmean_91_lag28": "average over the past quarter",
    "rmean_28_lag28": "average a month ago",
    "rstd_28": "recent day-to-day volatility",
    "rstd_91_lag28": "volatility over the past quarter",
    "yoy_mean_14": "same two weeks last year",
    "lag_7": "sales a week ago",
    "lag_14": "sales two weeks ago",
    "lag_28": "sales four weeks ago",
    "lag_364": "same weekday last year",
    "lag_365": "same date last year",
    "nz_share_91_lag28": "how often it sells",
    "dow": "day of week",
    "dom": "day of month",
    "month": "month",
    "weekofyear": "week of year",
    "doy_sin": "season",
    "doy_cos": "season",
    "is_closed_holiday": "holiday closure",
    "is_minor_holiday": "minor holiday",
    "is_yearend_break": "year-end break",
    "days_to_closed_holiday": "approaching holiday",
    "days_after_closed_holiday": "just after a holiday",
    "discount": "promotion discount",
    "on_promo": "on promotion",
    "days_into_promo": "promotion timing",
    "post_promo_7d": "week after a promotion",
    "net_price": "price",
    "price_index": "price vs list",
    "branch_id": "branch",
    "sku_id": "SKU",
    "region": "region",
    "size_tier": "branch size",
    "category": "category",
    "unit_cost": "unit cost",
    "list_price": "list price",
    "lead_time_days": "lead time",
    "case_pack": "case pack",
    "obsolescence_risk": "obsolescence risk",
}
MASK = access.mask_financials()


def esc(x: object) -> str:
    """HTML-escape any data-derived value before it goes into markup."""
    return html.escape(str(x), quote=True)


def m(x: float, decimals: int = 1) -> str:
    """Money, unless financials are masked for a demo / screenshot."""
    return "$•••" if MASK else money(x, decimals)


# --------------------------------------------------------------------------- data
def setup() -> None:
    """Theme and CSS; call once per script run after ``st.set_page_config``."""
    register_plotly_template()
    st.markdown(CSS, unsafe_allow_html=True)


def config_key() -> str:
    """The config file in use; part of every cache key."""
    return str(os.environ.get("STOCKFLOW_CONFIG") or DEFAULT_CONFIG)


@st.cache_resource
def _paths_for(cfg_path: str) -> Any:
    return get_paths(load_config(cfg_path))


def _paths() -> Any:
    return _paths_for(config_key())


def _csv(p: Path, **kw: Any) -> pd.DataFrame | None:
    return pd.read_csv(p, **kw) if p.exists() else None


def load_outputs() -> dict[str, Any]:
    """Small pipeline outputs. Raises FileNotFoundError if the pipeline has not run."""
    return _load_outputs(config_key())


@st.cache_data(show_spinner="Loading pipeline outputs…", ttl=600)
def _load_outputs(cfg_path: str) -> dict[str, Any]:
    p = _paths_for(cfg_path)
    out: dict[str, Any] = {
        "policy": pd.read_parquet(p.output / "inventory_policy.parquet"),
        "transfers": pd.read_parquet(p.output / "transfer_recommendations.parquet"),
        "impact": pd.read_csv(p.reports / "optimizer_impact.csv", index_col="metric"),
        "holdout": pd.read_csv(p.reports / "holdout_metrics_overall.csv"),
        "selection": pd.read_csv(p.reports / "model_selection.csv"),
        "vs_naive": pd.read_csv(p.reports / "holdout_selected_vs_naive_by_segment.csv"),
        "by_abc": pd.read_csv(p.reports / "holdout_selected_by_abc.csv"),
        "bias": pd.read_csv(p.reports / "bias_check.csv"),
        "coverage": pd.read_csv(p.reports / "interval_coverage.csv"),
        "branches": pd.read_csv(p.raw / "branches.csv"),
        "sensitivity": _csv(p.reports / "sensitivity.csv"),
        "explain": (
            pd.read_parquet(p.output / "forecast_explanations.parquet")
            if (p.output / "forecast_explanations.parquet").exists()
            else None
        ),
    }
    man = p.output / "run_manifest.json"
    out["manifest"] = json.loads(man.read_text()) if man.exists() else {}
    return out


def load_forecast(branch: str, sku: str) -> pd.DataFrame:
    return _load_forecast(config_key(), branch, sku)


@st.cache_data(show_spinner=False, max_entries=128, ttl=600)
def _load_forecast(cfg_path: str, branch: str, sku: str) -> pd.DataFrame:
    return pd.read_parquet(
        _paths_for(cfg_path).output / "forecasts", filters=[("branch_id", "==", branch), ("sku_id", "==", sku)]
    )


def load_history(branch: str, sku: str) -> pd.DataFrame:
    return _load_history(config_key(), branch, sku)


@st.cache_data(show_spinner=False, max_entries=128, ttl=600)
def _load_history(cfg_path: str, branch: str, sku: str) -> pd.DataFrame:
    """Cleaned daily history for one series; partition + predicate push-down keep this cheap."""
    return pd.read_parquet(
        _paths_for(cfg_path).processed / "sales_clean",
        columns=["date", "qty", "is_stockout"],
        filters=[("branch_id", "==", branch), ("sku_id", "==", sku)],
    )


def load_decisions(run: str) -> pd.DataFrame:
    return dlog.latest(_paths().decisions / "transfer_decisions.csv", run)


def get_data() -> dict[str, Any]:
    try:
        return load_outputs()
    except FileNotFoundError:
        st.title("No pipeline outputs yet")
        st.markdown("The dashboard reads the files written by the batch pipeline. Generate them with:")
        st.code("make run   # or: python run_pipeline.py", language="bash")
        st.stop()
    except Exception:  # noqa: BLE001 - never show a traceback or file paths in the UI
        st.title("Could not load pipeline outputs")
        st.markdown(
            "The output files exist but could not be read; they may be from an older version or a run "
            "that did not finish. Re-run the pipeline. If this persists, check the pipeline log."
        )
        st.stop()
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------- ui helpers
def lede(text: str, kicker: str | None = None) -> None:
    """Page headline: the decision the page supports. ``text`` must already be escaped."""
    if kicker:
        st.markdown(f'<div class="kicker">{esc(kicker)}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="lede">{text}</p>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<p class="note">{esc(text)}</p>', unsafe_allow_html=True)


def csv_button(df: pd.DataFrame, name: str, label: str = "Download CSV") -> None:
    st.download_button(
        label,
        df.to_csv(index=False).encode(),
        file_name=name,
        mime="text/csv",
        help="Downloads every row that matches the current filters",
    )


def page_slice(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Paginate large tables so the browser never receives hundreds of thousands of rows."""
    if len(df) <= PAGE_SIZE:
        return df
    pages = math.ceil(len(df) / PAGE_SIZE)
    c1, c2 = st.columns([1, 3])
    page = c1.number_input(f"Page (of {pages:,})", min_value=1, max_value=pages, value=1, key=f"{key}_page")
    c2.markdown(
        f'<p class="note" style="margin-top:2.1rem">Rows {(page - 1) * PAGE_SIZE + 1:,}–'
        f"{min(page * PAGE_SIZE, len(df)):,} of {len(df):,}. Sort within a page by clicking a column; "
        f"the CSV contains all rows.</p>",
        unsafe_allow_html=True,
    )
    return df.iloc[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]


def money_col(label: str | None = None, **kw: Any) -> Any:
    if MASK:
        return st.column_config.TextColumn(label, **kw)
    return st.column_config.NumberColumn(label, format="$%,.0f", **kw)


def mask_money_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.assign(**{c: "•••" for c in cols if c in df}) if MASK else df


# --------------------------------------------------------------------------- pages
def page_transfers(user: access.User) -> None:
    d = get_data()
    tr, imp, br = d["transfers"], d["impact"], d["branches"]
    run_id = str(d["manifest"].get("run_id", "unknown"))
    st.title("Transfer recommendations")
    if tr.empty:
        st.info("The optimiser found no transfer worth its cost in this run.")
        return
    o, g = imp["optimizer"], imp["greedy"]
    lede(
        f"Moving <b>{units(o['units_moved'])} units</b> in {int(o['transfers']):,} transfers costs "
        f"{m(o['transfer_cost'])} and saves a net <b>{m(o['net_savings'])}</b>: it covers "
        f"{units(o['stockout_units_avoided'])} units of expected lost sales and frees {m(o['excess_value_freed'])} "
        f"of excess. That is {m(o['net_savings'] - g['net_savings'])} more than a greedy rule-of-thumb plan.",
        kicker="Recommendations only · nothing moves until a planner approves it",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Net savings",
        m(o["net_savings"]),
        f"{m(o['net_savings'] - g['net_savings'])} vs greedy",
        delta_color="off" if MASK else "normal",
    )
    c2.metric("Expected lost units avoided", units(o["stockout_units_avoided"]))
    c3.metric("Excess freed", m(o["excess_value_freed"]))
    c4.metric(
        "Inventory turns",
        f"{o['inventory_turns_after']:.1f}",
        f"{o['inventory_turns_after'] - o['inventory_turns_before']:+.1f} from {o['inventory_turns_before']:.1f}",
    )

    with st.sidebar:
        st.markdown("### Filter transfers")
        f_from = st.multiselect("From branch", sorted(tr["from_branch"].unique()))
        f_to = st.multiselect("To branch", sorted(tr["to_branch"].unique()))
        f_cat = st.multiselect("Category", sorted(tr["category"].unique()))
        f_reason = st.multiselect("Reason", list(REASON_LABEL.values()))
        f_status = st.multiselect("Decision status", ["Pending", "Approved", "Rejected"])
        min_sav = st.number_input("Minimum net savings ($)", min_value=0, value=0, step=50, disabled=MASK)

    dec = load_decisions(run_id)
    v = tr.merge(dec[["transfer_id", "decision", "reviewer"]], on="transfer_id", how="left")
    v["Status"] = v["decision"].map({"approved": "Approved", "rejected": "Rejected"}).fillna("Pending")
    v["Reason"] = v["reason_code"].map(REASON_LABEL)
    for col, sel in (
        ("from_branch", f_from),
        ("to_branch", f_to),
        ("category", f_cat),
        ("Reason", f_reason),
        ("Status", f_status),
    ):
        if sel:
            v = v[v[col].isin(sel)]
    v = v[v["projected_savings"] >= min_sav]
    if v.empty:
        st.info("No transfers match these filters. Clear a filter in the sidebar to see more.")
        return

    st.markdown("## Main transfer corridors")
    flows = v.groupby(["from_branch", "to_branch"], as_index=False).agg(
        units=("qty", "sum"), savings=("projected_savings", "sum")
    )
    top_n = (
        st.slider("Corridors shown (largest by units)", 5, max(5, min(100, len(flows))), min(25, len(flows)), step=5)
        if len(flows) > 5
        else len(flows)
    )
    shown = flows.nlargest(top_n, "units")
    loc = br.set_index("branch_id")
    fig = go.Figure()
    mx = shown["units"].max()
    for r in shown.itertuples():
        a, b = loc.loc[r.from_branch], loc.loc[r.to_branch]
        fig.add_trace(
            go.Scattergeo(
                lon=[a.lon, b.lon],
                lat=[a.lat, b.lat],
                mode="lines",
                line=dict(width=1 + 6 * r.units / mx, color="rgba(11,110,105,0.6)"),
                hoverinfo="text",
                text=f"{r.from_branch} → {r.to_branch}: {r.units:,.0f} units, {m(r.savings)} net",
            )
        )
    send = shown.groupby("from_branch")["units"].sum()
    recv = shown.groupby("to_branch")["units"].sum()
    active = sorted(set(shown["from_branch"]) | set(shown["to_branch"]))
    nodes = loc.loc[active].assign(sent=send.reindex(active).fillna(0), received=recv.reindex(active).fillna(0))
    net_sender = nodes["sent"] > nodes["received"]
    fig.add_trace(
        go.Scattergeo(
            lon=nodes["lon"],
            lat=nodes["lat"],
            mode="markers+text",
            text=nodes.index,
            textposition="top center",
            textfont=dict(size=10, color=INK),
            marker=dict(
                size=9,
                symbol=np.where(net_sender, "circle", "circle-open"),
                color=ACCENT,
                line=dict(color=ACCENT, width=1.6),
            ),
            hovertext=[
                f"{i} ({n.branch_name}): sends {n.sent:,.0f}, receives {n.received:,.0f} units"
                for i, n in nodes.iterrows()
            ],
            hoverinfo="text",
        )
    )
    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor=SURFACE,
        subunitcolor="#d9dee3",
        countrycolor="#d9dee3",
        showlakes=False,
    )
    fig.update_layout(
        height=470,
        margin=dict(l=0, r=0, t=40, b=0),
        title=f"Top {len(shown)} of {len(flows):,} corridors by units moved",
    )
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    note(
        f"Line width is proportional to units moved. Filled dots are net senders, hollow dots net receivers. "
        f"The corridors shown carry {shown['units'].sum() / flows['units'].sum():.0%} of the filtered units."
    )

    st.markdown(f"## {len(v):,} recommended transfers")
    t = v.rename(
        columns={
            "from_branch": "From",
            "to_branch": "To",
            "sku_id": "SKU",
            "category": "Category",
            "qty": "Units",
            "value_moved": "Value moved",
            "transit_days": "Transit days",
            "transfer_cost": "Transfer cost",
            "projected_savings": "Net savings",
            "from_days_of_supply": "Sender days of supply",
            "to_days_of_supply": "Receiver days of supply",
            "reason": "Why",
        }
    )
    cols = [
        "From",
        "To",
        "SKU",
        "Status",
        "Reason",
        "Units",
        "Value moved",
        "Transfer cost",
        "Net savings",
        "Transit days",
        "Sender days of supply",
        "Receiver days of supply",
        "Category",
        "Why",
        "transfer_id",
    ]
    t = mask_money_cols(t[cols], ["Value moved", "Transfer cost", "Net savings"])
    page = page_slice(t, "transfers").copy()
    colcfg = {
        "From": st.column_config.TextColumn(pinned=True),
        "To": st.column_config.TextColumn(pinned=True),
        "SKU": st.column_config.TextColumn(pinned=True),
        "Units": st.column_config.NumberColumn(format="%d"),
        "Value moved": money_col(),
        "Transfer cost": money_col(),
        "Net savings": money_col(),
        "Sender days of supply": st.column_config.NumberColumn(format="%.0f"),
        "Receiver days of supply": st.column_config.NumberColumn(format="%.0f"),
        "Why": st.column_config.TextColumn(width="large"),
        "transfer_id": st.column_config.TextColumn("Transfer ID"),
    }
    if user.role == "approver":
        page.insert(3, "Decide", "")
        page.insert(4, "Note", "")
        edited = st.data_editor(
            page,
            hide_index=True,
            width="stretch",
            height=440,
            key="review_editor",
            disabled=[c for c in page.columns if c not in ("Decide", "Note")],
            column_config={
                **colcfg,
                "Decide": st.column_config.SelectboxColumn(
                    options=["", "Approve", "Reject"], help="Your decision for this transfer"
                ),
                "Note": st.column_config.TextColumn(max_chars=dlog.MAX_NOTE),
            },
        )
        picked = edited[edited["Decide"].isin(["Approve", "Reject"])]
        reviewer = user.name
        if not user.authenticated:
            reviewer = st.text_input(
                "Your name (recorded with each decision)",
                max_chars=80,
                help="Demo mode: there is no sign-in, so decisions carry the name you type.",
            )
        c1, c2 = st.columns([1, 3])
        if c1.button(f"Record {len(picked)} decision(s)", type="primary", disabled=picked.empty):
            try:
                n = dlog.record(
                    _paths().decisions / "transfer_decisions.csv",
                    run_id,
                    [
                        {
                            "transfer_id": r.transfer_id,
                            "decision": "approved" if r.Decide == "Approve" else "rejected",
                            "note": r.Note,
                        }
                        for r in picked.itertuples()
                    ],
                    reviewer,
                    user.role,
                )
                st.success(f"Recorded {n} decision(s). Approved transfers still have to be raised in the ERP.")
            except dlog.DecisionError as e:
                st.error(f"Not recorded: {e}")
            except OSError:
                st.error("Not recorded: the decision log could not be written. Nothing was changed.")
        c2.markdown(
            '<p class="note" style="margin-top:0.5rem">Decisions are appended to an audit log with your '
            "name, the time and this pipeline run. Nothing is executed automatically.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.dataframe(page, hide_index=True, width="stretch", height=440, column_config=colcfg)
        note("View only. Approving or rejecting transfers requires the approver role.")
    csv_button(t, "transfer_recommendations.csv")

    st.markdown("## Optimiser vs greedy baseline")
    rows = [
        ("Net savings", "net_savings", m),
        ("Transfer cost", "transfer_cost", m),
        ("Units moved", "units_moved", units),
        ("Expected lost units avoided", "stockout_units_avoided", units),
        ("Excess freed", "excess_value_freed", m),
        ("Positions still at stock-out risk", "series_at_risk_after", lambda x: f"{x:,.0f}"),
        ("Inventory turns after", "inventory_turns_after", lambda x: f"{x:.2f}"),
    ]
    st.dataframe(
        pd.DataFrame([(lbl, f(o[k]), f(g[k])) for lbl, k, f in rows], columns=["Measure", "Optimiser", "Greedy"]),
        hide_index=True,
        width="content",
    )
    note(
        "The optimiser minimises total cost, so it skips some low-value fixes the greedy rule makes; "
        "it can leave a few more positions at risk while still saving more money overall."
    )

    sens = d["sensitivity"]
    st.markdown("## How sensitive is the plan?")
    if sens is None or sens.empty:
        st.info("No sensitivity analysis in this run. Generate it with `make sensitivity`.")
    else:
        s = sens.copy()
        s["Net savings"] = s["net_savings"].map(m)
        s["Excess freed"] = s["excess_value_freed"].map(m)
        st.dataframe(
            s.rename(
                columns={
                    "scenario": "Scenario",
                    "setting": "Setting",
                    "transfers": "Transfers",
                    "units_moved": "Units moved",
                    "stockout_units_avoided": "Lost units avoided",
                }
            )[["Scenario", "Setting", "Transfers", "Units moved", "Lost units avoided", "Net savings", "Excess freed"]],
            hide_index=True,
            width="stretch",
            column_config={
                "Units moved": st.column_config.NumberColumn(format="%,.0f"),
                "Lost units avoided": st.column_config.NumberColumn(format="%,.0f"),
            },
        )
        note("Each row re-runs the policy and optimiser with one assumption changed; everything else is the base case.")


def page_health(user: access.User) -> None:
    d = get_data()
    pol = d["policy"]
    st.title("Inventory health")
    risk = pol[pol["stockout_risk"]]
    liq = pol[pol["liquidation_candidate"]]
    lede(
        f"<b>{len(risk):,} branch × SKU positions</b> at {risk['branch_id'].nunique()} branches are likely to "
        f"stock out before replenishment arrives. Meanwhile <b>{m(pol['excess_value'].sum())}</b> sits above "
        f"target levels, and <b>{m(liq['on_hand_value'].sum())}</b> of stagnant stock has no realistic buyer "
        f"anywhere in the network: a liquidation decision, not a transfer.",
        kicker="Position as of the last day of history",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Positions at stock-out risk", f"{len(risk):,}")
    c2.metric("Excess above target", m(pol["excess_value"].sum()))
    c3.metric(
        "Stagnant, no network demand",
        m(liq["on_hand_value"].sum()),
        help=f"{len(liq):,} positions: stagnant here, and the whole network holds over a year of supply",
    )

    options = ["Stock-out risk positions", "Days of supply (median)"] + ([] if MASK else ["Excess $", "Stagnant $"])
    metric = st.radio("Colour the grid by", options, horizontal=True)
    agg = {
        "Stock-out risk positions": ("stockout_risk", "sum"),
        "Excess $": ("excess_value", "sum"),
        "Stagnant $": ("stagnant_value", "sum"),
        "Days of supply (median)": ("days_of_supply", "median"),
    }[metric]
    grid = pol.pivot_table(index="branch_id", columns="category", values=agg[0], aggfunc=agg[1]).fillna(0)
    if metric.startswith("Days"):
        grid = grid.clip(upper=365)
    grid = grid.loc[grid.sum(axis=1).sort_values(ascending=False).index]
    if len(grid) > 40:
        note(f"Showing the 40 branches with the highest totals of {len(grid)}; the table below has every position.")
        grid = grid.head(40)
    fmt: Callable[[float], str] = (lambda v: money(v, 0)) if metric.endswith("$") else (lambda v: f"{v:.0f}")
    text = [[fmt(v) if v else "" for v in row] for row in grid.to_numpy()]
    fig = go.Figure(
        go.Heatmap(
            z=grid.to_numpy(),
            x=list(grid.columns),
            y=list(grid.index),
            colorscale=SEQUENTIAL,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11),
            xgap=2,
            ygap=2,
            colorbar=dict(title=dict(text=metric, side="right"), thickness=10),
            hovertemplate="%{y} · %{x}: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(460, 24 * len(grid) + 140),
        title=dict(text=f"{metric} by branch and category", y=0.99, yanchor="top"),
        margin=dict(l=8, r=8, t=96, b=8),
        xaxis=dict(side="top", title=None, tickangle=0),
        yaxis=dict(autorange="reversed", title=None),
    )
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    note(
        "Each cell shows its value as text, so the grid reads without relying on colour. "
        "Days of supply are capped at 365 for display."
    )

    st.markdown("## Positions needing action")
    view = pol[pol["health"] != "healthy"].copy()
    view["Status"] = view["health"].map(HEALTH_LABEL)
    view.loc[view["liquidation_candidate"], "Status"] = "Liquidate"
    status_pick = st.multiselect(
        "Status", ["Stock-out risk", "Excess", "Stagnant", "Liquidate"], default=["Stock-out risk"]
    )
    t = view.rename(
        columns={
            "branch_id": "Branch",
            "sku_id": "SKU",
            "category": "Category",
            "abc_class": "ABC",
            "on_hand": "On hand",
            "days_of_supply": "Days of supply",
            "safety_stock": "Safety stock",
            "reorder_point": "Reorder point",
            "p_stockout": "Stock-out probability",
            "excess_value": "Excess $",
            "stagnant_value": "Stagnant $",
        }
    )
    cols = [
        "Branch",
        "SKU",
        "Status",
        "Category",
        "ABC",
        "On hand",
        "Days of supply",
        "Safety stock",
        "Reorder point",
        "Stock-out probability",
        "Excess $",
        "Stagnant $",
    ]
    t = t[t["Status"].isin(status_pick)][cols].sort_values(["Stock-out probability", "Excess $"], ascending=False)
    t["Stock-out probability"] *= 100
    t = mask_money_cols(t, ["Excess $", "Stagnant $"])
    if t.empty:
        st.info("No positions match the selected statuses.")
        return
    st.dataframe(
        page_slice(t, "health"),
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "Branch": st.column_config.TextColumn(pinned=True),
            "SKU": st.column_config.TextColumn(pinned=True),
            "On hand": st.column_config.NumberColumn(format="%d"),
            "Days of supply": st.column_config.NumberColumn(format="%.0f"),
            "Safety stock": st.column_config.NumberColumn(format="%.1f"),
            "Reorder point": st.column_config.NumberColumn(format="%.1f"),
            "Stock-out probability": st.column_config.NumberColumn(format="%.0f%%"),
            "Excess $": money_col(),
            "Stagnant $": money_col(),
        },
    )
    csv_button(t, "inventory_positions.csv")


def page_forecast(user: access.User) -> None:
    d = get_data()
    pol = d["policy"]
    st.title("Forecast explorer")
    with st.sidebar:
        st.markdown("### Series")
        b = st.selectbox("Branch", sorted(pol["branch_id"].unique()), index=0)
        skus = pol.loc[pol["branch_id"] == b].sort_values("forecast_28d", ascending=False)
        labels = {r.sku_id: f"{r.sku_id} · {r.category} · class {r.abc_class}" for r in skus.itertuples()}
        s = st.selectbox("SKU (sorted by forecast volume)", list(labels), format_func=labels.get)
        days = st.slider("History shown (days)", 28, 365, 120, step=7)
    row = pol[(pol["branch_id"] == b) & (pol["sku_id"] == s)]
    try:
        f = load_forecast(b, s).sort_values("date")
    except Exception:  # noqa: BLE001
        f = pd.DataFrame()
    if row.empty or f.empty:
        st.info("No forecast for this branch and SKU. It may not be stocked at this branch.")
        return
    r = row.iloc[0]
    cover = "more than a year" if r["days_of_supply"] >= 365 else f"{r['days_of_supply']:.0f} days"
    lede(
        f"{esc(b)} is expected to sell <b>{units(r['forecast_28d'])} units</b> of {esc(s)} over the next 28 days "
        f"(80% range {units(r['p10_28d_approx'])}–{units(r['p90_28d_approx'])}). "
        f"On-hand stock covers <b>{cover}</b>. Status: {esc(HEALTH_LABEL.get(r['health'], r['health']))}.",
        kicker=f"{b} · {s}",
    )
    if r.get("is_cold_start", False):
        st.info(
            "New series with little history: this forecast is borrowed from the same SKU at other branches, "
            "scaled to this branch's size. Treat it as a starting point."
        )

    try:
        h = load_history(b, s)
    except Exception:  # noqa: BLE001
        h = pd.DataFrame()
    if not h.empty:
        h = h[h["date"] > h["date"].max() - pd.Timedelta(days=days)]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(f["date"]) + list(f["date"][::-1]),
            y=list(f["p90"]) + list(f["p10"][::-1]),
            fill="toself",
            fillcolor="rgba(11,110,105,0.14)",
            line=dict(width=0),
            hoverinfo="skip",
            name="80% interval",
        )
    )
    if not h.empty:
        fig.add_trace(
            go.Scatter(
                x=h["date"],
                y=h["qty"],
                mode="lines",
                line=dict(color=MUTED, width=1.2),
                hovertemplate="%{x|%a %d %b}: %{y:.0f} units<extra>Actual</extra>",
            )
        )
        so = h[h["is_stockout"]]
        if len(so):
            fig.add_trace(
                go.Scatter(
                    x=so["date"],
                    y=so["qty"],
                    mode="markers",
                    marker=dict(symbol="x", size=7, color=STATUS["risk"]),
                    hovertemplate="%{x|%d %b}: stock-out, sales understate demand<extra></extra>",
                )
            )
        fig.add_vline(x=f["date"].min(), line=dict(color=GRID, dash="dot"))
        fig.add_annotation(
            x=f["date"].min(),
            y=1,
            yref="paper",
            text="forecast",
            showarrow=False,
            xanchor="left",
            font=dict(size=11, color=MUTED),
        )
    fig.add_trace(
        go.Scatter(
            x=f["date"],
            y=f["forecast"],
            mode="lines",
            line=dict(color=ACCENT, width=2.2),
            hovertemplate="%{x|%a %d %b}: %{y:.1f} units<extra>Forecast</extra>",
        )
    )
    for q, y in (("P90", f["p90"].iloc[-1]), ("P10", f["p10"].iloc[-1])):
        fig.add_annotation(
            x=f["date"].iloc[-1], y=y, text=q, showarrow=False, xanchor="left", font=dict(size=11, color=ACCENT)
        )
    fig.update_layout(height=380, title=f"Daily units, {s} at {b}", yaxis_title="Units per day", xaxis_title=None)
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    note(
        "Grey: cleaned daily sales. Teal line and band: forecast with its 10th–90th percentile range. "
        "Red crosses: stock-out days, where sales understate demand."
    )

    ex = d["explain"]
    if ex is not None:
        e = ex[(ex["branch_id"] == b) & (ex["sku_id"] == s)].sort_values("rank")
        if not e.empty and r["model"] != "cold_start":
            st.markdown("## What drives this forecast")
            lines = []
            for x in e.itertuples():
                factor = math.exp(x.shap_log)
                lines.append(
                    f"- **{esc(FEATURE_LABEL.get(x.feature, x.feature)).capitalize()}**: "
                    f"×{factor:.2f} ({'higher' if factor >= 1 else 'lower'})"
                )
            st.markdown("\n".join(lines))
            used = MODEL_LABEL.get(r["model"], r["model"])
            note(
                "How much each factor moves this forecast compared with a typical branch × SKU in the network "
                "(SHAP contributions from the global LightGBM model, averaged over the horizon; factors multiply). "
                f"The forecast shown uses {used}"
                + (", so these describe the ML model's view of this series." if r["model"] != "lightgbm" else ".")
            )

    st.markdown("## Replenishment policy")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("On hand", units(r["on_hand"]), help="Units at the branch at the end of the last day of history")
    c2.metric(
        "Reorder point",
        units(r["reorder_point"]),
        help="Lead-time demand plus safety stock. Reorder when stock falls below this.",
    )
    c3.metric(
        "Safety stock",
        units(r["safety_stock"]),
        help=f"Covers forecast error over the {int(r['lead_time_days'])}-day lead time at the target service level",
    )
    c4.metric("Chance of stock-out in lead time", pct(r["p_stockout"], 0))
    note(
        f"Model: {MODEL_LABEL.get(r['model'], r['model'])} "
        f"(chosen for segment {str(r['segment']).replace('|', ' · ')}). "
        f"Lead time {int(r['lead_time_days'])} days; target maximum {units(r['target_max'])} units; "
        f"suggested order {units(r['recommended_order_qty'])} units."
    )


def page_accuracy(user: access.User) -> None:
    d = get_data()
    ho, vs = d["holdout"].set_index("model"), d["vs_naive"]
    st.title("Forecast accuracy")
    sel, naive = ho.loc["selected"], ho.loc["seasonal_naive"]
    ties = (vs["selected"] - vs["seasonal_naive"]).abs() < 1e-9
    losers = vs[~vs["selected_beats_naive"] & ~ties]
    worst = vs.sort_values("selected", ascending=False).iloc[0]
    lede(
        f"On the most recent 28 days, which no model saw during selection, the chosen models miss by "
        f"<b>{pct(sel['wape'], 0)}</b> of daily volume (WAPE) against {pct(naive['wape'], 0)} for a "
        f"seasonal-naive baseline: <b>{pct(1 - sel['wape'] / naive['wape'], 0)} better</b>, with a bias of "
        f"{pct(sel['bias'], 1)}. Over a full 28-day total the error is {pct(sel['wape_28d_total'], 0)}. "
        f"Weakest segment: {esc(str(worst['segment']).replace('|', ' · '))} ({pct(worst['selected'], 0)}).",
        kicker="Held-out fold · models chosen on the three earlier folds",
    )
    if len(losers):
        st.warning(
            f"The selected model does not beat seasonal naive in {len(losers)} segment(s): "
            + ", ".join(esc(x.replace("|", " · ")) for x in losers["segment"])
        )

    left, right = st.columns([1.1, 1])
    o = ho.reset_index().sort_values("wape")
    fig = go.Figure(
        go.Bar(
            y=[MODEL_LABEL.get(x, x) for x in o["model"]],
            x=o["wape"] * 100,
            orientation="h",
            marker_color=[ACCENT if x == "selected" else SUBTLE for x in o["model"]],
            text=[pct(v, 1) for v in o["wape"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: WAPE %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        title="WAPE by model, held-out fold (lower is better)",
        xaxis=dict(title="WAPE (% of actual units)", range=[0, o["wape"].max() * 118]),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=8, r=40, t=48, b=8),
    )
    left.plotly_chart(fig, width="stretch", config=PLOT_CFG)

    abc = d["by_abc"].sort_values("abc_class")
    fig2 = go.Figure(
        go.Bar(
            x=[f"Class {c}" for c in abc["abc_class"]],
            y=abc["wape"] * 100,
            marker_color=ACCENT,
            text=[pct(v, 0) for v in abc["wape"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}: WAPE %{y:.1f}%<extra></extra>",
        )
    )
    fig2.update_layout(
        height=320,
        title="Selected models: WAPE by ABC class",
        yaxis=dict(title="WAPE (%)", range=[0, abc["wape"].max() * 125]),
    )
    right.plotly_chart(fig2, width="stretch", config=PLOT_CFG)
    note(
        "C-class items sell on a minority of days, so daily error is large for any model; "
        "the 28-day totals that drive safety stock are far more accurate."
    )

    st.markdown("## Model chosen for each segment")
    s = d["selection"].merge(
        vs[["segment", "selected", "seasonal_naive", "selected_beats_naive"]], on="segment", how="left"
    )
    t = pd.DataFrame(
        {
            "Segment": s["segment"].str.replace("|", " · ", regex=False),
            "Model": s["model"].map(MODEL_LABEL),
            "Held-out WAPE": s["selected"] * 100,
            "Naive WAPE": s["seasonal_naive"] * 100,
            "Beats naive": np.where(
                (s["selected"] - s["seasonal_naive"]).abs() < 1e-9,
                "Tie (both forecast zero)",
                np.where(s["selected_beats_naive"], "Yes", "No"),
            ),
            "Bias (all folds)": s["bias"] * 100,
            "MASE": s["mase"],
        }
    )
    st.dataframe(
        t,
        hide_index=True,
        width="stretch",
        column_config={
            "Segment": st.column_config.TextColumn(pinned=True),
            "Held-out WAPE": st.column_config.NumberColumn(format="%.1f%%"),
            "Naive WAPE": st.column_config.NumberColumn(format="%.1f%%"),
            "Bias (all folds)": st.column_config.NumberColumn(format="%+.1f%%", help="Positive = over-forecast"),
            "MASE": st.column_config.NumberColumn(
                format="%.2f", help="Error relative to seasonal naive in-sample; <1 is better"
            ),
        },
    )
    csv_button(t, "model_selection.csv")

    st.markdown("## Bias check")
    bias = d["bias"]
    flagged = bias[bias["flag"] == "check"]
    dims = {"abc_class": "ABC class", "branch_id": "Branch", "category": "Category", "pattern": "Demand pattern"}
    if flagged.empty:
        st.success("No ABC class, branch, category or demand pattern is biased by more than 5% on the held-out fold.")
    else:
        note(f"{len(flagged)} group(s) are biased by more than 5% on the held-out fold:")
        st.dataframe(
            pd.DataFrame(
                {
                    "Dimension": flagged["dimension"].map(dims),
                    "Group": flagged["group"],
                    "Bias": flagged["bias"] * 100,
                    "WAPE": flagged["wape"] * 100,
                    "Direction": np.where(flagged["bias"] > 0, "Over-forecast", "Under-forecast"),
                }
            ),
            hide_index=True,
            width="content",
            column_config={
                "Bias": st.column_config.NumberColumn(format="%+.1f%%"),
                "WAPE": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
    bb = bias[bias["dimension"] == "branch_id"].sort_values("bias")
    fig3 = go.Figure(
        go.Scatter(
            x=bb["bias"] * 100,
            y=bb["group"],
            mode="markers",
            marker=dict(
                size=9,
                color=[STATUS["risk"] if abs(v) > 0.05 else ACCENT for v in bb["bias"]],
                symbol=["diamond" if abs(v) > 0.05 else "circle" for v in bb["bias"]],
            ),
            hovertemplate="%{y}: bias %{x:+.1f}%<extra></extra>",
        )
    )
    fig3.add_vrect(x0=-5, x1=5, fillcolor=SURFACE, line_width=0, layer="below")
    fig3.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig3.update_layout(
        height=max(320, 18 * len(bb) + 100),
        title="Bias by branch (shaded band = ±5%; diamonds fall outside)",
        xaxis_title="Bias (% of actual units; + means over-forecast)",
    )
    st.plotly_chart(fig3, width="stretch", config=PLOT_CFG)
    cov = d["coverage"]
    allc = cov.loc[cov["segment"] == "ALL", "p10_p90_coverage"].iloc[0]
    note(
        f"Prediction intervals: the P10–P90 band contained {allc:.0%} of actual daily sales on the held-out fold "
        f"(target 80%)."
    )


# --------------------------------------------------------------------------- shell
PAGES: dict[str, tuple[str, Callable[[access.User], None]]] = {
    "transfers": ("Transfer recommendations", page_transfers),
    "health": ("Inventory health", page_health),
    "forecast": ("Forecast explorer", page_forecast),
    "accuracy": ("Forecast accuracy", page_accuracy),
}


def current_user() -> access.User:
    info = None
    try:
        info = st.user.to_dict() if st.user.is_logged_in else None
    except Exception:  # noqa: BLE001 - auth not configured
        info = None
    return access.resolve(info)


def main() -> None:
    """App shell: sign-in gate, navigation and run footer."""
    setup()
    user = current_user()
    if access.require_login() and not user.authenticated:
        st.title("Sign in to Stockflow")
        st.markdown("Inventory and cost data are confidential. Sign in with your company account to continue.")
        if st.button("Sign in", type="primary"):
            st.login()
        st.stop()
    with st.sidebar:
        manifest: dict[str, Any] = {}
        with contextlib.suppress(Exception):  # the page itself shows the empty / error state
            manifest = load_outputs()["manifest"]
        nav = st.navigation(
            [
                st.Page(lambda f=fn: f(user), title=title, url_path=key, default=key == "transfers")
                for key, (title, fn) in PAGES.items()
            ]
        )
        footer = [
            (
                f"Signed in as {esc(user.name)} ({user.role})"
                if user.authenticated
                else f"Demo mode, no sign-in ({user.role})"
            )
        ]
        if manifest:
            footer.append(f"Run {esc(manifest.get('run_id', '–'))}")
        if MASK:
            footer.append("Financial values masked")
        st.markdown(f'<p class="note" style="margin-top:2rem">{"<br>".join(footer)}</p>', unsafe_allow_html=True)
    nav.run()


def render_page(key: str) -> None:
    """Render a single page without navigation (used by tests)."""
    setup()
    PAGES[key][1](current_user())
