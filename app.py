"""Stockflow dashboard: forecasts, model performance, inventory health and transfer plan.

Read-only over the pipeline outputs. Run ``make run`` first, then ``streamlit run app.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import get_paths, load_config
from src.ui_theme import (ACCENT, ACCENT_LIGHT, CSS, GRID, INK, MUTED, SEQUENTIAL, STATUS, SUBTLE, money, pct,
                          register_plotly_template, units)

st.set_page_config(page_title="Stockflow", layout="wide", initial_sidebar_state="expanded")
register_plotly_template()
st.markdown(CSS, unsafe_allow_html=True)

# map outlines are served from ./static so the app never calls cdn.plot.ly
PLOT_CFG: dict[str, Any] = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                             "topojsonURL": "app/static/topojson/"}
HEALTH_LABEL = {"stockout_risk": "Stock-out risk", "excess": "Excess", "stagnant": "Stagnant", "healthy": "Healthy"}
REASON_LABEL = {"excess->shortfall": "Excess to shortfall", "stagnant->demand": "Stagnant to demand",
                "excess->demand": "Excess to live demand"}
MODEL_LABEL = {"lightgbm": "LightGBM (global)", "ses": "Exp. smoothing", "tsb": "TSB", "croston_sba": "Croston (SBA)",
               "moving_average": "28-day average", "seasonal_naive": "Seasonal naive", "selected": "Selected per segment"}


# --------------------------------------------------------------------------- data
@st.cache_resource
def _paths() -> Any:
    return get_paths(load_config())


@st.cache_data(show_spinner="Loading pipeline outputs…")
def load_outputs() -> dict[str, pd.DataFrame]:
    """All small pipeline outputs; raises FileNotFoundError if the pipeline has not run."""
    p = _paths()
    return {
        "forecasts": pd.read_csv(p.output / "forecasts.csv", parse_dates=["date"]),
        "policy": pd.read_csv(p.output / "inventory_policy.csv"),
        "transfers": pd.read_csv(p.output / "transfer_recommendations.csv"),
        "impact": pd.read_csv(p.reports / "optimizer_impact.csv", index_col="metric"),
        "overall": pd.read_csv(p.reports / "backtest_metrics_overall.csv"),
        "selection": pd.read_csv(p.reports / "model_selection.csv"),
        "by_abc": pd.read_csv(p.reports / "backtest_metrics_by_abc.csv"),
        "by_branch": pd.read_csv(p.reports / "backtest_metrics_selected_by_branch.csv"),
        "sel_abc": pd.read_csv(p.reports / "backtest_metrics_selected_by_abc.csv"),
        "coverage": pd.read_csv(p.reports / "interval_coverage.csv"),
        "branches": pd.read_csv(p.raw / "branches.csv"),
    }


@st.cache_data(show_spinner=False, max_entries=64)
def load_history(branch: str, sku: str) -> pd.DataFrame:
    """Daily cleaned history for one series (predicate push-down keeps this cheap)."""
    return pd.read_parquet(_paths().processed / "sales_clean.parquet",
                           columns=["date", "qty", "qty_raw", "is_stockout", "is_missing", "is_outlier"],
                           filters=[("branch_id", "==", branch), ("sku_id", "==", sku)])


def get_data() -> dict[str, pd.DataFrame]:
    try:
        return load_outputs()
    except FileNotFoundError:
        st.title("No pipeline outputs yet")
        st.markdown("The dashboard reads the files written by the batch pipeline. Generate them with:")
        st.code("make run   # or: python run_pipeline.py", language="bash")
        st.stop()
    except Exception:  # noqa: BLE001 - never show a traceback in the UI
        st.title("Could not load pipeline outputs")
        st.markdown("The output files exist but could not be read. Re-run the pipeline; if this persists, "
                    "check the pipeline log for the failing stage.")
        st.stop()


def lede(text: str, kicker: str | None = None) -> None:
    if kicker:
        st.markdown(f'<div class="kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="lede">{text}</p>', unsafe_allow_html=True)


def csv_button(df: pd.DataFrame, name: str, label: str = "Download CSV") -> None:
    st.download_button(label, df.to_csv(index=False).encode(), file_name=name, mime="text/csv")


# --------------------------------------------------------------------------- pages
def page_forecast() -> None:
    d = get_data()
    pol, fc = d["policy"], d["forecasts"]
    st.title("Forecast explorer")

    with st.sidebar:
        st.markdown("### Series")
        branches = sorted(pol["branch_id"].unique())
        b = st.selectbox("Branch", branches, index=0)
        skus = pol.loc[pol["branch_id"] == b].sort_values("forecast_28d", ascending=False)
        sku_labels = {r.sku_id: f"{r.sku_id} · {r.category} · class {r.abc_class}" for r in skus.itertuples()}
        s = st.selectbox("SKU (sorted by forecast volume)", list(sku_labels), format_func=sku_labels.get)
        days = st.slider("History shown (days)", 28, 365, 120, step=7)

    row = pol[(pol["branch_id"] == b) & (pol["sku_id"] == s)]
    f = fc[(fc["branch_id"] == b) & (fc["sku_id"] == s)]
    if row.empty or f.empty:
        st.info("No forecast for this branch and SKU. It may not be stocked at this branch.")
        return
    r = row.iloc[0]
    cover = "more than a year" if r["days_of_supply"] >= 365 else f"{r['days_of_supply']:.0f} days"
    status = HEALTH_LABEL.get(r["health"], r["health"])
    lede(f"{b} is expected to sell <b>{units(r['forecast_28d'])} units</b> of {s} over the next 28 days "
         f"(80% range {units(r['p10_28d_approx'])}–{units(r['p90_28d_approx'])}). "
         f"On-hand stock covers <b>{cover}</b>. Status: <b>{status}</b>.", kicker=f"{b} · {s}")

    try:
        h = load_history(b, s)
    except Exception:  # noqa: BLE001
        h = pd.DataFrame()
    h = h[h["date"] > h["date"].max() - pd.Timedelta(days=days)] if not h.empty else h

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(f["date"]) + list(f["date"][::-1]), y=list(f["p90"]) + list(f["p10"][::-1]),
                             fill="toself", fillcolor="rgba(11,110,105,0.14)", line=dict(width=0),
                             hoverinfo="skip", name="80% interval"))
    if not h.empty:
        fig.add_trace(go.Scatter(x=h["date"], y=h["qty"], mode="lines", line=dict(color=MUTED, width=1.2),
                                 name="Actual", hovertemplate="%{x|%a %d %b}: %{y:.0f} units<extra>Actual</extra>"))
        so = h[h["is_stockout"]]
        if len(so):
            fig.add_trace(go.Scatter(x=so["date"], y=so["qty"], mode="markers", name="Stock-out day",
                                     marker=dict(symbol="x", size=7, color=STATUS["risk"]),
                                     hovertemplate="%{x|%d %b}: stock-out, sales censored<extra></extra>"))
    fig.add_trace(go.Scatter(x=f["date"], y=f["forecast"], mode="lines", line=dict(color=ACCENT, width=2.2),
                             name="Forecast", hovertemplate="%{x|%a %d %b}: %{y:.1f} units<extra>Forecast</extra>"))
    if not h.empty:
        fig.add_vline(x=f["date"].min(), line=dict(color=GRID, dash="dot"))
        fig.add_annotation(x=f["date"].min(), y=1, yref="paper", text="forecast starts", showarrow=False,
                           xanchor="left", font=dict(size=11, color=MUTED))
    fig.add_annotation(x=f["date"].iloc[-1], y=f["p90"].iloc[-1], text="P90", showarrow=False, xanchor="left",
                       font=dict(size=11, color=ACCENT))
    fig.add_annotation(x=f["date"].iloc[-1], y=f["p10"].iloc[-1], text="P10", showarrow=False, xanchor="left",
                       font=dict(size=11, color=ACCENT))
    fig.update_layout(height=380, title=f"Daily units, {s} at {b}", yaxis_title="Units per day",
                      xaxis_title=None)
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    st.caption("Grey: cleaned daily sales. Teal line and band: forecast with its 10th–90th percentile range. "
               "Red crosses mark stock-out days, where sales understate demand.")

    st.markdown("## Replenishment policy")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("On hand", units(r["on_hand"]), help="Units at the branch at end of the last day of history")
    c2.metric("Reorder point", units(r["reorder_point"]),
              help="Lead-time demand plus safety stock. Reorder when stock falls below this.")
    c3.metric("Safety stock", units(r["safety_stock"]),
              help=f"Covers forecast error over the {int(r['lead_time_days'])}-day lead time at the target service level")
    c4.metric("Chance of stock-out in lead time", pct(r["p_stockout"], 0))
    st.markdown(
        f'<p class="note">Model: {MODEL_LABEL.get(r["model"], r["model"])} (chosen for segment '
        f'{r["segment"].replace("|", " · ")}). Lead time {int(r["lead_time_days"])} days; target maximum '
        f'{units(r["target_max"])} units; suggested order {units(r["recommended_order_qty"])} units.</p>',
        unsafe_allow_html=True)


def page_performance() -> None:
    d = get_data()
    ov, sel = d["overall"], d["selection"]
    st.title("Forecast accuracy")
    best = ov.sort_values("wape").iloc[0]
    naive = ov.loc[ov["model"] == "seasonal_naive"].iloc[0]
    worst = sel.sort_values("wape", ascending=False).iloc[0]
    lede(f"The selected models miss by <b>{pct(best['wape'], 0)}</b> of daily volume (WAPE), against "
         f"{pct(naive['wape'], 0)} for a seasonal-naive baseline: a <b>{pct(1 - best['wape'] / naive['wape'], 0)} "
         f"reduction</b>. Accuracy is weakest for {worst['abc_class']}-class {worst['pattern']} items "
         f"({pct(worst['wape'], 0)}), where daily demand is sparse.",
         kicker="4-fold rolling backtest · 28-day horizon")

    left, right = st.columns([1.1, 1])
    o = ov.sort_values("wape", ascending=True)
    fig = go.Figure(go.Bar(
        y=[MODEL_LABEL.get(m, m) for m in o["model"]], x=o["wape"] * 100, orientation="h",
        marker_color=[ACCENT if m == best["model"] else SUBTLE for m in o["model"]],
        text=[pct(v, 1) for v in o["wape"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: WAPE %{x:.1f}%<extra></extra>"))
    fig.update_layout(height=300, title="WAPE by model, all series (lower is better)",
                      xaxis_title="WAPE (% of actual units)", yaxis=dict(autorange="reversed"))
    left.plotly_chart(fig, width="stretch", config=PLOT_CFG)

    abc = d["by_abc"]
    piv = abc[abc["model"].isin(["seasonal_naive", "lightgbm"])].pivot(index="abc_class", columns="model", values="wape")
    fig2 = go.Figure()
    for m, col in [("seasonal_naive", SUBTLE), ("lightgbm", ACCENT)]:
        fig2.add_trace(go.Bar(x=[f"Class {c}" for c in piv.index], y=piv[m] * 100, marker_color=col,
                              name=MODEL_LABEL[m], text=[MODEL_LABEL[m].split(" ")[0] if i == 0 else "" for i in
                                                         range(len(piv))],
                              textposition="outside", hovertemplate="%{x}: %{y:.1f}%<extra>" + MODEL_LABEL[m] + "</extra>"))
    fig2.update_layout(barmode="group", height=300, title="WAPE by ABC class: LightGBM vs seasonal naive",
                       yaxis_title="WAPE (%)")
    right.plotly_chart(fig2, width="stretch", config=PLOT_CFG)

    st.markdown("## Model chosen for each segment")
    t = sel.assign(
        Segment=sel["abc_class"] + " · " + sel["pattern"],
        Model=sel["model"].map(MODEL_LABEL),
        **{"Series-folds": sel["series"], "WAPE": sel["wape"] * 100, "Naive WAPE": sel["seasonal_naive_wape"] * 100,
           "Improvement": sel["improvement_vs_naive"] * 100, "Bias": sel["bias"] * 100, "MASE": sel["mase"]})
    t = t[["Segment", "Model", "Series-folds", "WAPE", "Naive WAPE", "Improvement", "Bias", "MASE"]]
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        "Segment": st.column_config.TextColumn(pinned=True),
        "WAPE": st.column_config.NumberColumn(format="%.1f%%"),
        "Naive WAPE": st.column_config.NumberColumn(format="%.1f%%"),
        "Improvement": st.column_config.NumberColumn("vs naive", format="%.0f%%"),
        "Bias": st.column_config.NumberColumn(format="%+.1f%%", help="Positive = over-forecast"),
        "MASE": st.column_config.NumberColumn(format="%.2f"),
    })
    csv_button(t, "model_selection.csv")

    st.markdown("## Bias by branch")
    bb = d["by_branch"].sort_values("bias")
    fig3 = go.Figure(go.Scatter(
        x=bb["bias"] * 100, y=bb["branch_id"], mode="markers",
        marker=dict(size=9, color=[STATUS["risk"] if abs(v) > 0.05 else ACCENT for v in bb["bias"]],
                    symbol=["diamond" if abs(v) > 0.05 else "circle" for v in bb["bias"]]),
        hovertemplate="%{y}: bias %{x:+.1f}%<extra></extra>"))
    fig3.add_vrect(x0=-5, x1=5, fillcolor=SURFACE_RGBA, line_width=0)
    fig3.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig3.update_layout(height=520, title="Forecast bias by branch, selected models (band = ±5%)",
                       xaxis_title="Bias (% of actual units; + means over-forecast)")
    st.plotly_chart(fig3, width="stretch", config=PLOT_CFG)
    cov = d["coverage"]
    allc = cov.loc[cov["segment"] == "ALL", "p10_p90_coverage"].iloc[0]
    st.markdown(f'<p class="note">Prediction intervals: the P10–P90 band contained {pct(allc, 0)} of actual daily '
                f'sales on the held-out fold (target 80%).</p>', unsafe_allow_html=True)


SURFACE_RGBA = "rgba(245,247,249,1)"


def page_health() -> None:
    d = get_data()
    pol = d["policy"]
    st.title("Inventory health")
    risk = pol[pol["stockout_risk"]]
    n_br = risk["branch_id"].nunique()
    lede(f"<b>{len(risk):,} branch × SKU positions</b> at {n_br} branches are likely to stock out before "
         f"replenishment arrives. Meanwhile <b>{money(pol['excess_value'].sum())}</b> sits above target levels "
         f"and <b>{money(pol['stagnant_value'].sum())}</b> has not sold in 90+ days.",
         kicker="Position as of the last day of history")

    c1, c2, c3 = st.columns(3)
    c1.metric("Positions at stock-out risk", f"{len(risk):,}")
    c2.metric("Excess above target", money(pol["excess_value"].sum()))
    c3.metric("Stagnant (90+ days unsold)", money(pol["stagnant_value"].sum()))

    metric = st.radio("Show", ["Stock-out risk positions", "Excess $", "Stagnant $", "Days of supply (median)"],
                      horizontal=True)
    agg = {"Stock-out risk positions": ("stockout_risk", "sum"), "Excess $": ("excess_value", "sum"),
           "Stagnant $": ("stagnant_value", "sum"), "Days of supply (median)": ("days_of_supply", "median")}[metric]
    grid = pol.pivot_table(index="branch_id", columns="category", values=agg[0], aggfunc=agg[1]).fillna(0)
    if metric.startswith("Days"):
        grid = grid.clip(upper=365)
    grid = grid.loc[grid.sum(axis=1).sort_values(ascending=False).index]
    fmt = (lambda v: f"{v:.0f}") if not metric.endswith("$") else (lambda v: money(v, 0) if v else "")
    text = [[fmt(v) if v else "" for v in row] for row in grid.to_numpy()]
    fig = go.Figure(go.Heatmap(
        z=grid.to_numpy(), x=grid.columns, y=grid.index, colorscale=SEQUENTIAL, text=text,
        texttemplate="%{text}", textfont=dict(size=11), xgap=2, ygap=2,
        colorbar=dict(title=dict(text=metric, side="right"), thickness=10),
        hovertemplate="%{y} · %{x}: %{text}<extra></extra>"))
    fig.update_layout(height=max(420, 24 * len(grid)), title=f"{metric} by branch and category",
                      xaxis=dict(side="top", title=None), yaxis=dict(autorange="reversed", title=None))
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)

    st.markdown("## Positions needing action")
    view = pol[pol["health"] != "healthy"].copy()
    view["Status"] = view["health"].map(HEALTH_LABEL)
    t = view.rename(columns={"branch_id": "Branch", "sku_id": "SKU", "category": "Category", "abc_class": "ABC",
                             "on_hand": "On hand", "days_of_supply": "Days of supply",
                             "safety_stock": "Safety stock", "reorder_point": "Reorder point",
                             "p_stockout": "Stock-out probability", "excess_value": "Excess $",
                             "stagnant_value": "Stagnant $"})
    cols = ["Branch", "SKU", "Status", "Category", "ABC", "On hand", "Days of supply", "Safety stock",
            "Reorder point", "Stock-out probability", "Excess $", "Stagnant $"]
    status_pick = st.multiselect("Status", list(HEALTH_LABEL.values())[:3], default=["Stock-out risk"])
    t = t[t["Status"].isin(status_pick)][cols].sort_values(["Stock-out probability", "Excess $"], ascending=False)
    if t.empty:
        st.info("No positions match the selected statuses.")
    else:
        st.dataframe(t, hide_index=True, width="stretch", height=420, column_config={
            "Branch": st.column_config.TextColumn(pinned=True), "SKU": st.column_config.TextColumn(pinned=True),
            "On hand": st.column_config.NumberColumn(format="%d"),
            "Days of supply": st.column_config.NumberColumn(format="%.0f"),
            "Safety stock": st.column_config.NumberColumn(format="%.1f"),
            "Reorder point": st.column_config.NumberColumn(format="%.1f"),
            "Stock-out probability": st.column_config.NumberColumn(format="%.0f%%"),
            "Excess $": st.column_config.NumberColumn(format="$%,.0f"),
            "Stagnant $": st.column_config.NumberColumn(format="$%,.0f"),
        })
        csv_button(t, "inventory_positions.csv")


def page_transfers() -> None:
    d = get_data()
    tr, imp, br = d["transfers"], d["impact"], d["branches"]
    st.title("Transfer recommendations")
    o = imp["optimizer"]
    g = imp["greedy"]
    lede(f"Moving <b>{units(o['units_moved'])} units</b> in {int(o['transfers']):,} transfers costs "
         f"{money(o['transfer_cost'])} and saves a net <b>{money(o['net_savings'])}</b>: it covers "
         f"{units(o['stockout_units_avoided'])} units of expected lost sales and frees {money(o['excess_value_freed'])} "
         f"of excess. That is {money(o['net_savings'] - g['net_savings'])} more than a greedy rule-of-thumb plan.",
         kicker="Recommendations only · nothing moves until a planner approves")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net savings", money(o["net_savings"]), f"{money(o['net_savings'] - g['net_savings'])} vs greedy")
    c2.metric("Expected lost units avoided", units(o["stockout_units_avoided"]))
    c3.metric("Excess freed", money(o["excess_value_freed"]))
    c4.metric("Inventory turns", f"{o['inventory_turns_after']:.1f}",
              f"{o['inventory_turns_after'] - o['inventory_turns_before']:+.1f} from {o['inventory_turns_before']:.1f}")

    with st.sidebar:
        st.markdown("### Filter transfers")
        f_from = st.multiselect("From branch", sorted(tr["from_branch"].unique()))
        f_to = st.multiselect("To branch", sorted(tr["to_branch"].unique()))
        f_cat = st.multiselect("Category", sorted(tr["category"].unique()))
        f_reason = st.multiselect("Reason", list(REASON_LABEL.values()))
        min_sav = st.number_input("Minimum net savings ($)", min_value=0, value=0, step=50)
    v = tr.copy()
    v["Reason"] = v["reason_code"].map(REASON_LABEL)
    if f_from:
        v = v[v["from_branch"].isin(f_from)]
    if f_to:
        v = v[v["to_branch"].isin(f_to)]
    if f_cat:
        v = v[v["category"].isin(f_cat)]
    if f_reason:
        v = v[v["Reason"].isin(f_reason)]
    v = v[v["projected_savings"] >= min_sav]

    st.markdown("## Flows between branches")
    if v.empty:
        st.info("No transfers match the filters.")
        return
    flows = v.groupby(["from_branch", "to_branch"], as_index=False).agg(units=("qty", "sum"),
                                                                         savings=("projected_savings", "sum"))
    loc = br.set_index("branch_id")
    fig = go.Figure()
    mx = flows["units"].max()
    for r in flows.itertuples():
        a, b = loc.loc[r.from_branch], loc.loc[r.to_branch]
        fig.add_trace(go.Scattergeo(lon=[a.lon, b.lon], lat=[a.lat, b.lat], mode="lines",
                                    line=dict(width=0.8 + 5 * r.units / mx, color="rgba(11,110,105,0.55)"),
                                    hoverinfo="text",
                                    text=f"{r.from_branch} → {r.to_branch}: {r.units:,.0f} units, {money(r.savings)} net"))
    send = v.groupby("from_branch")["qty"].sum()
    recv = v.groupby("to_branch")["qty"].sum()
    nodes = loc.assign(sent=send.reindex(loc.index).fillna(0), received=recv.reindex(loc.index).fillna(0))
    fig.add_trace(go.Scattergeo(
        lon=nodes["lon"], lat=nodes["lat"], mode="markers+text", text=nodes.index, textposition="top center",
        textfont=dict(size=10, color=INK),
        marker=dict(size=8, color=[ACCENT if s > r else "white" for s, r in zip(nodes["sent"], nodes["received"])],
                    line=dict(color=ACCENT, width=1.5)),
        hovertext=[f"{i} ({n.branch_name}): sends {n.sent:,.0f}, receives {n.received:,.0f} units"
                   for i, n in nodes.iterrows()], hoverinfo="text"))
    fig.update_geos(scope="usa", projection_type="albers usa", showland=True, landcolor="#f5f7f9",
                    subunitcolor="#d9dee3", countrycolor="#d9dee3", showlakes=False)
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=40, b=0),
                      title="Units moved between branches (line width = units; filled dot = net sender)")
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)

    st.markdown(f"## {len(v):,} recommended transfers")
    t = v.rename(columns={"from_branch": "From", "to_branch": "To", "sku_id": "SKU", "category": "Category",
                          "qty": "Units", "value_moved": "Value moved", "transit_days": "Transit days",
                          "transfer_cost": "Transfer cost", "projected_savings": "Net savings",
                          "from_days_of_supply": "Sender days of supply", "to_days_of_supply": "Receiver days of supply"})
    cols = ["From", "To", "SKU", "Reason", "Category", "Units", "Value moved", "Transfer cost", "Net savings",
            "Transit days", "Sender days of supply", "Receiver days of supply"]
    st.dataframe(t[cols], hide_index=True, width="stretch", height=440, column_config={
        "From": st.column_config.TextColumn(pinned=True), "To": st.column_config.TextColumn(pinned=True),
        "SKU": st.column_config.TextColumn(pinned=True),
        "Units": st.column_config.NumberColumn(format="%d"),
        "Value moved": st.column_config.NumberColumn(format="$%,.0f"),
        "Transfer cost": st.column_config.NumberColumn(format="$%,.0f"),
        "Net savings": st.column_config.NumberColumn(format="$%,.0f"),
        "Sender days of supply": st.column_config.NumberColumn(format="%.0f"),
        "Receiver days of supply": st.column_config.NumberColumn(format="%.0f"),
    })
    csv_button(t[cols], "transfer_recommendations.csv")

    st.markdown("## Optimizer vs greedy baseline")
    rows = [("Net savings", "net_savings", money), ("Transfer cost", "transfer_cost", money),
            ("Units moved", "units_moved", units), ("Expected lost units avoided", "stockout_units_avoided", units),
            ("Excess freed", "excess_value_freed", money),
            ("Positions still at risk", "series_at_risk_after", lambda x: f"{x:,.0f}"),
            ("Inventory turns after", "inventory_turns_after", lambda x: f"{x:.2f}")]
    st.dataframe(pd.DataFrame([(lbl, f(o[k]), f(g[k])) for lbl, k, f in rows],
                              columns=["Measure", "Optimizer", "Greedy"]), hide_index=True, width="content")


pages = [st.Page(page_transfers, title="Transfer recommendations", url_path="transfers", default=True),
         st.Page(page_health, title="Inventory health", url_path="health"),
         st.Page(page_forecast, title="Forecast explorer", url_path="forecast"),
         st.Page(page_performance, title="Forecast accuracy", url_path="accuracy")]
st.navigation(pages).run()
