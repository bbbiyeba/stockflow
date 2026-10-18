"""Charts and facts for the Stryker deck, computed only from pipeline outputs.

Writes presentation/assets/*.png and presentation/assets/facts.json. Every number used on a
slide comes from facts.json, which is built here from files under outputs/, reports/ and
data/processed/ (run `make run` first).
"""

from __future__ import annotations

import json
import os
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from PIL import Image, ImageChops  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "assets"
sys.path.insert(0, str(ROOT))

# Stryker palette (gold per Brandfetch / BrandColorsHub; neutrals per the brief)
GOLD, BLACK, CHARCOAL, GRAY, LIGHT, LIGHTER = "#FFB502", "#000000", "#333333", "#6E6E6E", "#D9D9D9", "#F2F2F2"

plt.rcParams.update(
    {
        "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 13,
        "axes.edgecolor": LIGHT,
        "axes.labelcolor": CHARCOAL,
        "xtick.color": CHARCOAL,
        "ytick.color": CHARCOAL,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    }
)


def money(x: float) -> str:
    """Format dollars as $1.2M, or $45k below a million."""
    return f"${x / 1e6:.1f}M" if abs(x) >= 1e6 else f"${x / 1e3:.0f}k"


def facts() -> dict:
    """Collect every number the deck quotes from the pipeline outputs into assets/facts.json."""
    pol = pd.read_parquet(ROOT / "outputs/inventory_policy.parquet")
    imp = pd.read_csv(ROOT / "reports/optimizer_impact.csv", index_col="metric")
    ho = pd.read_csv(ROOT / "reports/holdout_metrics_overall.csv").set_index("model")
    sens = pd.read_csv(ROOT / "reports/sensitivity.csv")
    cov = pd.read_csv(ROOT / "reports/interval_coverage.csv").set_index("segment")
    man = json.loads((ROOT / "outputs/run_manifest.json").read_text())
    so = pd.read_parquet(ROOT / "data/processed/sales_clean", columns=["is_stockout"])["is_stockout"]
    o, g = imp["optimizer"], imp["greedy"]
    by = pol.groupby("health")["on_hand_value"].agg(["size", "sum"])
    f = {
        "run_id": man["run_id"],
        "git_commit": man.get("git_commit", "")[:8],
        "branches": int(pol["branch_id"].nunique()),
        "skus": int(pol["sku_id"].nunique()),
        "series": len(pol),
        "on_hand_value": float(pol["on_hand_value"].sum()),
        "excess_value": float(pol["excess_value"].sum()),
        "stagnant_value": float(pol["stagnant_value"].sum()),
        "liquidation_value": float(pol.loc[pol["liquidation_candidate"], "on_hand_value"].sum()),
        "liquidation_positions": int(pol["liquidation_candidate"].sum()),
        "risk_positions": int(pol["stockout_risk"].sum()),
        "stockout_day_rate": float(so.mean()),
        "health": {k: {"positions": int(r["size"]), "value": float(r["sum"])} for k, r in by.iterrows()},
        "transfers": int(o["transfers"]),
        "units_moved": float(o["units_moved"]),
        "transfer_cost": float(o["transfer_cost"]),
        "net_savings": float(o["net_savings"]),
        "greedy_net_savings": float(g["net_savings"]),
        "excess_before": float(o["excess_value_before"]),
        "excess_after": float(o["excess_value_after"]),
        "excess_freed": float(o["excess_value_freed"]),
        "carry_before": float(o["annual_carrying_cost_excess_before"]),
        "carry_after": float(o["annual_carrying_cost_excess_after"]),
        "stockout_units_before": float(o["expected_stockout_units_before"]),
        "stockout_units_after": float(o["expected_stockout_units_after"]),
        "stockout_units_avoided": float(o["stockout_units_avoided"]),
        "risk_after": int(o["series_at_risk_after"]),
        "risk_after_greedy": int(g["series_at_risk_after"]),
        "turns_before": float(o["inventory_turns_before"]),
        "turns_after": float(o["inventory_turns_after"]),
        "stagnant_before": float(o["stagnant_value_before"]),
        "stagnant_after": float(o["stagnant_value_after"]),
        "models": {
            m: {k: float(ho.loc[m, k]) for k in ("wape", "mae", "bias", "mase", "wape_28d_total")} for m in ho.index
        },
        "coverage": float(cov.loc["ALL", "p10_p90_coverage"]),
        "sens_overlap_min": float(sens["plan_overlap"].min()),
        "sens_overlap_max": float(sens.loc[sens["scenario"] != "Base case", "plan_overlap"].max()),
        "sens_savings_min": float(sens["net_savings"].min()),
        "sens_savings_max": float(sens["net_savings"].max()),
    }
    f["stockout_avoided_share"] = f["stockout_units_avoided"] / f["stockout_units_before"]
    f["excess_share"] = f["excess_value"] / f["on_hand_value"]
    return f


def chart_problem(f: dict) -> None:
    """Bar chart of on-hand value by position status."""
    h = f["health"]
    rows = [
        ("Overstocked", "excess"),
        ("Stagnant 90+ days", "stagnant"),
        ("Healthy", "healthy"),
        ("At stock-out risk", "stockout_risk"),
    ]
    vals = [h[k]["value"] / 1e6 for _, k in rows]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    colors = [CHARCOAL, GRAY, LIGHT, GOLD]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=colors, height=0.62)
    for yi, v, (_, k) in zip(y, vals, rows, strict=True):
        ax.text(v + 0.12, yi, f"${v:.1f}M  ·  {h[k]['positions']:,} positions", va="center", fontsize=12, color=BLACK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=12)
    ax.set_xlabel("Inventory on hand ($ millions, at cost)")
    ax.set_xlim(0, max(vals) * 1.75)
    ax.tick_params(axis="y", length=0)
    fig.savefig(ASSETS / "problem.png")
    plt.close(fig)


def chart_forecast(f: dict) -> dict:
    """Held-out forecast vs actual with the P10-P90 band for a typical A-class series."""
    ho = pd.read_parquet(ROOT / "reports/holdout_forecasts.parquet")
    pol = pd.read_parquet(ROOT / "outputs/inventory_policy.parquet")[["branch_id", "sku_id", "abc_class", "category"]]
    ho = ho.merge(pol, on=["branch_id", "sku_id"])
    g = ho.dropna(subset=["actual"]).groupby(["branch_id", "sku_id", "abc_class", "category"], observed=True)
    s = g.apply(
        lambda d: pd.Series(
            {
                "wape": (d["forecast"] - d["actual"]).abs().sum() / max(d["actual"].sum(), 1),
                "units": d["actual"].sum(),
                "n": len(d),
            }
        ),
        include_groups=False,
    ).reset_index()
    # representative = an A-class series with full scoring whose WAPE is closest to the A-class median
    a = s[(s["abc_class"] == "A") & (s["n"] == 28) & (s["units"] >= 300)]
    med = a["wape"].median()
    pick = a.iloc[(a["wape"] - med).abs().argsort().iloc[0]]
    d = ho[(ho["branch_id"] == pick["branch_id"]) & (ho["sku_id"] == pick["sku_id"])].sort_values("date")
    hist = pd.read_parquet(
        ROOT / "data/processed/sales_clean",
        columns=["date", "qty"],
        filters=[("branch_id", "==", pick["branch_id"]), ("sku_id", "==", pick["sku_id"])],
    )
    hist = hist[(hist["date"] < d["date"].min()) & (hist["date"] >= d["date"].min() - pd.Timedelta(days=28))]
    act = pd.read_parquet(
        ROOT / "data/processed/sales_clean",
        columns=["date", "qty"],
        filters=[("branch_id", "==", pick["branch_id"]), ("sku_id", "==", pick["sku_id"])],
    )
    act = act[act["date"].isin(d["date"])]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.fill_between(d["date"], d["p10"], d["p90"], color=GOLD, alpha=0.28, linewidth=0)
    ax.plot(hist["date"], hist["qty"], color=GRAY, lw=1.4)
    ax.plot(act["date"], act["qty"], color=BLACK, lw=1.6)
    ax.plot(d["date"], d["forecast"], color=GOLD, lw=2.4)
    ax.axvline(d["date"].min(), color=LIGHT, lw=1, ls="--")
    end = d["date"].max()
    ax.annotate(
        "Actual",
        (act["date"].iloc[-1], act["qty"].iloc[-1]),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=11,
        color=BLACK,
        fontweight="bold",
    )
    ax.annotate(
        "Forecast",
        (end, d["forecast"].iloc[-1]),
        xytext=(6, -14),
        textcoords="offset points",
        fontsize=11,
        color="#8A6100",
        fontweight="bold",
    )
    top = float(max(d["p90"].max(), act["qty"].max()))
    ax.text(d["date"].iloc[1], top * 1.08, "Shaded: P10–P90 range", fontsize=10, color=CHARCOAL, va="bottom")
    ax.text(
        d["date"].min(),
        ax.get_ylim()[1] * 0.97,
        "  held-out month (model never saw it)",
        fontsize=10,
        color=GRAY,
        va="top",
    )
    ax.set_ylabel("Units per day")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    ax.set_xlim(hist["date"].min(), end + pd.Timedelta(days=6))
    fig.savefig(ASSETS / "forecast.png")
    plt.close(fig)
    return {
        "branch_id": pick["branch_id"],
        "sku_id": pick["sku_id"],
        "category": pick["category"],
        "series_wape": float(pick["wape"]),
        "a_class_median_wape": float(med),
        "start": str(d["date"].min().date()),
        "end": str(end.date()),
        "series_coverage": float(((d["actual"] >= d["p10"]) & (d["actual"] <= d["p90"]))[d["actual"].notna()].mean()),
    }


def _render_html(fig: go.Figure, name: str, width: int, height: int) -> None:
    """Render a Plotly geo figure to PNG with headless Chromium (map outlines served locally)."""
    from playwright.sync_api import sync_playwright

    html = ASSETS / f"_{name}.html"
    fig.write_html(
        html, include_plotlyjs=True, full_html=True, config={"staticPlot": True, "topojsonURL": "/static/topojson/"}
    )
    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            # a pre-installed Chromium (e.g. PLAYWRIGHT_BROWSERS_PATH) if present, else `playwright install`'s
            exe = next(
                Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")).glob(
                    "chromium-*/chrome-linux/chrome"
                ),
                None,
            )
            b = p.chromium.launch(executable_path=str(exe)) if exe else p.chromium.launch()
            page = b.new_page(viewport={"width": width, "height": height}, device_scale_factor=2)
            page.goto(f"http://127.0.0.1:{srv.server_address[1]}/presentation/assets/_{name}.html")
            page.wait_for_timeout(2500)
            page.screenshot(path=str(ASSETS / f"{name}.png"))

            im = Image.open(ASSETS / f"{name}.png").convert("RGB")  # trim white margins
            bbox = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
            if bbox:
                pad = 20
                im.crop(
                    (
                        max(bbox[0] - pad, 0),
                        max(bbox[1] - pad, 0),
                        min(bbox[2] + pad, im.width),
                        min(bbox[3] + pad, im.height),
                    )
                ).save(ASSETS / f"{name}.png")
            b.close()
    finally:
        srv.shutdown()
        html.unlink()


def _geo(fig: go.Figure) -> None:
    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor=LIGHTER,
        subunitcolor="#C8C8C8",
        countrycolor="#C8C8C8",
        showlakes=False,
        bgcolor="white",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        showlegend=False,
        font=dict(family="Arial, Liberation Sans, sans-serif", size=13, color=BLACK),
    )


def chart_map(f: dict, top: int = 20) -> dict:
    """Map of the largest transfer corridors."""
    tr = pd.read_parquet(ROOT / "outputs/transfer_recommendations.parquet")
    br = pd.read_csv(ROOT / "data/raw/branches.csv").set_index("branch_id")
    flows = tr.groupby(["from_branch", "to_branch"], as_index=False).agg(
        units=("qty", "sum"), savings=("projected_savings", "sum")
    )
    shown = flows.nlargest(top, "units")
    fig = go.Figure()
    mx = shown["units"].max()
    for r in shown.itertuples():
        a, b = br.loc[r.from_branch], br.loc[r.to_branch]
        fig.add_trace(
            go.Scattergeo(
                lon=[a.lon, b.lon],
                lat=[a.lat, b.lat],
                mode="lines",
                line=dict(width=1.2 + 7 * r.units / mx, color="rgba(51,51,51,0.72)"),
            )
        )
    sent = shown.groupby("from_branch")["units"].sum()
    recv = shown.groupby("to_branch")["units"].sum()
    act = sorted(set(shown["from_branch"]) | set(shown["to_branch"]))
    n = br.loc[act]
    is_sender = sent.reindex(act).fillna(0).to_numpy() > recv.reindex(act).fillna(0).to_numpy()
    fig.add_trace(
        go.Scattergeo(
            lon=n["lon"],
            lat=n["lat"],
            mode="markers+text",
            text=list(n.index),
            textposition="top center",
            textfont=dict(size=12, color=BLACK),
            marker=dict(size=13, color=np.where(is_sender, BLACK, GOLD), line=dict(color=BLACK, width=1.4)),
        )
    )
    _geo(fig)
    _render_html(fig, "transfer_map", 900, 560)
    return {
        "corridors_shown": int(len(shown)),
        "corridors_total": int(len(flows)),
        "share_units_shown": float(shown["units"].sum() / flows["units"].sum()),
    }


def chart_title_map() -> None:
    """Map of branch locations for the title slide."""
    br = pd.read_csv(ROOT / "data/raw/branches.csv")
    fig = go.Figure(
        go.Scattergeo(
            lon=br["lon"],
            lat=br["lat"],
            mode="markers",
            marker=dict(size=11, color=GOLD, line=dict(color=BLACK, width=1.2)),
        )
    )
    _geo(fig)
    _render_html(fig, "branch_map", 900, 560)


def top_transfers(n: int = 5) -> list[dict]:
    """The highest-saving recommended transfers."""
    tr = pd.read_parquet(ROOT / "outputs/transfer_recommendations.parquet").nlargest(n, "projected_savings")
    return [
        {
            "from": r.from_branch,
            "to": r.to_branch,
            "sku": r.sku_id,
            "category": r.category,
            "units": int(r.qty),
            "cost": float(r.transfer_cost),
            "savings": float(r.projected_savings),
            "reason": r.reason_code,
            "reason_text": r.reason,
        }
        for r in tr.itertuples()
    ]


# crop boxes (left, top, right, bottom) in docs/screenshots/transfers.png, taken at a 1366 x 1700 viewport
DASH_TOP_BOX = (360, 60, 1300, 1066)  # headline, metrics and corridor map
DASH_TABLE_BOX = (370, 1241, 880, 1576)  # first rows of the transfer table with Decide / Status / Reason


def dashboard_crops() -> None:
    """Crop the slide 7 images from the dashboard screenshot (retake it with the README's screenshot steps)."""
    im = Image.open(ROOT / "docs/screenshots/transfers.png").convert("RGB")
    im.crop(DASH_TOP_BOX).save(ASSETS / "dash_top.png")
    im.crop(DASH_TABLE_BOX).save(ASSETS / "dash_table.png")


def main() -> None:
    """Write facts.json and every chart image."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    f = facts()
    chart_problem(f)
    f["forecast_example"] = chart_forecast(f)
    f["map"] = chart_map(f)
    chart_title_map()
    f["top_transfers"] = top_transfers()
    dashboard_crops()
    (ASSETS / "facts.json").write_text(json.dumps(f, indent=2, default=str))
    print(json.dumps({k: v for k, v in f.items() if k not in ("top_transfers",)}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
