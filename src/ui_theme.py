"""Single source of truth for dashboard colours, chart styling and number formats."""

from __future__ import annotations

import math

import plotly.graph_objects as go
import plotly.io as pio

# neutrals + one accent; red / amber / green are reserved for status only
INK = "#1f2933"
MUTED = "#616e7c"  # secondary text, 5.2:1 on white (WCAG AA)
SUBTLE = "#838d99"  # de-emphasised marks only, never text; 3.4:1 on white (WCAG 1.4.11)
GRID = "#e4e7eb"
SURFACE = "#f5f7f9"
ACCENT = "#0b6e69"  # 6.1:1 on white
ACCENT_LIGHT = "#a7d3cf"
STATUS = {"risk": "#b42318", "warn": "#b54708", "ok": "#2f7d32"}

# sequential scale for heatmaps (white -> accent); status colour is never used here
SEQUENTIAL = [[0.0, "#ffffff"], [0.15, "#e3f1ef"], [0.5, "#7fbdb7"], [1.0, ACCENT]]


def register_plotly_template() -> None:
    """Register and activate the 'stockflow' Plotly template."""
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family="Source Sans Pro, Source Sans 3, system-ui, sans-serif", size=13, color=INK),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=[ACCENT, MUTED, SUBTLE, INK],
        xaxis=dict(
            showgrid=False,
            linecolor=GRID,
            ticks="outside",
            tickcolor=GRID,
            zeroline=False,
            title=dict(font=dict(size=12, color=MUTED)),
        ),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, title=dict(font=dict(size=12, color=MUTED))),
        margin=dict(l=8, r=8, t=48, b=8),
        hoverlabel=dict(bgcolor="white", font_size=12),
        showlegend=False,
    )
    pio.templates["stockflow"] = t
    pio.templates.default = "stockflow"


def money(x: float, decimals: int = 1) -> str:
    """$1.2M / $340k / $950."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x / 1e6:.{decimals}f}M"
    if x >= 1e4:
        return f"{sign}${x / 1e3:.0f}k"
    if x >= 1e3:
        return f"{sign}${x / 1e3:.{decimals}f}k"
    return f"{sign}${x:,.0f}"


def pct(x: float, decimals: int = 1) -> str:
    """0.043 -> '4.3%'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x * 100:.{decimals}f}%"


def units(x: float) -> str:
    """12,345 units, compact above 100k."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x / 1e3:,.0f}k" if abs(x) >= 1e5 else f"{x:,.0f}"


CSS = f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
  h1 {{ font-size: 1.65rem !important; font-weight: 650 !important; letter-spacing: -0.01em; }}
  h2 {{ font-size: 1.15rem !important; font-weight: 600 !important; margin-top: 1.6rem !important; }}
  h3 {{ font-size: 1.0rem !important; font-weight: 600 !important; }}
  .lede {{ font-size: 1.05rem; color: {INK}; margin: -0.4rem 0 1.2rem 0; max-width: 62rem; }}
  .lede b {{ color: {ACCENT}; }}
  .kicker {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: {MUTED};
             margin-bottom: 0.1rem; }}
  .note {{ font-size: 0.85rem; color: {MUTED}; }}
  [data-testid="stMetricValue"] {{ font-size: 1.45rem; }}
  [data-testid="stMetricLabel"] p {{ color: {MUTED}; }}
</style>
"""
