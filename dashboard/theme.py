"""Theme palette and chart defaults shared across pages."""
from __future__ import annotations

import plotly.graph_objects as go

PALETTE = {
    "bg": "#0B1220",
    "panel": "#111A2E",
    "card": "#16213D",
    "border": "#1F2A48",
    "text": "#E6EDF7",
    "muted": "#8696B8",
    "primary": "#6EA8FF",
    "primary_soft": "#3A6EE0",
    "accent": "#7CF4D4",
    "warn": "#FFB454",
    "danger": "#F26D6D",
    "good": "#5CD08F",
    "grid": "rgba(255,255,255,0.05)",
}

SEQUENCE = ["#6EA8FF", "#7CF4D4", "#FFB454", "#F26D6D", "#B894FF", "#5CD08F", "#FF8DC7", "#F0E167"]


def base_layout(title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(color=PALETTE["text"], size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], family="Inter, system-ui, sans-serif"),
        margin=dict(l=20, r=20, t=50, b=30),
        xaxis=dict(
            gridcolor=PALETTE["grid"],
            zerolinecolor=PALETTE["grid"],
            linecolor=PALETTE["border"],
            tickfont=dict(color=PALETTE["muted"]),
            title_font=dict(color=PALETTE["muted"]),
        ),
        yaxis=dict(
            gridcolor=PALETTE["grid"],
            zerolinecolor=PALETTE["grid"],
            linecolor=PALETTE["border"],
            tickfont=dict(color=PALETTE["muted"]),
            title_font=dict(color=PALETTE["muted"]),
        ),
        legend=dict(font=dict(color=PALETTE["muted"])),
        hoverlabel=dict(bgcolor=PALETTE["panel"], bordercolor=PALETTE["border"], font_color=PALETTE["text"]),
    )


def styled_figure(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(**base_layout(title))
    return fig


CSS = """
<style>
:root {
    --bg: #0B1220;
    --panel: #111A2E;
    --card: #16213D;
    --border: #1F2A48;
    --text: #E6EDF7;
    --muted: #8696B8;
    --primary: #6EA8FF;
    --accent: #7CF4D4;
    --good: #5CD08F;
    --warn: #FFB454;
    --danger: #F26D6D;
}
html, body, .stApp {
    background: radial-gradient(circle at 0% 0%, #14213d 0%, #0B1220 55%) fixed;
    color: var(--text);
}
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text); }
h1, h2, h3, h4, h5 { color: var(--text); letter-spacing: -0.01em; }
p, span, label, li, div { color: var(--text); }
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    height: 100%;
}
.metric-label {
    font-size: 12px;
    font-weight: 500;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
}
.metric-delta-positive { color: var(--good); font-size: 13px; margin-top: 6px; }
.metric-delta-negative { color: var(--danger); font-size: 13px; margin-top: 6px; }
.metric-delta-neutral { color: var(--muted); font-size: 13px; margin-top: 6px; }
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin: 18px 0 8px;
    color: var(--text);
}
.section-sub {
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 18px;
}
.tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    background: rgba(110, 168, 255, 0.15);
    color: var(--primary);
    border: 1px solid rgba(110, 168, 255, 0.3);
}
.tag.good { background: rgba(92,208,143,0.12); color: var(--good); border-color: rgba(92,208,143,0.3); }
.tag.warn { background: rgba(255,180,84,0.12); color: var(--warn); border-color: rgba(255,180,84,0.3); }
.tag.danger { background: rgba(242,109,109,0.14); color: var(--danger); border-color: rgba(242,109,109,0.3); }
.brand-row {
    display: flex; align-items: center; gap: 12px;
    padding: 6px 0 18px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 18px;
}
.brand-mark {
    width: 36px; height: 36px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; color: #0B1220;
}
.brand-name { font-size: 18px; font-weight: 700; }
.brand-sub  { font-size: 12px; color: var(--muted); }
.stDataFrame, .stTable { border-radius: 12px; overflow: hidden; }
.stButton > button {
    background: var(--primary);
    color: #0B1220;
    border: none;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 600;
}
.stButton > button:hover { background: #8FBDFF; color: #0B1220; }
div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--text);
}
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: var(--card);
    border-radius: 10px 10px 0 0;
    padding: 8px 16px;
    color: var(--muted);
}
.stTabs [aria-selected="true"] {
    background: var(--panel);
    color: var(--text);
    border-bottom: 2px solid var(--primary);
}
</style>
"""
