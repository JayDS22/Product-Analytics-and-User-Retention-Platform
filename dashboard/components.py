"""Reusable Streamlit UI components."""
from __future__ import annotations

from typing import Optional

import streamlit as st


def header(subtitle: str = "Product analytics and churn intelligence") -> None:
    st.markdown(
        f"""
        <div class="brand-row">
            <div class="brand-mark">RP</div>
            <div>
                <div class="brand-name">Retention Platform</div>
                <div class="brand-sub">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, delta: Optional[str] = None, kind: str = "neutral") -> None:
    delta_html = ""
    if delta:
        cls = {
            "positive": "metric-delta-positive",
            "negative": "metric-delta-negative",
            "neutral": "metric-delta-neutral",
        }.get(kind, "metric-delta-neutral")
        delta_html = f'<div class="{cls}">{delta}</div>'

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def tag(text: str, kind: str = "default") -> str:
    cls = {"good": "tag good", "warn": "tag warn", "danger": "tag danger"}.get(kind, "tag")
    return f'<span class="{cls}">{text}</span>'
