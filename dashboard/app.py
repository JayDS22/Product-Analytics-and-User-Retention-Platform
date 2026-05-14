"""Streamlit demo platform entrypoint.

Run from the repo root:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.cohort import cohort_matrix, cohort_revenue
from src.analytics.funnel import funnel_conversion, journey_paths
from src.analytics.metrics import dau, dau_mau_ratio, mau, wau
from src.analytics.retention import nth_day_retention, retention_curve
from src.config import (
    EVENTS_FILE,
    FEATURES_FILE,
    METRICS_FILE,
    MODEL_FILE,
    USERS_FILE,
)
from src.data.synthetic_generator import generate_dataset
from src.models.evaluator import calibration_table, evaluate, lift_table
from src.models.glm import ChurnGLM
from src.models.train import train_pipeline

from dashboard.components import header, metric_card, section, tag
from dashboard.theme import CSS, PALETTE, SEQUENCE, styled_figure


st.set_page_config(
    page_title="Retention Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.data.loader import load_events, load_users

    return load_events(), load_users()


@st.cache_data(show_spinner=False)
def _load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_FILE)


@st.cache_resource(show_spinner=False)
def _load_model() -> ChurnGLM:
    return ChurnGLM.load(MODEL_FILE)


@st.cache_data(show_spinner=False)
def _load_metrics() -> dict:
    return json.loads(METRICS_FILE.read_text())


def _ensure_artifacts() -> bool:
    return all(p.exists() for p in (EVENTS_FILE, USERS_FILE, FEATURES_FILE, MODEL_FILE, METRICS_FILE))


def _bootstrap_panel() -> None:
    header()
    section(
        "Initialize the platform",
        "Generate synthetic event data, engineer 120+ user-level features, and train the churn GLM. "
        "This runs once and persists artifacts under data/ and artifacts/.",
    )
    cols = st.columns([1, 1, 2])
    with cols[0]:
        n_users = st.number_input("Users", min_value=1000, max_value=30000, value=6000, step=1000)
    with cols[1]:
        regenerate = st.toggle("Regenerate", value=False)
    with cols[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Build pipeline", use_container_width=True)

    if run:
        with st.status("Building pipeline", expanded=True) as status:
            st.write("Generating synthetic events")
            generate_dataset(n_users=int(n_users))
            st.write("Engineering features and training model")
            result = train_pipeline()
            st.write(f"Trained on {result.metrics['training_rows']:,} users with {result.metrics['feature_count']} features")
            status.update(label="Pipeline ready", state="complete")
        _load_dataset.clear()
        _load_features.clear()
        _load_model.clear()
        _load_metrics.clear()
        st.rerun()


def _format_int(n: int | float) -> str:
    return f"{int(n):,}"


def _format_pct(p: float, digits: int = 1) -> str:
    return f"{p * 100:.{digits}f}%"


def _format_money(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v/1_000:.1f}k"
    return f"${v:,.0f}"


def page_overview(events: pd.DataFrame, users: pd.DataFrame, metrics: dict) -> None:
    section("Overview", "Top-line health of the product across users, activity, and revenue.")

    total_users = users["user_id"].nunique()
    total_events = len(events)
    total_revenue = float(events["revenue"].sum())
    last_30 = events[events["event_time"] >= events["event_time"].max() - pd.Timedelta(days=30)]
    prev_30 = events[
        (events["event_time"] >= events["event_time"].max() - pd.Timedelta(days=60))
        & (events["event_time"] < events["event_time"].max() - pd.Timedelta(days=30))
    ]
    active_30 = last_30["user_id"].nunique()
    active_prev = prev_30["user_id"].nunique()
    delta = (active_30 - active_prev) / max(active_prev, 1)
    rev_30 = float(last_30["revenue"].sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total users", _format_int(total_users))
    with c2:
        metric_card("Events ingested", _format_int(total_events))
    with c3:
        kind = "positive" if delta >= 0 else "negative"
        metric_card("Active users (30d)", _format_int(active_30), f"{delta*100:+.1f}% vs prior 30d", kind=kind)
    with c4:
        metric_card("Revenue", _format_money(total_revenue), f"{_format_money(rev_30)} last 30d")

    section("Active users", "Daily, weekly, and monthly distinct users.")
    dau_df = dau(events)
    wau_df = wau(events)
    mau_df = mau(events)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dau_df["date"], y=dau_df["dau"], name="DAU", line=dict(color=SEQUENCE[0], width=2)))
    fig.add_trace(go.Scatter(x=wau_df["week"], y=wau_df["wau"], name="WAU", line=dict(color=SEQUENCE[1], width=2)))
    fig.add_trace(go.Scatter(x=mau_df["month"], y=mau_df["mau"], name="MAU", line=dict(color=SEQUENCE[2], width=2)))
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(styled_figure(fig), use_container_width=True)

    section("Stickiness", "DAU divided by 30-day rolling unique users.")
    stick = dau_mau_ratio(events).dropna()
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=stick["date"],
            y=stick["stickiness"],
            line=dict(color=SEQUENCE[0], width=2),
            fill="tozeroy",
            fillcolor="rgba(110,168,255,0.12)",
            name="Stickiness",
        )
    )
    fig2.update_yaxes(tickformat=".0%")
    st.plotly_chart(styled_figure(fig2), use_container_width=True)

    section("Plan mix and revenue share")
    c1, c2 = st.columns(2)
    plan_users = users.groupby("plan")["user_id"].nunique().reindex(["free", "starter", "pro", "enterprise"])
    plan_rev = events.groupby("plan")["revenue"].sum().reindex(["free", "starter", "pro", "enterprise"]).fillna(0)
    with c1:
        fig3 = go.Figure(go.Bar(x=plan_users.index, y=plan_users.values, marker_color=SEQUENCE[:4]))
        st.plotly_chart(styled_figure(fig3, "Users by plan"), use_container_width=True)
    with c2:
        fig4 = go.Figure(go.Bar(x=plan_rev.index, y=plan_rev.values, marker_color=SEQUENCE[:4]))
        st.plotly_chart(styled_figure(fig4, "Revenue by plan"), use_container_width=True)


def page_retention(events: pd.DataFrame, users: pd.DataFrame) -> None:
    section("Retention curves", "Probability of returning N days after signup.")
    max_day = st.slider("Window (days)", 7, 60, 30, step=1)
    curve = retention_curve(events, users, max_day=max_day)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["day"],
            y=curve["retained_rate"],
            mode="lines+markers",
            line=dict(color=SEQUENCE[0], width=2.5),
            marker=dict(color=SEQUENCE[0], size=6),
            fill="tozeroy",
            fillcolor="rgba(110,168,255,0.12)",
        )
    )
    fig.update_yaxes(tickformat=".0%", title_text="Retained")
    fig.update_xaxes(title_text="Day since signup")
    st.plotly_chart(styled_figure(fig), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, n in zip((c1, c2, c3, c4), (1, 7, 14, 30)):
        if n <= max_day:
            value = nth_day_retention(events, users, n)
            with col:
                metric_card(f"D{n} retention", _format_pct(value))

    section("Cohort retention heatmap", "Weekly signup cohorts and their return rate over time.")
    matrix = cohort_matrix(events, users, max_period=10, freq="W")
    if not matrix.empty:
        fig2 = px.imshow(
            matrix.values * 100,
            labels=dict(x="Weeks after signup", y="Signup week", color="Retained %"),
            x=matrix.columns,
            y=matrix.index,
            color_continuous_scale=[
                [0.0, "#0B1220"],
                [0.5, "#3A6EE0"],
                [1.0, "#7CF4D4"],
            ],
            aspect="auto",
        )
        fig2.update_layout(coloraxis_colorbar=dict(tickformat=".0f"))
        st.plotly_chart(styled_figure(fig2), use_container_width=True)


def page_churn(features: pd.DataFrame, model: ChurnGLM, metrics: dict, events: pd.DataFrame, users: pd.DataFrame) -> None:
    section("Churn model", "Logistic GLM trained on engineered user-level features.")

    test = metrics["test"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Accuracy", _format_pct(test["accuracy"], 2))
    with c2:
        metric_card("ROC AUC", f"{test['roc_auc']:.3f}")
    with c3:
        metric_card("Precision", _format_pct(test["precision"], 2))
    with c4:
        metric_card("Recall", _format_pct(test["recall"], 2))

    proba = model.predict_proba(features.drop(columns=["user_id"]))
    scored = features[["user_id"]].copy()
    scored["churn_probability"] = proba

    plan_lookup = users.set_index("user_id")["plan"]
    scored["plan"] = scored["user_id"].map(plan_lookup)

    section("Risk distribution", "Population-level distribution of predicted churn probability.")
    fig = go.Figure(go.Histogram(x=proba, nbinsx=40, marker_color=SEQUENCE[0], opacity=0.85))
    fig.update_xaxes(title_text="Predicted churn probability", tickformat=".0%")
    fig.update_yaxes(title_text="Users")
    st.plotly_chart(styled_figure(fig), use_container_width=True)

    section("Decile lift", "Targeting effectiveness by descending predicted risk.")
    from src.features.engineering import FeatureSpec, label_churn
    from src.models.train import _select_reference_date

    ref = _select_reference_date(events)
    spec = FeatureSpec(reference_date=ref)
    labels = label_churn(events, users, spec).reindex(features["user_id"]).fillna(1).astype(int)
    lift = lift_table(labels.values, proba)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=lift["decile"], y=lift["lift"], marker_color=SEQUENCE[0], name="Lift"))
    fig2.add_hline(y=1.0, line_dash="dash", line_color=PALETTE["muted"])
    fig2.update_xaxes(title_text="Decile (1 = highest risk)")
    fig2.update_yaxes(title_text="Lift over base rate")
    st.plotly_chart(styled_figure(fig2), use_container_width=True)

    section("Calibration", "Predicted vs observed positive rate by probability bucket.")
    calib = calibration_table(labels.values, proba)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=PALETTE["muted"]), name="Ideal"))
    fig3.add_trace(
        go.Scatter(
            x=calib["predicted"],
            y=calib["observed"],
            mode="lines+markers",
            line=dict(color=SEQUENCE[1], width=2.5),
            marker=dict(size=8, color=SEQUENCE[1]),
            name="Model",
        )
    )
    fig3.update_xaxes(title_text="Predicted", tickformat=".0%")
    fig3.update_yaxes(title_text="Observed", tickformat=".0%")
    st.plotly_chart(styled_figure(fig3), use_container_width=True)

    section("Top feature drivers", "Largest absolute standardized coefficients in the GLM.")
    importance = model.feature_importance.head(20).iloc[::-1]
    colors = [SEQUENCE[1] if c > 0 else SEQUENCE[3] for c in importance["coefficient"]]
    fig4 = go.Figure(go.Bar(x=importance["coefficient"], y=importance["feature"], orientation="h", marker_color=colors))
    fig4.update_xaxes(title_text="Standardized coefficient")
    fig4.update_layout(height=520)
    st.plotly_chart(styled_figure(fig4), use_container_width=True)

    section("At-risk users", "Highest predicted churn probabilities. Useful as a retention worklist.")
    top_n = st.slider("Show top N", 10, 200, 50, step=10)
    top = scored.sort_values("churn_probability", ascending=False).head(top_n)
    st.dataframe(
        top.assign(churn_probability=top["churn_probability"].map("{:.3f}".format)),
        use_container_width=True,
        hide_index=True,
    )


def page_funnels(events: pd.DataFrame) -> None:
    section("Funnel analysis", "Step-by-step conversion through a defined event sequence.")
    default_steps = ["session_start", "page_view", "add_to_cart", "checkout", "purchase"]
    available = sorted(events["event_type"].unique().tolist())
    steps = st.multiselect("Define funnel steps in order", available, default=default_steps)
    if len(steps) < 2:
        st.info("Select at least two steps to render the funnel.")
        return

    funnel = funnel_conversion(events, steps)
    fig = go.Figure(
        go.Funnel(
            y=funnel["step"],
            x=funnel["users"],
            textinfo="value+percent initial",
            marker=dict(color=SEQUENCE[: len(funnel)]),
            connector=dict(line=dict(color=PALETTE["border"])),
        )
    )
    st.plotly_chart(styled_figure(fig), use_container_width=True)

    section("Step conversion rates")
    st.dataframe(
        funnel.assign(
            share_of_total=funnel["share_of_total"].map("{:.1%}".format),
            step_conversion=funnel["step_conversion"].map("{:.1%}".format),
        ),
        use_container_width=True,
        hide_index=True,
    )

    section("Common user journeys", "Most frequent event prefixes after the first interaction.")
    paths = journey_paths(events, max_steps=5, top_n=15)
    fig2 = go.Figure(
        go.Bar(
            x=paths["users"],
            y=paths["path"],
            orientation="h",
            marker_color=SEQUENCE[0],
        )
    )
    fig2.update_layout(height=520, yaxis=dict(autorange="reversed"))
    fig2.update_xaxes(title_text="Users")
    st.plotly_chart(styled_figure(fig2), use_container_width=True)


def page_cohorts(events: pd.DataFrame, users: pd.DataFrame) -> None:
    section("Cohort revenue", "Cumulative average revenue per user by signup cohort.")
    matrix = cohort_revenue(events, users, max_period=10, freq="W")
    if matrix.empty:
        st.info("Not enough revenue data to build cohort matrix.")
        return
    fig = px.imshow(
        matrix.values,
        labels=dict(x="Weeks after signup", y="Signup week", color="ARPU"),
        x=matrix.columns,
        y=matrix.index,
        color_continuous_scale=[
            [0.0, "#0B1220"],
            [0.5, "#FFB454"],
            [1.0, "#7CF4D4"],
        ],
        aspect="auto",
    )
    st.plotly_chart(styled_figure(fig), use_container_width=True)
    section("Raw cohort table")
    st.dataframe(matrix.round(2), use_container_width=True)


def page_dataset(events: pd.DataFrame, users: pd.DataFrame, features: pd.DataFrame) -> None:
    section("Dataset summary", "Schema and sample rows powering this demo.")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Users", _format_int(len(users)))
    with c2:
        metric_card("Events", _format_int(len(events)))
    with c3:
        metric_card("Features per user", _format_int(features.shape[1] - 1))

    tab_e, tab_u, tab_f = st.tabs(["Events", "Users", "Features"])
    with tab_e:
        st.dataframe(events.head(500), use_container_width=True, hide_index=True)
    with tab_u:
        st.dataframe(users.head(500), use_container_width=True, hide_index=True)
    with tab_f:
        st.dataframe(features.head(500), use_container_width=True, hide_index=True)


def main() -> None:
    if not _ensure_artifacts():
        _bootstrap_panel()
        return

    events, users = _load_dataset()
    features = _load_features()
    model = _load_model()
    metrics = _load_metrics()

    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand-row">
                <div class="brand-mark">RP</div>
                <div>
                    <div class="brand-name">Retention Platform</div>
                    <div class="brand-sub">v0.1.0</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation",
            ["Overview", "Retention", "Churn", "Funnels", "Cohorts", "Dataset"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Pipeline**")
        st.caption(f"Reference date: {metrics.get('reference_date', '-')[:10]}")
        st.caption(f"Features: {metrics['feature_count']}")
        st.caption(f"Train rows: {metrics['training_rows']:,}")
        st.caption(f"Test ROC AUC: {metrics['test']['roc_auc']:.3f}")
        st.caption(f"Test accuracy: {metrics['test']['accuracy']:.3f}")
        st.divider()
        if st.button("Rebuild pipeline", use_container_width=True):
            with st.status("Retraining", expanded=True) as status:
                st.write("Training model on cached events")
                train_pipeline()
                status.update(label="Done", state="complete")
            _load_features.clear()
            _load_model.clear()
            _load_metrics.clear()
            st.rerun()

    header()

    if page == "Overview":
        page_overview(events, users, metrics)
    elif page == "Retention":
        page_retention(events, users)
    elif page == "Churn":
        page_churn(features, model, metrics, events, users)
    elif page == "Funnels":
        page_funnels(events)
    elif page == "Cohorts":
        page_cohorts(events, users)
    elif page == "Dataset":
        page_dataset(events, users, features)


if __name__ == "__main__":
    main()
