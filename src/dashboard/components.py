"""Reusable presentation components."""

import pandas as pd
import streamlit as st


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="eyebrow">CYBERGUARD · RISK INTELLIGENCE</div><h1>{title}</h1>', unsafe_allow_html=True)
    st.caption(subtitle)


def metric_row(metrics: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value, detail) in zip(columns, metrics):
        column.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div><div class="metric-detail">{detail}</div></div>',
            unsafe_allow_html=True,
        )


def empty_state(message: str, detail: str | None = None) -> None:
    st.info(message)
    if detail:
        st.caption(detail)


def format_probability(value) -> str:
    return "—" if pd.isna(value) else f"{float(value):.2%}"


def format_risk(value) -> str:
    return "—" if pd.isna(value) else f"{float(value):,.3f}"
