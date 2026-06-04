from __future__ import annotations

import os

import streamlit as st

from src.config import APP_TITLE, GEMINI_MODEL, NAVIGATION
from src.data_factory import filter_cases, make_fraud_cases
from src.pages.enterprise import render_page
from src.services.audit import audit_event, initialize_session_state
from src.ui.enterprise import inject_enterprise_theme, render_sidebar


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_session_state()
    inject_enterprise_theme(st.session_state.theme_mode)

    cases = make_fraud_cases()
    sidebar_state = render_sidebar(
        cases=cases,
        navigation=NAVIGATION,
        gemini_model=GEMINI_MODEL,
        default_api_key=os.getenv("GEMINI_API_KEY", ""),
    )
    filtered_cases = filter_cases(
        cases,
        regions=sidebar_state["regions"],
        risk_levels=sidebar_state["risk_levels"],
        statuses=sidebar_state["statuses"],
        date_range=sidebar_state["date_range"],
    )

    audit_event(
        "Navigation",
        f"Ouverture page {sidebar_state['page']}",
        actor="Analyste fraude",
        metadata={"rows": int(len(filtered_cases))},
    )
    render_page(
        page=sidebar_state["page"],
        cases=filtered_cases,
        all_cases=cases,
        gemini_api_key=sidebar_state["api_key"],
    )


if __name__ == "__main__":
    main()
