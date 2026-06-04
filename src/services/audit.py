from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st


def initialize_session_state() -> None:
    st.session_state.setdefault("theme_mode", "Light")
    st.session_state.setdefault("audit_log", [])
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("selected_case_id", None)


def audit_event(action: str, detail: str, actor: str = "Systeme", metadata: dict[str, Any] | None = None) -> None:
    initialize_session_state()
    st.session_state.audit_log.insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "actor": actor,
            "action": action,
            "detail": detail,
            "metadata": metadata or {},
        },
    )
    st.session_state.audit_log = st.session_state.audit_log[:300]
