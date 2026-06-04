from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import PALETTE, RISK_COLORS
from src.data_factory import case_feature_frame
from src.services.audit import audit_event
from src.services.explainability import contribution_table, importance_figure, shap_contribution_table, waterfall_figure
from src.services.forecast import forecast_fraud
from src.services.gemini_service import analyze_case, chat, fallback_investigation
from src.services.ml_service import compare_anomaly_models, load_model_metrics, load_random_forest_model
from src.services.report_service import build_pdf_report
from src.ui.enterprise import apply_chart_theme, kpi_card, panel, render_hero, risk_badge


def _selected_case(cases: pd.DataFrame) -> pd.Series:
    if not len(cases):
        st.stop()
    ids = cases.sort_values("risk_score", ascending=False)["case_id"].tolist()
    current = st.session_state.get("selected_case_id")
    index = ids.index(current) if current in ids else 0
    case_id = st.selectbox("Dossier", ids, index=index)
    st.session_state.selected_case_id = case_id
    return cases.loc[cases["case_id"] == case_id].iloc[0]


def _download_excel(df: pd.DataFrame) -> bytes | None:
    try:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="FraudDetect")
        return buffer.getvalue()
    except Exception:
        return None


def home_page(cases: pd.DataFrame) -> None:
    render_hero(
        "FraudDetect AI",
        "Plateforme intelligente de detection et d'investigation des fraudes dans les services publics.",
        ["IA", "Random Forest", "Gemini AI", "GovTech"],
    )
    frauds = cases[cases["is_fraud"]]
    cols = st.columns(6)
    with cols[0]:
        kpi_card("Dossiers analyses", f"{len(cases):,}".replace(",", " "), "Portefeuille courant", "D")
    with cols[1]:
        kpi_card("Fraudes detectees", f"{len(frauds):,}".replace(",", " "), "Signalements positifs", "!", PALETTE["danger"])
    with cols[2]:
        kpi_card("Montant suspect", f"{frauds['amount'].sum()/1_000_000:.2f} M€", "Exposition estimee", "€", PALETTE["warning"])
    with cols[3]:
        kpi_card("Risque moyen", f"{cases['risk_score'].mean():.0f}/100", "Score portefeuille", "R", PALETTE["accent"])
    with cols[4]:
        kpi_card("Agents actifs", "12", "Cellule investigation", "A", PALETTE["success"])
    with cols[5]:
        kpi_card("Temps moyen", f"{cases['processing_hours'].mean():.1f}h", "Traitement estime", "T", "#7C3AED")

    left, right = st.columns([0.62, 0.38], gap="large")
    with left:
        weekly = cases.groupby("week", as_index=False).agg(fraudes=("is_fraud", "sum"), montant=("amount", "sum"))
        fig = px.area(weekly, x="week", y="fraudes", title="Activite fraude recente", markers=True)
        fig.update_traces(line_color=PALETTE["accent"], fillcolor="rgba(58,134,255,.18)")
        st.plotly_chart(apply_chart_theme(fig, 360), use_container_width=True)
    with right:
        st.markdown("### Derniers incidents critiques")
        st.dataframe(
            cases.sort_values("risk_score", ascending=False).head(8)[["case_id", "risk_level", "region", "amount", "status"]],
            use_container_width=True,
            hide_index=True,
        )


def dashboard_page(cases: pd.DataFrame) -> None:
    render_hero("Dashboard executif", "Pilotage temps reel des risques, pertes potentielles et performances du dispositif anti-fraude.", ["Executive", "Plotly", "Risk Intelligence"])
    frauds = cases[cases["is_fraud"]]
    critical = cases[cases["risk_level"] == "Critique"]
    kpis = st.columns(6)
    values = [
        ("Dossiers analyses", f"{len(cases):,}".replace(",", " "), "Volume filtre", "D", PALETTE["accent"]),
        ("Fraudes detectees", f"{len(frauds):,}".replace(",", " "), "Cas suspects", "!", PALETTE["danger"]),
        ("Montant suspect", f"{frauds['amount'].sum()/1_000_000:.2f} M€", "Exposition", "€", PALETTE["warning"]),
        ("Taux de fraude", f"{len(frauds)/max(len(cases),1):.1%}", "Taux detecte", "%", "#7C3AED"),
        ("Temps moyen", f"{cases['processing_hours'].mean():.1f}h", "Traitement", "T", PALETTE["success"]),
        ("Cas critiques", f"{len(critical):,}".replace(",", " "), "Priorite max", "C", PALETTE["danger"]),
    ]
    for col, item in zip(kpis, values):
        with col:
            kpi_card(*item)

    weekly = cases.groupby("week", as_index=False).agg(fraudes=("is_fraud", "sum"), montant=("amount", "sum"), risque=("risk_score", "mean"))
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = px.line(weekly, x="week", y="fraudes", title="Evolution des fraudes dans le temps", markers=True)
        fig.update_traces(line=dict(color=PALETTE["danger"], width=3))
        st.plotly_chart(apply_chart_theme(fig), use_container_width=True)
    with c2:
        fig = px.bar(weekly, x="week", y="montant", title="Evolution du montant frauduleux")
        fig.update_traces(marker_color=PALETTE["warning"])
        st.plotly_chart(apply_chart_theme(fig), use_container_width=True)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        fig = px.pie(cases, names="fraud_type", values="amount", hole=.55, title="Repartition par type de fraude")
        st.plotly_chart(apply_chart_theme(fig), use_container_width=True)
    with c4:
        region = cases.groupby("region", as_index=False).agg(fraudes=("is_fraud", "sum"), montant=("amount", "sum"))
        fig = px.bar(region.sort_values("fraudes"), x="fraudes", y="region", orientation="h", title="Repartition par region")
        fig.update_traces(marker_color=PALETTE["accent"])
        st.plotly_chart(apply_chart_theme(fig), use_container_width=True)

    c5, c6 = st.columns(2, gap="large")
    with c5:
        hour = cases.groupby("hour", as_index=False).agg(dossiers=("case_id", "count"), fraudes=("is_fraud", "sum"))
        fig = px.bar(hour, x="hour", y="fraudes", title="Repartition par heure de depot")
        fig.update_traces(marker_color="#7C3AED")
        st.plotly_chart(apply_chart_theme(fig), use_container_width=True)
    with c6:
        fig = px.bar(cases, x="user_profile", color="risk_level", title="Repartition par profil utilisateur", color_discrete_map=RISK_COLORS)
        st.plotly_chart(apply_chart_theme(fig), use_container_width=True)

    c7, c8 = st.columns(2, gap="large")
    with c7:
        heat = pd.crosstab(cases["day_name"], cases["hour"], values=cases["is_fraud"].astype(int), aggfunc="sum").fillna(0)
        fig = px.imshow(heat, aspect="auto", title="Heatmap Fraude x Heure x Jour", color_continuous_scale=["#E0F2FE", "#3A86FF", "#E63946"])
        st.plotly_chart(apply_chart_theme(fig), use_container_width=True)
    with c8:
        top = cases.sort_values("risk_score", ascending=False).head(10).sort_values("risk_score")
        fig = px.bar(top, x="risk_score", y="case_id", orientation="h", title="Top 10 dossiers a risque", color="risk_score", color_continuous_scale=["#FFBE0B", "#E63946"])
        st.plotly_chart(apply_chart_theme(fig), use_container_width=True)

    c9, c10 = st.columns([0.36, 0.64], gap="large")
    with c9:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=float(cases["risk_score"].mean()), title={"text": "Gauge du risque global"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": PALETTE["danger"]}}))
        st.plotly_chart(apply_chart_theme(fig, 330), use_container_width=True)
    with c10:
        metrics = load_model_metrics()
        if {"metric", "value"}.issubset(metrics.columns):
            fig = px.line(metrics, x="metric", y="value", markers=True, title="Courbe des performances du modele")
        else:
            fig = px.line(weekly, x="week", y="risque", markers=True, title="Courbe des performances du modele")
        fig.update_traces(line=dict(color=PALETTE["success"], width=3))
        st.plotly_chart(apply_chart_theme(fig, 330), use_container_width=True)


def alerts_page(cases: pd.DataFrame) -> None:
    render_hero("Centre d'alertes", "Priorisation dynamique des dossiers selon criticite, montant, score et statut.", ["Alerting", "SLA", "Human Review"])
    level = st.segmented_control("Niveau", ["Tous", "Critique", "Eleve", "Moyen", "Faible"], default="Tous")
    search = st.text_input("Recherche dossier, region ou type")
    data = cases.copy()
    if level != "Tous":
        data = data[data["risk_level"] == level]
    if search:
        q = search.lower()
        data = data[data.astype(str).apply(lambda row: q in " ".join(row).lower(), axis=1)]
    display = data.sort_values("risk_score", ascending=False)[["submitted_at", "case_id", "risk_level", "risk_score", "amount", "region", "fraud_type", "status"]]
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button("Exporter CSV", display.to_csv(index=False).encode("utf-8"), "alertes_frauddetect.csv", "text/csv")


def ai_analysis_page(cases: pd.DataFrame, api_key: str) -> None:
    render_hero("Analyse IA", "Agent IA enqueteur avec Gemini, fallback local, sortie JSON structuree et controle humain.", ["Gemini Investigator", "JSON", "Fallback"])
    case = _selected_case(cases)
    if st.button("Analyser le dossier"):
        st.session_state.last_ai_case = case["case_id"]
        st.session_state.last_ai_result = analyze_case(api_key, case)
        audit_event("Prediction IA", f"Analyse IA du dossier {case['case_id']}", "IA", {"score": int(case["risk_score"])})
    result = st.session_state.get("last_ai_result")
    if not result or st.session_state.get("last_ai_case") != case["case_id"]:
        result = analyze_case("", case)
    if result.error:
        st.warning(f"Gemini indisponible, fallback automatique utilise. Detail: {result.error}")
    st.caption(f"Fournisseur: {result.provider}")
    c1, c2 = st.columns([0.42, 0.58], gap="large")
    with c1:
        kpi_card("Score risque", f"{case['risk_score']}/100", str(case["risk_level"]), "R", RISK_COLORS.get(case["risk_level"], PALETTE["accent"]))
        st.json(result.data)
    with c2:
        panel("Resume executif", result.data.get("resume", "-"))
        st.markdown("### Anomalies")
        for anomaly in result.data.get("anomalies", []):
            st.markdown(f"- {anomaly}")
        st.markdown("### Actions recommandees")
        for action in result.data.get("actions", []):
            st.markdown(f"- {action}")
        st.info(f"Decision proposee: {result.data.get('decision', '-')}")
    st.markdown("### Chat IA interne")
    question = st.chat_input("Pose une question sur ce dossier")
    if question:
        st.session_state.chat_history.append({"role": "analyste", "content": question})
        answer = chat(api_key, case, st.session_state.chat_history, question)
        st.session_state.chat_history.append({"role": "assistant", "content": answer.data["answer"]})
        audit_event("Chat IA", f"Question sur {case['case_id']}", "Analyste fraude")
    for msg in st.session_state.chat_history[-8:]:
        with st.chat_message("user" if msg["role"] == "analyste" else "assistant"):
            st.write(msg["content"])


def investigation_page(cases: pd.DataFrame) -> None:
    render_hero("Pourquoi ce dossier est suspect ?", "Explicabilite IA avec SHAP lorsque disponible, contributions positives et negatives, waterfall et top facteurs de risque.", ["SHAP", "Explainability", "Audit"])
    case = _selected_case(cases)
    c1, c2 = st.columns([0.45, 0.55], gap="large")
    contrib, provider = shap_contribution_table(cases, case)
    with c1:
        st.markdown(f"### {case['case_id']} {risk_badge(case['risk_level'])}", unsafe_allow_html=True)
        st.caption(provider)
        st.dataframe(contrib, use_container_width=True, hide_index=True)
        st.markdown("### Top facteurs de risque")
        for _, row in contrib.head(5).iterrows():
            sign = "+" if float(row["contribution"]) >= 0 else ""
            st.markdown(f"- {row['facteur']} : {sign}{row['contribution']:.0f}%")
    with c2:
        st.plotly_chart(apply_chart_theme(waterfall_figure(case, contrib), 430), use_container_width=True)
    st.plotly_chart(apply_chart_theme(importance_figure(cases), 380), use_container_width=True)


def portfolio_page(cases: pd.DataFrame) -> None:
    render_hero("Portefeuille dossiers", "Table enterprise avec recherche, filtres, tri natif et exports CSV / Excel.", ["DataTable", "Export", "Operations"])
    search = st.text_input("Recherche globale")
    data = cases.copy()
    if search:
        q = search.lower()
        data = data[data.astype(str).apply(lambda row: q in " ".join(row).lower(), axis=1)]
    cols = ["case_id", "submitted_at", "risk_level", "risk_score", "amount", "region", "service", "fraud_type", "user_profile", "status"]
    st.dataframe(data[cols], use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Exporter CSV", data[cols].to_csv(index=False).encode("utf-8"), "portefeuille_frauddetect.csv", "text/csv")
    with c2:
        excel = _download_excel(data[cols])
        if excel:
            st.download_button("Exporter Excel", excel, "portefeuille_frauddetect.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.caption("Export Excel indisponible: installe openpyxl.")


def monitoring_page(cases: pd.DataFrame) -> None:
    render_hero("Monitoring modeles", "Comparaison Random Forest, Isolation Forest, LOF et consensus operationnel.", ["Random Forest", "Isolation Forest", "LOF"])
    comp = compare_anomaly_models(cases)
    model = load_random_forest_model()
    st.success("Random Forest charge depuis models/fraudai_model.joblib" if model else "Random Forest indisponible: fallback score metier utilise")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = px.scatter(comp, x="random_forest", y="isolation_forest", color="decision", size="amount", hover_name="case_id", title="Random Forest vs Isolation Forest")
        st.plotly_chart(apply_chart_theme(fig), use_container_width=True)
    with c2:
        fig = px.histogram(comp, x="consensus", color="decision", nbins=30, title="Distribution du consensus")
        st.plotly_chart(apply_chart_theme(fig), use_container_width=True)
    st.dataframe(comp.head(30), use_container_width=True, hide_index=True)


def forecast_page(cases: pd.DataFrame) -> None:
    render_hero("Prevision", "Projection du volume de fraudes, pertes potentielles et cas critiques attendus.", ["Forecast", "ARIMA fallback", "Planning"])
    future = forecast_fraud(cases)
    if future.empty:
        st.warning("Pas assez de donnees pour generer une prevision.")
        return
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = px.line(future, x="day", y="fraudes_prevues", markers=True, title="Volume de fraudes futur")
        st.plotly_chart(apply_chart_theme(fig), use_container_width=True)
    with c2:
        fig = px.area(future, x="day", y="pertes_potentielles", title="Pertes potentielles")
        fig.update_traces(line_color=PALETTE["danger"], fillcolor="rgba(230,57,70,.18)")
        st.plotly_chart(apply_chart_theme(fig), use_container_width=True)
    st.dataframe(future, use_container_width=True, hide_index=True)


def map_page(cases: pd.DataFrame) -> None:
    render_hero("Carte fraude", "Concentration geographique des fraudes, montants suspects et clusters regionaux.", ["Plotly Map", "Geo Risk", "Regions"])
    fig = px.scatter_mapbox(
        cases,
        lat="lat",
        lon="lon",
        color="risk_level",
        size="amount",
        hover_name="case_id",
        hover_data=["region", "fraud_type", "risk_score", "amount"],
        color_discrete_map=RISK_COLORS,
        zoom=4.6,
        height=620,
        mapbox_style="carto-positron",
        title="Concentration des fraudes par territoire",
    )
    st.plotly_chart(apply_chart_theme(fig, 650), use_container_width=True)


def reports_page(cases: pd.DataFrame, api_key: str) -> None:
    render_hero("Rapports", "Generation de rapports PDF professionnels pour comite, controle interne ou superviseur.", ["PDF", "Investigation", "Export"])
    case = _selected_case(cases)
    analysis = analyze_case(api_key, case).data
    contrib = contribution_table(case)
    st.json(analysis)
    pdf = build_pdf_report(case, analysis, contrib)
    st.download_button("Exporter le rapport", pdf, f"rapport_{case['case_id']}.pdf", "application/pdf")
    audit_event("Export", f"Rapport genere pour {case['case_id']}", "Analyste fraude")


def audit_page() -> None:
    render_hero("Journal d'audit", "Timeline visuelle des actions utilisateur, predictions IA, validations et exports.", ["Auditability", "Traceability", "Compliance"])
    for item in st.session_state.audit_log[:80]:
        st.markdown(
            f"""
            <div class="timeline-item">
                <div class="timeline-time">{item['time']} - {item['actor']} → {item['action']}</div>
                <div class="timeline-detail">{item['detail']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def responsible_ai_page(cases: pd.DataFrame) -> None:
    render_hero("IA Responsable", "Cadre de confiance: human-in-the-loop, transparence, explicabilite, auditabilite, RGPD et non-discrimination.", ["Governance", "RGPD", "Human Control"])
    items = [
        ("Human in the Loop", "Aucune decision punitive ou financiere definitive n'est automatisee."),
        ("Transparence", "Chaque score est accompagne d'une justification et d'un journal d'audit."),
        ("Explicabilite", "Les facteurs de risque sont visibles via contributions et waterfall."),
        ("Auditabilite", "Navigation, predictions, validations et exports sont traces en session."),
        ("Protection des donnees", "Cette demonstration utilise des donnees simulees et minimise les donnees sensibles."),
        ("RGPD", "Les usages doivent reposer sur une base legale, une finalite claire et une conservation limitee."),
        ("Non-discrimination", "Les variables sensibles doivent etre exclues ou controlees par tests de biais."),
    ]
    cols = st.columns(2, gap="large")
    for i, (title, body) in enumerate(items):
        with cols[i % 2]:
            panel(title, body)
    fairness = pd.DataFrame({"profil": cases["user_profile"].unique()})
    fairness["taux_controle"] = fairness["profil"].map(lambda p: float((cases.loc[cases["user_profile"] == p, "risk_level"].isin(["Eleve", "Critique"])).mean()))
    fig = px.bar(fairness, x="profil", y="taux_controle", title="Indicateur de controle par profil utilisateur")
    st.plotly_chart(apply_chart_theme(fig), use_container_width=True)


def admin_page(cases: pd.DataFrame) -> None:
    render_hero("Administration", "Etat systeme, session, modeles, donnees et controles operationnels.", ["Ops", "System", "Settings"])
    c1, c2, c3 = st.columns(3)
    with c1:
        panel("Statut ML", "Random Forest conserve comme modele principal. Les modeles d'anomalies servent au consensus.")
    with c2:
        panel("Statut donnees", f"{len(cases)} dossiers disponibles dans le portefeuille filtre.")
    with c3:
        panel("Statut IA", "Gemini utilise si une cle API est fournie, fallback local sinon.")


def settings_page() -> None:
    render_hero("Parametres", "Configuration locale de l'experience analyste et seuils de pilotage.", ["Configuration", "Theme", "Thresholds"])
    st.slider("Seuil risque eleve", 50, 95, 68)
    st.slider("Seuil risque critique", 70, 99, 85)
    st.info("Les parametres sont conserves en session Streamlit pour cette demonstration.")


def render_page(page: str, cases: pd.DataFrame, all_cases: pd.DataFrame, gemini_api_key: str) -> None:
    if page == "Accueil":
        home_page(cases)
    elif page == "Dashboard executif":
        dashboard_page(cases)
    elif page == "Centre d'alertes":
        alerts_page(cases)
    elif page == "Analyse IA":
        ai_analysis_page(cases, gemini_api_key)
    elif page == "Investigation":
        investigation_page(cases)
    elif page == "Portefeuille dossiers":
        portfolio_page(cases)
    elif page == "Monitoring modeles":
        monitoring_page(cases)
    elif page == "Prevision":
        forecast_page(cases)
    elif page == "Carte fraude":
        map_page(cases)
    elif page == "Rapports":
        reports_page(cases, gemini_api_key)
    elif page == "Journal d'audit":
        audit_page()
    elif page == "IA Responsable":
        responsible_ai_page(cases)
    elif page == "Administration":
        admin_page(cases)
    elif page == "Parametres":
        settings_page()
    else:
        home_page(cases)
