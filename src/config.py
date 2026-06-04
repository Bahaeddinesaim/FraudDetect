from __future__ import annotations

APP_TITLE = "FraudDetect AI"
GEMINI_MODEL = "gemini-2.5-flash"
MODEL_PATH = "models/fraudai_model.joblib"
METRICS_PATH = "models/model_metrics.csv"

PALETTE = {
    "primary": "#0B132B",
    "secondary": "#1C2541",
    "accent": "#3A86FF",
    "success": "#06D6A0",
    "warning": "#FFBE0B",
    "danger": "#E63946",
    "background": "#F8FAFC",
    "card": "#FFFFFF",
    "text": "#111827",
    "muted": "#64748B",
    "border": "#E5E7EB",
}

RISK_LEVELS = ["Critique", "Eleve", "Moyen", "Faible"]
RISK_COLORS = {
    "Critique": PALETTE["danger"],
    "Eleve": "#FB8500",
    "Moyen": PALETTE["warning"],
    "Faible": PALETTE["success"],
}

NAVIGATION = [
    "Accueil",
    "Dashboard executif",
    "Centre d'alertes",
    "Analyse IA",
    "Investigation",
    "Portefeuille dossiers",
    "Monitoring modeles",
    "Prevision",
    "Carte fraude",
    "Rapports",
    "Journal d'audit",
    "IA Responsable",
    "Administration",
    "Parametres",
]

REGION_COORDS = {
    "Ile-de-France": (48.8566, 2.3522),
    "Auvergne-Rhone-Alpes": (45.7640, 4.8357),
    "Nouvelle-Aquitaine": (44.8378, -0.5792),
    "Occitanie": (43.6047, 1.4442),
    "Hauts-de-France": (50.6292, 3.0573),
    "Grand Est": (48.5734, 7.7521),
    "Provence-Alpes-Cote d'Azur": (43.2965, 5.3698),
    "Bretagne": (48.1173, -1.6778),
    "Normandie": (49.1829, -0.3707),
    "Pays de la Loire": (47.2184, -1.5536),
}
