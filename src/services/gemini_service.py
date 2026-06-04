from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import GEMINI_MODEL


@dataclass
class GeminiResult:
    data: dict[str, Any]
    provider: str
    error: str | None = None


def _safe_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def fallback_investigation(case: pd.Series | dict[str, Any], question: str = "") -> dict[str, Any]:
    score = int(case.get("risk_score", 0))
    anomalies = []
    if float(case.get("amount", 0)) > 12000:
        anomalies.append("Montant significativement superieur au profil attendu.")
    if int(case.get("hour", 12)) <= 5 or int(case.get("hour", 12)) >= 22:
        anomalies.append("Depot effectue sur une plage horaire atypique.")
    if int(case.get("duplicate_count", 0)) >= 3:
        anomalies.append("Multiples demandes similaires detectees.")
    if float(case.get("doc_mismatch", 0)) >= 0.45:
        anomalies.append("Incoherences documentaires elevees.")
    if not anomalies:
        anomalies.append("Score eleve par combinaison de signaux faibles.")

    level = str(case.get("risk_level", "Moyen"))
    return {
        "resume": f"Dossier {case.get('case_id', 'N/A')} relatif a {case.get('service', 'un service public')} avec un risque {level.lower()}.",
        "niveau_risque": level,
        "score": score,
        "anomalies": anomalies,
        "justification": "Le score combine montant, heure de depot, doublons, incoherences documentaires, velocite et antecedents.",
        "actions": [
            "Verifier l'identite et les pieces justificatives.",
            "Comparer avec les demandes recentes du meme beneficiaire ou mandataire.",
            "Soumettre le dossier a validation humaine avant toute decision.",
            "Documenter la decision dans le journal d'audit.",
        ],
        "decision": "Investigation humaine prioritaire" if score >= 68 else "Controle standard renforce",
    }


def investigator_prompt(case: pd.Series | dict[str, Any]) -> str:
    payload = {k: str(v) for k, v in dict(case).items() if k not in {"lat", "lon"}}
    return f"""
Tu es un agent IA enqueteur pour une plateforme GovTech de lutte contre la fraude.
Analyse le dossier ci-dessous et retourne uniquement un JSON valide avec exactement:
resume, niveau_risque, score, anomalies, justification, actions, decision.

Contraintes:
- score est un entier 0-100.
- anomalies et actions sont des listes de chaines.
- Ne prends jamais une decision punitive automatique.
- Mentionne le besoin de controle humain si le risque est eleve.

Dossier:
{json.dumps(payload, ensure_ascii=False)}
"""


def analyze_case(api_key: str, case: pd.Series | dict[str, Any]) -> GeminiResult:
    if not api_key:
        return GeminiResult(fallback_investigation(case), "Fallback local")
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            investigator_prompt(case),
            generation_config={"temperature": 0.15, "response_mime_type": "application/json"},
            request_options={"timeout": 30},
        )
        data = _safe_json(response.text)
        return GeminiResult(data=data, provider=f"Gemini API - {GEMINI_MODEL}")
    except Exception as exc:
        return GeminiResult(fallback_investigation(case), "Fallback local", error=str(exc))


def chat(api_key: str, case: pd.Series | dict[str, Any], history: list[dict[str, str]], question: str) -> GeminiResult:
    if not api_key:
        data = fallback_investigation(case, question)
        answer = (
            f"{data['resume']} Les principaux facteurs sont: "
            f"{', '.join(data['anomalies'])}. Decision proposee: {data['decision']}."
        )
        return GeminiResult({"answer": answer}, "Fallback local")
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        context = json.dumps({k: str(v) for k, v in dict(case).items()}, ensure_ascii=False)
        messages = "\n".join(f"{m['role']}: {m['content']}" for m in history[-8:])
        prompt = f"Contexte dossier: {context}\nHistorique:\n{messages}\nQuestion analyste: {question}\nReponds en francais, de facon concise et operationnelle."
        response = model.generate_content(prompt, generation_config={"temperature": 0.25}, request_options={"timeout": 30})
        return GeminiResult({"answer": response.text}, f"Gemini API - {GEMINI_MODEL}")
    except Exception as exc:
        return GeminiResult({"answer": fallback_investigation(case)["justification"]}, "Fallback local", error=str(exc))
