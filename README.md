# FraudAI - Detection de fraude par IA

Projet fil rouge MSc Machine Learning EPITA: solution de detection de fraude dans les services publics avec pipeline ML, business case, gouvernance IA et prototype Streamlit.

## Choix metier

Le projet cible la fraude aux aides publiques et aux demandes de remboursement, avec transposition pedagogique sur le dataset Kaggle ULB `Credit Card Fraud Detection`. La logique ML est identique: une tres faible proportion de dossiers frauduleux doit etre reperee sans automatiser la decision administrative.

## Structure

- `train.py`: entrainement, comparaison de 3 modeles et sauvegarde du meilleur modele.
- `app.py`: prototype Streamlit pour les agents publics.
- `src/fraudai_core.py`: chargement des donnees, feature engineering, modeles, seuil de decision et scoring.
- `docs/business_case.md`: cadrage financier, KPI, ROI et planning.
- `docs/gouvernance_ia.md`: RGPD, AI Act, monitoring, biais et supervision humaine.
- `docs/presentation.md`: trame de slides pour la soutenance.
- `notebooks/fraudai_notebook.py`: notebook Jupyter au format percent script.

## Donnees

Option recommandee: telecharger `creditcard.csv` depuis Kaggle et le placer dans `data/creditcard.csv`.

Colonnes attendues: `Time`, `Amount`, `V1` a `V28`, `Class`.

Si le fichier Kaggle n'est pas present, le projet genere automatiquement un dataset synthetique compatible. Cela permet de verifier le pipeline et le prototype sans bloquer le rendu.

## Installation

```powershell
pip install -r requirements.txt
```

## Telechargement Kaggle

```powershell
python download_data.py
```

Ce script utilise KaggleHub pour telecharger `mlg-ulb/creditcardfraud`, puis copie `creditcard.csv` vers `data/creditcard.csv`.

## Entrainement

```powershell
python train.py
```

Sorties:

- `models/fraudai_model.joblib`
- `models/model_metrics.csv`

## Application Streamlit

```powershell
streamlit run app.py
```

Le prototype affiche un score de risque de 0 a 100 et une action recommandee:

- 0-30: risque faible, traitement standard.
- 31-70: risque modere, controle selon capacite.
- 71-100: risque eleve, controle prioritaire par un agent.

## Message cle du projet

FraudAI aide les agents publics a prioriser les controles. Il ne remplace jamais la decision humaine, et chaque score doit rester explicable, auditable et contestable.
