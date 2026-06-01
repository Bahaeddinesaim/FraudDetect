# Presentation finale - FraudAI

## Slide 1 - Titre

FraudAI: detection de fraude dans les services publics par Machine Learning.

Message: aider les agents a prioriser, sans automatiser la decision.

## Slide 2 - Probleme

- Fraude publique a fort impact financier.
- Ressources de controle limitees.
- Fraude rare donc difficile a detecter par controles aleatoires.
- Besoin d'un outil explicable, auditable et conforme.

## Slide 3 - Cas d'usage

Fraude aux aides publiques, transposable CAF, CPAM, DGFiP et subventions.

Donnees pedagogiques: Kaggle ULB Credit Card Fraud Detection.

## Slide 4 - Business Case

- Cout pilote: 650 kEUR.
- Gain annuel prudent: 2,1 MEUR.
- ROI annee 1 estime: 223%.
- KPI principaux: AUC-PR, recall, precision, faux positifs, temps agent.

## Slide 5 - Pipeline ML

1. Chargement donnees.
2. Feature engineering: heure, log montant, indicateur nuit.
3. Split stratifie.
4. Benchmark de 3 modeles.
5. Optimisation du seuil selon recall et F1.
6. Sauvegarde du meilleur modele.

## Slide 6 - Resultats

Presenter `models/model_metrics.csv` apres entrainement:

- Logistic Regression + SMOTE.
- Random Forest.
- HistGradientBoosting.

Metric principale: AUC-PR, car les classes sont fortement desequilibrees.

## Slide 7 - Prototype Streamlit

- Saisie d'un dossier.
- Score de risque 0-100.
- Niveau faible, modere, eleve.
- Action recommandee.
- Rappel de supervision humaine.

## Slide 8 - Gouvernance IA

- Systeme haut risque AI Act.
- RGPD: minimisation, conservation, droit d'acces, recours.
- Data lineage du dossier au score.
- Journalisation des acces et versions modele.

## Slide 9 - Biais et erreurs

- Faux positifs: risque de controle injustifie.
- Faux negatifs: perte financiere.
- Audits par segment.
- Monitoring drift et recalibration.

## Slide 10 - Recommandation

Lancer un pilote de 8 mois sur un perimetre limite, avec agents volontaires, DPO, RSSI et comite IA.

Condition de passage a l'echelle: performance stable, faux positifs maitrises, recours operationnel et audit favorable.
