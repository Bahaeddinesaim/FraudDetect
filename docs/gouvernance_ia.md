# Gouvernance IA, RGPD et IA responsable

## 1. Positionnement reglementaire

FraudAI est un systeme d'aide a la decision utilise par une autorite publique pour evaluer un risque individuel. Dans le cadre de l'AI Act, il doit etre traite comme un systeme a haut risque. Dans le cadre du RGPD, la base legale, la minimisation, la duree de conservation, l'information des personnes et le droit de recours doivent etre formalises.

Principe directeur: l'IA priorise les controles, mais ne prend jamais seule une decision ayant un effet juridique sur un citoyen ou une entreprise.

## 2. Data ownership et data lineage

| Element | Decision de gouvernance |
| --- | --- |
| Proprietaire metier | Administration responsable du dispositif: CAF, CPAM, DGFiP ou collectivite |
| Responsable traitement | Direction administrative competente |
| Sous-traitants | Hebergeur, integrateur, equipe data selon contrat RGPD |
| Source donnees | Systeme de demandes, historiques de controles, paiements, justificatifs structures |
| Lineage | Journaliser source, date extraction, transformations, version modele, score et agent consultant |

## 3. Qualite des donnees

Controles avant scoring:

- Schema attendu et types de donnees.
- Taux de valeurs manquantes par variable.
- Detection des montants anormaux.
- Verification des doublons.
- Controle de fraicheur des donnees.
- Rejet ou file de revue manuelle si qualite insuffisante.

## 4. Explicabilite

Le prototype inclut les variables du dossier et le modele peut etre complete par SHAP ou feature importance dans le notebook. Pour la production, chaque score doit fournir:

- Les principaux facteurs contribuant au risque.
- Le niveau de confiance.
- La version du modele.
- Une explication lisible par l'agent.

Les explications ne doivent pas etre utilisees comme preuve automatique de fraude. Elles orientent l'investigation.

## 5. Analyse des biais

Biais potentiels:

- Sur-representation de certains territoires dans les controles historiques.
- Effet revenu ou type de prestation.
- Biais lie a la qualite variable des donnees selon guichet.
- Biais de retour: les dossiers plus controles alimentent davantage les labels.

Mesures:

- Exclure les variables directement sensibles.
- Tester les performances par segment metier non sensible.
- Comparer taux de faux positifs par zone, canal, montant et type de prestation.
- Valider les seuils avec juristes, DPO et representants metier.

## 6. Faux positifs et faux negatifs

| Erreur | Consequence | Mesure |
| --- | --- | --- |
| Faux positif | Controle inutile, stress usager, charge agent | Seuil prudent, revue humaine, justification du controle |
| Faux negatif | Fraude non detectee, perte financiere | Recall eleve, controles aleatoires, re-entrainement |

## 7. Monitoring production

| Indicateur | Frequence | Alerte |
| --- | --- | --- |
| AUC-PR et recall sur labels confirmes | Mensuelle | Baisse > 10% |
| Taux de dossiers risque eleve | Hebdomadaire | Variation > 30% |
| Taux de faux positifs confirme | Mensuelle | > 5% ou hausse continue |
| Distribution des scores | Hebdomadaire | Derive statistique |
| Drift montant / variables principales | Mensuelle | PSI > 0,2 |
| Delai moyen de traitement agent | Mensuelle | Degradation > 15% |

## 8. Roadmap 12 mois

| Mois | Jalons | Responsable |
| --- | --- | --- |
| M1-M2 | Cadrage, DPIA, cartographie donnees | PM, DPO, metier |
| M3-M4 | Pipeline donnees, baseline ML, qualite | Data team |
| M5 | Benchmark modeles, seuils, explicabilite | Data science |
| M6 | Prototype agent et tests utilisateurs | Product, agents pilotes |
| M7 | Audit biais, securite, documentation | DPO, RSSI, data |
| M8 | Deploiement pilote | PM, IT |
| M9-M10 | Monitoring, calibration seuils | MLOps, metier |
| M11 | Audit interne modele | Audit, DPO |
| M12 | Decision generalisation ou ajustement | Comite IA |
