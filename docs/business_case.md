# Business Case IA - FraudAI

## 1. Probleme metier

Les administrations traitent un volume eleve de demandes d'aides, de remboursements et de transactions publiques. Les fraudes sont rares, mais leur impact budgetaire est majeur. Le probleme ML consiste a attribuer un score de risque a chaque dossier afin de prioriser les controles humains.

Le cas d'usage retenu est la fraude aux aides publiques, transposable aux aides sociales, remboursements CPAM, subventions et paiements administratifs.

## 2. Objectifs

- Detecter davantage de dossiers frauduleux sans augmenter proportionnellement les controles manuels.
- Reduire le temps passe sur les dossiers a faible risque.
- Limiter les faux positifs afin d'eviter la stigmatisation de citoyens ou entreprises legitimes.
- Garantir une decision finale humaine, explicable et contestable.

## 3. KPI

| KPI | Objectif pilote | Justification |
| --- | ---: | --- |
| Recall fraude | >= 80% | Ne pas manquer les dossiers a risque eleve |
| Precision | A optimiser selon capacite agents | Reduire les controles inutiles |
| Taux de faux positifs | < 5% cible metier | Proteger les usagers honnetes |
| AUC-PR | Suivi principal | Adapte aux classes tres desequilibrees |
| Delai moyen de tri | -40% | Priorisation automatique des dossiers |
| Dossiers controles / agent | +25% | Meilleure allocation des ressources |

## 4. Estimation budgetaire pilote

| Poste | Cout estime |
| --- | ---: |
| Equipe projet data science, PM, metier | 180 000 EUR |
| Collecte, qualite et preparation des donnees | 70 000 EUR |
| Developpement modele ML et evaluation | 90 000 EUR |
| Interface agent et integration SI | 80 000 EUR |
| Infrastructure securisee | 55 000 EUR |
| Conformite RGPD, AI Act, securite | 65 000 EUR |
| Tests utilisateurs et formation | 35 000 EUR |
| Maintenance et monitoring annee 1 | 75 000 EUR |
| Total pilote | 650 000 EUR |

## 5. ROI estime

Hypothese pilote prudente:

- Perimetre: 1 million de dossiers par an.
- Montant moyen d'anomalie confirmee: 2 000 EUR.
- Taux de fraude reel estime: 0,5%.
- Fraudes detectables par priorisation IA: 35% des fraudes non detectees auparavant.
- Taux de recuperation effectif: 60%.

Gain annuel estime:

`1 000 000 x 0,5% x 35% x 2 000 x 60% = 2,1 M EUR`

ROI annee 1:

`(2,1 M - 0,65 M) / 0,65 M = 223%`

Sur 3 ans, avec couts de run plus faibles apres le pilote, le gain net peut depasser 5 M EUR sur un perimetre administratif limite.

## 6. Planning

| Phase | Duree |
| --- | ---: |
| Cadrage metier et juridique | 1 mois |
| Collecte et comprehension des donnees | 1 mois |
| Preparation, qualite et feature engineering | 1 mois |
| Modelisation et benchmarking | 2 mois |
| Interface Streamlit et integration pilote | 1 mois |
| Tests agents, formation, ajustement seuils | 1 mois |
| Deploiement pilote et monitoring initial | 1 mois |
| Total | 8 mois |

## 7. Risques metier

| Risque | Impact | Mitigation |
| --- | --- | --- |
| Faux positifs | Controle injustifie d'usagers honnetes | Seuil ajuste, revue humaine, droit de recours |
| Faux negatifs | Fraudes non detectees | Recall prioritaire, audit erreurs, re-entrainement |
| Biais socio-economique | Discrimination indirecte | Tests par segments, variables sensibles exclues, audits |
| Derive du modele | Perte de performance | Monitoring mensuel et revalidation trimestrielle |
| Non-conformite RGPD / AI Act | Risque juridique | DPIA, documentation, supervision humaine |
