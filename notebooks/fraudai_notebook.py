# %% [markdown]
# # FraudAI - Notebook ML
#
# Pipeline de detection de fraude pour services publics.
# Ce fichier est un notebook Jupyter au format percent script. Il peut etre ouvert dans VS Code ou converti en `.ipynb`.

# %%
import pandas as pd
from src.fraudai_core import load_transactions, split_xy, train_and_select

# %% [markdown]
# ## 1. Chargement des donnees
#
# Le projet utilise `data/creditcard.csv` si disponible. Sinon, un dataset synthetique compatible est genere.

# %%
df = load_transactions("../data/creditcard.csv")
df.head()

# %%
df["Class"].value_counts(normalize=True).rename("rate")

# %% [markdown]
# ## 2. Preparation et feature engineering
#
# Variables ajoutees: heure, logarithme du montant, indicateur transaction de nuit.

# %%
X, y = split_xy(df)
X[["Time", "Amount", "Hour", "LogAmount", "IsNight"]].describe()

# %% [markdown]
# ## 3. Modelisation
#
# Trois modeles sont compares:
# - Logistic Regression + SMOTE
# - Random Forest
# - HistGradientBoosting

# %%
model, best_report, reports, X_test, y_test = train_and_select(df)
pd.DataFrame([r.__dict__ for r in reports]).sort_values("average_precision", ascending=False)

# %% [markdown]
# ## 4. Interpretation
#
# Pour un rendu complet, ajouter SHAP sur un echantillon de test:
#
# ```python
# import shap
# explainer = shap.Explainer(model.predict_proba, X_test.sample(200, random_state=42))
# values = explainer(X_test.sample(50, random_state=42))
# shap.plots.beeswarm(values[..., 1])
# ```
#
# Dans le contexte public, ces explications servent a orienter l'agent et non a justifier une sanction automatique.
