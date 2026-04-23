# Projet ML - Prédiction du Churn Client Telco

Projet de machine learning réalisé dans le cadre d'un cours de modélisation prédictive.

## Résumé

- **Dataset** : Telco Customer Churn (7043 observations, IBM)
- **Objectif** : Classification binaire (prédire si un client va quitter = Churn)
- **Modèles** : Régression Logistique, SVM, Random Forest, Gradient Boosting
- **Interprétabilité** : SHAP (global + local), feature importance, LIME

## Prérequis

- Python 3.10+
- [UV](https://docs.astral.sh/uv/) (gestionnaire de paquets)

## Installation et exécution

```bash
# Cloner le repo
git clone https://github.com/Fsisnio/SVMproject.git
cd SVMproject

# Installer les dépendances avec UV
uv sync

# Exécuter le pipeline ML (notebook)
uv run jupyter notebook telco_churn_ml.ipynb
```

## Structure

| Fichier | Description |
|---------|-------------|
| `telco_churn_ml.ipynb` | Pipeline ML (notebook Jupyter) |
| `REPORT.md` | Rapport détaillé et analyse |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dataset brut |
| `pyproject.toml` / `uv.lock` | Dépendances (UV) |
| `best_model_hyperparameters.json` | Hyperparamètres du meilleur modèle |
| `.python-version` | Version Python |

## Livrables

- [x] Fichier `.md` avec commentaires et résultats
- [x] Notebook `.ipynb` exécutable avec commentaires (le script `.py` reste en local, non versionné)
- [x] Dataset brut
- [x] `pyproject.toml` et `uv.lock` (UV)
- [x] Spécification des hyperparamètres du meilleur modèle
- [x] Version Python spécifiée

## Validation du dataset


