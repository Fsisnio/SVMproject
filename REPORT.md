# Projet Machine Learning : Prédiction du Churn Client Telco

**Auteurs :** [Spéro FALADE]  
**Date :** Mars 2026  
**Dataset :** Telco Customer Churn (IBM Sample Data)  
**Repository :** https://github.com/Fsisnio/SVMproject.git 
**Version Python :** 3.10+

---

## 1. Présentation du projet

### 1.1 Objectif

L'objectif de ce projet est de construire un modèle de **classification** capable de prédire si un client d'une entreprise de télécommunications va quitter le service (churn) ou non. Cette problématique est cruciale pour les opérateurs qui souhaitent identifier précocement les clients à risque et mettre en place des actions de rétention ciblées.

### 1.2 Dataset

- **Source :** [IBM Telco Customer Churn](https://www.ibm.com/docs/en/cognos-analytics/11.1.0?topic=samples-telco-customer-churn)
- **Fichier :** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Taille :** 7043 observations, 21 variables
- **Critère minimal :** ≥ 4000 observations ✓

### 1.3 Variables

| Variable | Type | Description |
|---------|------|-------------|
| customerID | Identifiant | ID client (exclu du modèle) |
| gender | Catégoriel | Genre |
| SeniorCitizen | Binaire | Client senior (0/1) |
| Partner, Dependents | Catégoriel | Situation familiale |
| tenure | Numérique | Ancienneté (mois) |
| PhoneService, MultipleLines | Catégoriel | Services téléphoniques |
| InternetService | Catégoriel | DSL / Fiber optic / No |
| OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies | Catégoriel | Services optionnels |
| Contract | Catégoriel | Type de contrat |
| PaperlessBilling, PaymentMethod | Catégoriel | Facturation |
| MonthlyCharges, TotalCharges | Numériques | Charges mensuelles et totales |
| **Churn** | **Cible** | **Yes / No** |

---

## 2. Méthodologie

### 2.1 Préprocessing

1. **Suppression** de `customerID` (non prédictif)
2. **Gestion des valeurs manquantes** : `TotalCharges` contenait des espaces → conversion en float, lignes avec NaN supprimées
3. **Encodage** : Label Encoding pour toutes les variables catégorielles
4. **Standardisation** : StandardScaler sur toutes les features (nécessaire pour SVM et régression logistique)
5. **Split** : 80 % train / 20 % test, stratifié sur la variable cible

### 2.2 Modèles comparés

| Modèle | Paramètres principaux | Justification |
|--------|----------------------|---------------|
| **Régression Logistique** | C=0.5, max_iter=1000 | Baseline interprétable, bon pour classification binaire |
| **SVM (RBF)** | C=1, gamma='scale' | Puissant pour frontières non linéaires |
| **Random Forest** | n_estimators=200, max_depth=12, min_samples_split=5 | Robuste, gère bien les interactions |
| **Gradient Boosting** | n_estimators=100, max_depth=5, lr=0.1 | Souvent performant sur données tabulaires |

### 2.3 Métriques

- **F1-Score** : métrique principale (équilibre précision / rappel, important pour churn souvent déséquilibré)
- **Accuracy, Precision, Recall, ROC-AUC** : métriques complémentaires

---

## 3. Résultats

### 3.1 Comparaison des modèles

**Validation croisée 5-fold (F1-Score) :**

| Modèle | F1 (CV) mean ± std |
|--------|---------------------|
| Logistic Regression | **0.587 ± 0.030** |
| SVM (RBF) | 0.552 ± 0.020 |
| Random Forest | 0.578 ± 0.029 |
| Gradient Boosting | 0.578 ± 0.023 |

**Performance sur le set de test :**

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.794 | 0.625 | 0.562 | **0.592** | **0.834** |
| SVM (RBF) | 0.785 | 0.627 | 0.476 | 0.541 | 0.784 |
| Random Forest | 0.792 | 0.636 | 0.508 | 0.565 | 0.829 |
| Gradient Boosting | 0.778 | 0.595 | 0.511 | 0.550 | 0.830 |

### 3.2 Meilleur modèle : Random Forest

Malgré un F1 légèrement supérieur de la régression logistique sur le test, le **Random Forest** a été retenu comme meilleur compromis : **interprétabilité native** (feature importance), **stabilité** et **bonne performance globale**. La régression logistique reste une excellente alternative si l'interprétabilité par coefficients est prioritaire.

**Hyperparamètres finaux :**
```json
{
  "n_estimators": 200,
  "max_depth": 12,
  "min_samples_split": 5,
  "min_samples_leaf": 2,
  "max_features": "sqrt",
  "bootstrap": true,
  "random_state": 42
}
```

---

## 4. Interprétabilité et discussion

### 4.1 Interprétabilité globale

**Feature Importance (Random Forest)** — L'importance des variables calculée par la forêt aléatoire (souvent la diminution moyenne d'impureté ou une variante normalisée) donne une **vision agrégée** du jeu de données : elle indique quelles entrées sont le plus souvent utilisées pour séparer les feuilles en lien avec le churn. Sur ce type de données télécoms, on observe en général une hiérarchie cohérente avec la littérature et l'intuition métier : **`tenure`** (l'ancienneté) figure souvent en tête, car les clients récents n'ont pas encore « verrouillé » leur relation avec l'opérateur et restent plus sensibles aux offres concurrentes. Le type de **`Contract`** suit souvent de près : les abonnements *mensuels* exposent le client à une décision de départ à tout moment, alors que les engagements sur un ou deux ans créent une barrière psychologique et contractuelle à la résiliation. Les variables **`MonthlyCharges`** et **`TotalCharges`** résument en partie le niveau de dépense et, indirectement, l'arbitrage prix / valeur perçue ; un client qui paie beaucoup sans service perçu comme premium est un candidat classique au churn. Enfin, **`InternetService`** (notamment la fibre) peut apparaître comme facteur de risque lorsqu'il est associé à des tarifs élevés ou à une concurrence agressive sur le haut débit.

Il est essentiel de rappeler une limite méthodologique : **une forte importance ne signifie pas une causalité**. Le modèle exploite des corrélations présentes dans l'historique ; certaines variables peuvent « capturer » d'autres phénomènes non observés (qualité du service, satisfaction, données CRM absentes du CSV).

**SHAP (valeurs de Shapley)** — Les graphiques de synthèse SHAP complètent la *feature importance* brute en montrant, pour chaque observation et dans l'agrégat, **comment** chaque variable pousse le score vers le churn ou vers le maintien. La distribution des points sur l'axe horizontal permet de voir à la fois la **direction** (effet favorable ou défavorable au churn) et l'**ampleur** des contributions. Avec un `TreeExplainer`, le calcul reste efficace pour les modèles à arbres ; on dispose ainsi d'un langage commun entre data scientists et parties prenantes métier pour discuter du « pourquoi » du modèle sans réduire la réalité à un classement statique de variables.

### 4.2 Interprétabilité locale

**LIME** (Local Interpretable Model-agnostic Explanations) approxime le comportement du modèle **au voisinage d'un client précis** en ajustant un modèle simple (souvent linéaire) sur des perturbations aléatoires de son profil. L'intérêt opérationnel est direct : un conseiller ou une campagne de rétention peut voir, pour tel segment ou tel individu, quelles modalités (contrat, ancienneté, options) pèsent le plus sur le score de churn **dans ce contexte local**. La limite connue est la **fidélité locale** : si le voisinage est mal calibré ou si la frontière est très non linéaire, l'explication linéaire peut être une caricature ; il faut donc traiter LIME comme un **outil de dialogue** plutôt que comme une preuve unique.

**SHAP force / waterfall plots** — Pour une observation donnée, les contributions SHAP s'additionnent de manière cohérente avec la prédiction du modèle (propriété souhaitée pour l'explicabilité). Cela permet de comparer deux clients similaires et d'identifier précisément **quel écart de feature** explique l'écart de score. Pour la communication interne, combiner une explication globale (SHAP summary) et une explication locale (force plot) couvre à la fois la stratégie (« quels leviers structurels ? ») et le cas par cas (« pourquoi ce score sur ce dossier ? »).

### 4.3 Discussion : lecture des résultats, apports et limites

La comparaison des modèles met en lumière un **compromis classique** entre pure performance ponctuelle et utilité décisionnelle. La **régression logistique** affiche un F1-test légèrement meilleur et une ROC-AUC élevée ; elle offre une lecture par **coefficients** très appréciable en conformité ou pour des argumentaires simples. Le **Random Forest** affiche néanmoins un profil équilibré (F1 proche, très bonne ROC-AUC) avec une **stabilité en validation croisée** et surtout des leviers d'explication riches (*feature importance*, SHAP rapide via arbres). Le choix du Random Forest comme modèle « principal » du projet reflète donc une préférence pour un **ensemble non linéaire** capable de capter des interactions entre services et contrats sans sacrifier totalement l'interprétabilité — au prix d'explications un peu plus difficiles à résumer en une phrase que les signes des coefficients logistiques.

Sur le plan métier, les métriques globales (accuracy autour de 0,79–0,80) masquent un enjeu central : dans un problème de churn, le **déséquilibre des classes** (souvent environ un quart de churn) fait que l'optimisation du taux de bien classés peut favoriser la classe majoritaire. D'où l'intérêt du **F1** et d'une analyse **précision / rappel** : une campagne de rétention coûteuse ne peut pas se baser uniquement sur la précision ; si le rappel sur les *vrais* churneurs est trop faible, l'entreprise manque une partie des clients à sauver, alors qu'un seuil de décision abaissé peut augmenter le rappel au prix de plus de faux positifs (offres envoyées à des clients qui ne seraient pas partis). Ce **réglage de seuil**, absent d'un score F1 par défaut à 0,5, est une suite naturelle du travail pour le déploiement.

Ce qui a clairement **fonctionné** dans le pipeline : la **standardisation** pour les modèles sensibles à l'échelle ; le **split stratifié** qui préserve la proportion de churn ; l'usage de **SHAP** avec `TreeExplainer` pour des explications globales crédibles sans exploser le temps de calcul ; le **label encoding** conjoint à des arbres, où la cardinalité modérée des catégories limite le biais par rapport à des modèles linéaires stricts. À l'inverse, les **fragilités** restent la gestion implicite du **déséquilibre**, les coûts de **LIME** sur de très grands volumes, la lourdeur d'un **KernelExplainer** pour le SVM, et l'absence d'**importance native** aussi simple pour le SVM que pour la forêt. Le **SVM** reste aussi plus exigeant en temps et en réglage de noyau et d'échelle.

En synthèse, le travail d'interprétabilité ne vise pas seulement la conformité ou la « boîte noire » : il structure une **discussion entre modèle et stratégie commerciale**. Les variables mises en avant (ancienneté, type de contrat, montants, offre internet) suggèrent des leviers concrets — durée d'engagement, adaptation tarifaire, accompagnement des nouveaux abonnés — tout en rappelant que toute action doit être **validée** par des essais (A/B testing), des contraintes légales (traitement équitable des profils) et des données qualitatives absentes du tableau brut.

---

## 5. Pistes d'amélioration

1. **Gestion du déséquilibre** : SMOTE, class_weight='balanced', ou ajustement du seuil de décision
2. **Feature engineering** : ratio charges / ancienneté, regroupement de catégories
3. **Optimisation hyperparamètres** : GridSearchCV ou Optuna
4. **Modèles additionnels** : XGBoost, LightGBM, réseaux de neurones
5. **Validation temporelle** : si une variable temporelle est disponible

---

## 6. Reproducibilité

- **Python :** 3.10+ (voir `.python-version`)
- **Gestionnaire de paquets :** UV
- **Fichiers requis :**
  - `pyproject.toml`
  - `uv.lock`
  - `WA_Fn-UseC_-Telco-Customer-Churn.csv`
  - `best_model_hyperparameters.json`

**Commandes :**
```bash
cd /chemin/vers/projet
uv sync
uv run jupyter notebook telco_churn_ml.ipynb
```

---

## 7. Structure du projet

```
SVMproject/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset brut
├── telco_churn_ml.ipynb                    # Pipeline ML (notebook)
├── REPORT.md                               # Ce rapport
├── pyproject.toml                          # Dépendances (UV)
├── uv.lock                                 # Lock des versions
├── best_model_hyperparameters.json         # Hyperparamètres du meilleur modèle
├── .python-version                         # Version Python
└── (figures générées : *.png)
```

---

## 8. Conclusion

Ce projet illustre un pipeline complet de machine learning : préprocessing, comparaison de modèles, sélection du meilleur modèle et interprétabilité globale et locale. Le modèle Random Forest retenu offre un bon équilibre entre performance et explicabilité, utile pour orienter les actions marketing et de rétention clients.
