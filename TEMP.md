# Prédiction du Diabète avec Machine Learning

Ce projet de classification utilise le **Pima Indians Diabetes Dataset** pour prédire la probabilité de développer un diabète. Le modèle a été développé avec plusieurs algorithmes de machine learning, dont **Random Forest**, **XGBoost**, et **Logistic Regression**. L'objectif est de fournir une solution de prédiction rapide, interprétable et performante basée sur des données médicales.

## Table des matières
1. [Contexte](#contexte)
2. [Données](#données)
3. [Pipeline du Projet](#pipeline-du-projet)
   - [Exploration des données](#exploration-des-données)
   - [Prétraitement des données](#prétraitement-des-données)
   - [Modélisation et Entraînement](#modélisation-et-entrainement)
   - [Évaluation des Performances](#évaluation-des-performances)
   - [Interprétabilité](#interprétabilité)
4. [Utilisation du Modèle](#utilisation-du-modèle)
5. [Installation et Configuration](#installation-et-configuration)
6. [Auteurs](#auteurs)
7. [License](#license)

## Contexte

Le **Pima Indians Diabetes Dataset** contient des informations sur des patient·e·s de l'Inde Pima (une communauté amérindienne), utilisées pour prédire la probabilité de diabète. Chaque ligne du dataset contient des informations comme les niveaux de glucose, la pression artérielle, le nombre de grossesses, l'indice de masse corporelle (IMC), etc.

Ce projet met en œuvre plusieurs techniques de machine learning pour prédire la présence de diabète en fonction de ces informations.

## Données

Les données sont disponibles sous forme de fichier CSV et incluent les caractéristiques suivantes :
- **Pregnancies** : Nombre de grossesses
- **Glucose** : Concentration de glucose dans le sang
- **BloodPressure** : Pression artérielle diastolique (mm Hg)
- **SkinThickness** : Épaisseur de la peau
- **Insulin** : Niveau d'insuline sérique (mu U/ml)
- **BMI** : Indice de masse corporelle (kg/m²)
- **DiabetesPedigreeFunction** : Fonction d'ascendance génétique pour le diabète
- **Age** : Âge du patient
- **Outcome** : Variable cible (0 = Pas de diabète, 1 = Diabète)

## Pipeline du Projet

### Exploration des données

Dans cette étape, nous avons exploré les données pour comprendre leur structure, visualiser les relations entre les variables, et identifier les valeurs manquantes ou aberrantes. Cette phase est cruciale pour définir comment nettoyer les données et quelles transformations appliquer.

### Prétraitement des données

Les données ont été traitées de la manière suivante :
- Les valeurs manquantes ont été remplacées par la médiane de la colonne correspondante.
- Les caractéristiques ont été **normalisées** pour garantir une échelle comparable lors de l'entraînement des modèles.
- Des transformations ont été appliquées pour rendre les données prêtes à être utilisées dans des modèles de machine learning.

### Modélisation et Entraînement

Plusieurs modèles ont été testés :
- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost**
- **Support Vector Machine (SVM)**

Pour chaque modèle, nous avons effectué une **optimisation des hyperparamètres** à l'aide de **GridSearchCV** pour trouver la meilleure configuration de paramètres.

### Évaluation des Performances

Les performances des modèles ont été évaluées à l’aide de plusieurs métriques :
- **Précision (Accuracy)**
- **Rappel (Recall)**
- **F1-Score**
- **AUC (Area Under Curve)**
  
La **matrice de confusion** a été utilisée pour visualiser les erreurs de classification et mieux comprendre les performances du modèle.

### Interprétabilité

Afin d'interpréter les décisions du modèle, nous avons utilisé **SHAP** (Shapley Additive Explanations), une méthode d'explicabilité des modèles de machine learning. Cela permet de comprendre l'impact de chaque caractéristique sur les prédictions du modèle.

## Utilisation du Modèle

Après l’entraînement du modèle, vous pouvez charger le modèle sauvegardé et l'utiliser pour faire des prédictions avec de nouvelles données.

### Exemple de prédiction :

```python
import joblib

# Charger le modèle et le scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Nouvelles données à prédire
new_data = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]  # Exemple de nouvelle entrée
new_data_scaled = scaler.transform(new_data)

# Prédiction
prediction = model.predict(new_data_scaled)
print(f"Prediction: {prediction[0]} (0 = Pas diabète, 1 = Diabète)")
