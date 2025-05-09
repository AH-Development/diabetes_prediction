# Prédiction du Diabète avec Machine Learning

Ce projet de classification utilise le **Diabetes Dataset** pour prédire la probabilité de développer un diabète. Le modèle a été développé avec plusieurs algorithmes de machine learning, dont **Random Forest**, **Support Vector Machin**, et **Logistic Regression**. L'objectif est de fournir une solution de prédiction rapide, interprétable et performante basée sur des données médicales.

## Table des matières
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

## Données

Les données sont disponibles sous forme de fichier CSV et incluent les caractéristiques suivantes :
- **gender** : Sexe (Female/Male/Others)
- **age** : Âge de l'individu
- **heart_disease** : Antécédents de maladie cardiaque (0 = Non, 1 = Oui)
- **hypertension** : Hypertension (0 = Non, 1 = Oui)
- **smoking_history** : Historique de tabagisme (Never/Former/Current/No Info)
- **bmi** : Indice de masse corporelle (kg/m²)
- **HbA1c_level** : Taux d'HbA1c (glycémie sur 3 mois)
- **blood_glucose_level** : Glycémie à jeun
- **diabetes** : Présence de diabète (0 = Non, 1 = Oui)

## Pipeline du Projet

### Exploration des données

Dans cette étape, nous avons exploré les données pour comprendre leur structure, visualiser les relations entre les variables, et identifier les valeurs manquantes ou aberrantes. Cette phase est cruciale pour définir comment nettoyer les données et quelles transformations appliquer.

### Prétraitement des données

Les données ont été traitées de la manière suivante :
- Les valeurs manquantes ont été supprimées.
- Les caractéristiques ont été **normalisées** pour garantir une échelle comparable lors de l'entraînement des modèles.
- Des transformations ont été appliquées pour rendre les données prêtes à être utilisées dans des modèles de machine learning.

### Modélisation et Entraînement

Plusieurs modèles ont été testés :
- **Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **XGBoost**

Nous avons effectué une **optimisation des hyperparamètres** à l'aide de **GridSearchCV** pour trouver la meilleure configuration de paramètres.

### Évaluation des Performances

Les performances des modèles ont été évaluées à l’aide de plusieurs métriques :
- **Précision (Accuracy)**
- **Rappel (Recall)**
- **F1-Score**
- **AUC (Area Under Curve)**
  
La **matrice de confusion** a été utilisée pour visualiser les erreurs de classification et mieux comprendre les performances du modèle.

