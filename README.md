# Project_04: Harnessing Machine Learning for Proactive Diabetes Risk Prediction

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
<!-- Add other relevant badges if applicable -->

## Table of Contents

1.  [Overview](#overview)
2.  [The Challenge: Why Diabetes Prediction Matters](#the-challenge-why-diabetes-prediction-matters)
3.  [Project Objectives](#project-objectives)
4.  [Technical Approach & Methodology](#technical-approach--methodology)
    *   [Data Foundation (Versions 1-2)](#data-foundation-versions-1-2)
    *   [EDA-Driven Refinement (Versions 2-5)](#eda-driven-refinement-versions-2-5)
    *   [Model Development & Optimization (Versions 6-13)](#model-development--optimization-versions-6-13)
5.  [Key Findings & Results](#key-findings--results)
    *   [Exploratory Data Analysis Insights](#exploratory-data-analysis-insights)
    *   [Feature Importance](#feature-importance)
    *   [Model Performance & Selection](#model-performance--selection)
6.  [Technology Stack](#technology-stack)
7.  [Dataset](#dataset)
8.  [Installation](#installation)
9.  [Usage](#usage)
    *   [Training the Models](#training-the-models)
    *   [Using Pre-trained Models (`main.py`)](#using-pre-trained-models-mainpy)
10. [Actionable Insights & Impact](#actionable-insights--impact)
11. [Project Structure](#project-structure)
12. [Disclaimer](#disclaimer)
13. [Contributors](#contributors)
14. [License](#license)
15. [Acknowledgements](#acknowledgements)

## Overview

This project implements and evaluates various machine learning models to predict the risk of diabetes onset using readily available health indicators. Driven by the significant global health burden of Diabetes Mellitus, the goal was not just to build a predictive model but to undertake a rigorous, end-to-end data science workflow. This involved meticulous data acquisition and preprocessing, in-depth exploratory data analysis (EDA), thoughtful feature engineering, diverse model development, comprehensive evaluation, and the generation of actionable insights. The project culminates in a high-performing Gradient Boosting model, demonstrating the potential of machine learning to empower early detection, personalized interventions, and ultimately, improve public health outcomes.

The journey, meticulously documented through iterative versions (see [Optimization Table](./Optimization%20Table.pdf) and [Project Notes](./NOTES%20-%20HARNESSING%20MACHINE%20LEARNING%20FOR%20PROACTIVE%20DIABETES%20RISK%20PREDICTION%20-%20EMPOWERING%20EARLY%20DETECTION%20AND%20MANAGEMENT.pdf)), highlights the power of data-driven refinement and the importance of integrating domain understanding with empirical model evaluation.

## The Challenge: Why Diabetes Prediction Matters

Diabetes Mellitus affects hundreds of millions globally, imposing immense burdens on individuals and healthcare systems. Beyond lifestyle adjustments, it significantly elevates the risk of severe complications like cardiovascular disease, neuropathy, nephropathy, and retinopathy. Early detection and proactive management are critical for mitigating these risks, improving patient outcomes, and reducing associated healthcare costs. This project addresses the urgent need for more effective diabetes risk prediction tools, aiming to identify at-risk individuals sooner.

## Project Objectives

This project was structured around the following key objectives (as outlined in the [Project Proposal](./PROJECT%20PROPOSAL%20-%20HARNESSING%20MACHINE%20LEARNING%20FOR%20PROACTIVE%20DIABETES%20RISK%20PREDICTION%20-%20EMPOWERING%20EARLY%20DETECTION%20AND%20MANAGEMENT.pdf)):

1.  **Data Acquisition and Intelligent Preprocessing:** Acquire and meticulously preprocess the "Diabetes Health Indicators Dataset," ensuring data quality and readiness for modeling.
2.  **In-depth EDA and Feature Engineering:** Conduct comprehensive EDA to understand data characteristics, identify patterns/correlations, and engineer new, informative features.
3.  **Development and Training of Diverse ML Models:** Develop, train, and rigorously test a range of models (Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, Neural Networks) using Scikit-learn and TensorFlow/Keras.
4.  **Comprehensive Model Evaluation and Comparative Analysis:** Rigorously evaluate models using metrics like F1-score, AUC-ROC, Accuracy, Precision, and Recall, comparing their performance to identify the optimal approach.
5.  **Actionable Insights and Data-Driven Recommendations:** Extract insights on key risk factors from EDA and model results, translating predictions into practical recommendations for healthcare professionals.
6.  **Effective Communication and Visualization of Findings:** Create compelling visualizations and reports (including this README, the notebook, and the presentation) to communicate the project's lifecycle, results, and impact effectively.

## Technical Approach & Methodology

The project followed a structured, iterative methodology, documented across 13 distinct versions in the [Project Notes](./NOTES%20-%20HARNESSING%20MACHINE%20LEARNING%20FOR%20PROACTIVE%20DIABETES%20RISK%20PREDICTION%20-%20EMPOWERING%20EARLY%20DETECTION%20AND%20MANAGEMENT.pdf). Each step was deliberately taken based on empirical results and analytical insights:

### Data Foundation (Versions 1-2)

*   **Dataset:** Utilized the `diabetes_binary_health_indicators_BRFSS2015.csv` dataset.
*   **Outlier Handling:** Addressed outliers in numerical features (BMI, Age, Income, MentHlth, PhysHlth) using **KNN Imputation** (Version 1). This was chosen over removal or capping to preserve data integrity while mitigating extreme value influence, recognizing the sensitivity of health data where extremes might be valid cases. Ordinal features (GenHlth, Education) were correctly excluded from this numerical outlier handling.
*   **Baseline Establishment:** Confirmed baseline performance after initial handling (Version 2).

### EDA-Driven Refinement (Versions 2-5)

*   **Deep EDA:** Heavily emphasized EDA using visualizations like KDE-enhanced Pair Plots, Box Plots, Violin Plots, and Correlation Matrices (detailed in Notebook/Notes for V2 & V4 analysis).
*   **Key Feature Identification:** Confirmed the importance of BMI, Age, GenHlth, HighBP, and Income. Revealed non-linear relationships, justifying the exploration of more complex models.
*   **Feature Engineering (V3):** Created interaction terms (e.g., `Health_Risk_Index = GenHlth * HighBP`, `BMI_Age_Interaction`) and categorical BMI features (`BMI_Category`) to capture synergistic effects observed in EDA.
*   **Scaling Experiments (V4):** Compared `RobustScaler` with `StandardScaler`. Found minimal performance difference after initial outlier handling, retaining `StandardScaler` for broader compatibility.
*   **Multicollinearity Management (V5):** Assessed multicollinearity using **Variance Inflation Factor (VIF)**. Implemented an iterative feature removal process, pruning redundant features (initially `BMI_Category_Underweight`, later `Education` and `CholCheck`) while monitoring model integrity and performance stability.

### Model Development & Optimization (Versions 6-13)

A diverse set of algorithms was explored, aligning with Objective 3:

*   **Logistic Regression (V1-9):** Served as an interpretable baseline. Initial optimization via `GridSearchCV` (V6-7, V7 expanded grid) showed limited gains. Feature importances were derived from coefficients (V8), and further analysis included Calibration Curves and Odds Ratios (V9) for deeper insights, though its predictive ceiling was apparent (AUC ~0.824). Regularization (L1/L2) was employed to manage potential multicollinearity.
*   **Random Forest (V10):** Introduced to capture non-linearities. Optimized using `RandomizedSearchCV` (balancing thoroughness and efficiency), providing improved performance over the baseline LR (AUC ~0.803).
*   **Gradient Boosting (V9 initial, V11-12):** Emerged as the top performer. Systematically optimized using `GridSearchCV` (V11), incorporating `subsample`, `max_features`, and crucially, **early stopping** (V12). Version 11 achieved the highest AUC-ROC (0.828541) and F1-Score (0.837105). Version 12 maintained strong performance with potential improvements in generalization and efficiency via early stopping.
*   **Neural Network (V12 initial, V13):** Explored the potential of deep learning using Keras/TensorFlow. Optimized using `Keras Tuner` (RandomSearch) with L2 regularization and early stopping (V13). Achieved respectable results (AUC ~0.8246) but did not surpass the optimized Gradient Boosting models in this instance, potentially requiring further tuning or larger datasets.

## Key Findings & Results

### Exploratory Data Analysis Insights

*   **Key Discriminators:** EDA (Pair Plots, Box Plots, Violin Plots) strongly highlighted **BMI, Age, General Health perception (GenHlth), High Blood Pressure (HighBP), and Income** as significant predictors of diabetes risk. `HighBP` showed near-binary separation. `GenHlth` also showed strong discriminatory power.
*   **Non-Linearities:** KDE plots (e.g., BMI vs. Age, BMI vs. GenHlth) visually confirmed non-linear relationships and interactions between features, supporting the use of tree-based ensembles and neural networks.
*   **Outliers:** Outliers were present, particularly in BMI and Age, reinforcing the need for robust handling (KNN imputation) or models less sensitive to them (like tree-based methods).

### Feature Importance

Feature importance analysis (derived from Logistic Regression coefficients, Random Forest Gini impurity, and Gradient Boosting feature importance) consistently highlighted:

*   **High Importance:** `GenHlth`, `HighBP`, `BMI`, `Age`, `Income`.
*   **Interaction Terms:** Engineered features like `BMI_Age_Interaction` and `Health_Risk_Index` also demonstrated predictive value, capturing combined effects.
*   *(Refer to the feature importance plots within the [Jupyter Notebook](./Project_04_Diabetes_Prediction.ipynb).*

### Model Performance & Selection

*   **Iterative Improvement:** The [Optimization Table](./Optimization%20Table.pdf) clearly tracks the quantitative impact of each methodological step (outlier handling, feature engineering, VIF reduction, hyperparameter tuning) on model performance (F1-score, AUC-ROC).
*   **Final Model:** **Gradient Boosting Version 11** achieved the highest combination of F1-score (0.837105) and AUC-ROC (0.828541). **Version 12** (Gradient Boosting with early stopping) offered highly competitive performance (F1: 0.837746, AUC: 0.827785) with added benefits of efficiency and potential robustness against overfitting.
*   **Justification:** Gradient Boosting (V11 or V12) was selected as the most successful model, significantly outperforming the baseline Logistic Regression and demonstrating strong predictive capability, balancing discrimination (AUC) and classification accuracy (F1).
*   **Supporting Visualizations:** Model performance is further detailed through ROC Curves and Confusion Matrices in the notebook. Calibration Curves were analyzed, particularly for Logistic Regression, confirming reasonable probability calibration.

## Technology Stack

*   **Language:** Python (3.8+)
*   **Core Libraries:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn (for LR, DT, RF, GB, Preprocessing, Metrics, Tuning), TensorFlow, Keras, Keras Tuner (for NN development and tuning)
*   **Statistics:** Statsmodels (for VIF calculation)
*   **Visualization:** Matplotlib, Seaborn
*   **Environment:** Jupyter Notebook, Conda

## Dataset

*   **Source:** "Diabetes Health Indicators Dataset" derived from the Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey, conducted by the CDC.
*   **File:** `diabetes_binary_health_indicators_BRFSS2015.csv` (Located in the repository root).
*   **Description:** Contains health indicators (like high blood pressure, cholesterol, BMI, smoking status, physical activity, fruit/vegetable consumption, general health, age, education, income) and a binary target variable indicating diabetes status (0 = No Diabetes, 1 = Diabetes).
*   **Reference:** [https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

## Installation

To set up the environment and run this project locally:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/CamdenBeck/Project_04.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd Project_04
    ```
3.  **Create a Conda environment** (Python 3.8 recommended):
    ```bash
    conda create --name project04_env python=3.8 
    ```
    *(Replace `project04_env` with your desired environment name)*
4.  **Activate the environment:**
    ```bash
    conda activate project04_env
    ```
5.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Installing TensorFlow might require specific C++ Redistributables or CUDA setup depending on your system if GPU support is desired. CPU-based TensorFlow should install via pip.*

## Usage

### Training the Models

1.  Open the Jupyter Notebook: `Project_04_Diabetes_Prediction.ipynb`.
2.  Execute the cells sequentially in Google Colab to perform data loading, preprocessing, EDA, feature engineering, model training, tuning, and evaluation.
3.  **Warning:** Training and hyperparameter tuning (especially GridSearchCV and Keras Tuner) can be computationally intensive and time-consuming, particularly on standard hardware. Running on platforms like Google Colab with GPU/TPU acceleration (for Neural Networks) is recommended for faster execution. In our experience, it took hours to run the notebook. There is a pretrained model located in the `Models` directory if you choose to use that.

### Using Pre-trained Models (`main.py`)

A simple script `main.py` is provided to demonstrate predictions using pre-trained models (assuming they are saved, e.g., as `.joblib` or `.keras` files - *Note: Saving/loading logic might need to be added/confirmed in the notebook/script if not already present*).

1.  Ensure the required pre-trained model files are present in the repository.
2.  Run the script from the activated environment:
    ```bash
    python main.py
    ```
3.  Follow the on-screen prompts (likely involving selecting a model and inputting feature values) to get a prediction.

## Actionable Insights & Impact

This project delivers more than just code; it provides valuable insights and a potentially impactful tool:

*   **Risk Factor Identification:** Confirms and quantifies the impact of key factors (BMI, Age, GenHlth, HighBP, Income) on diabetes risk.
*   **Early Screening:** The developed model (especially the optimized Gradient Boosting) can serve as a powerful, non-invasive screening tool to identify individuals at higher risk earlier than traditional methods might allow.
*   **Personalized Strategies:** Understanding individual risk profiles based on these indicators allows healthcare professionals to tailor preventative strategies (lifestyle advice, monitoring frequency) more effectively.
*   **Reduced Healthcare Burden:** Proactive management facilitated by early detection can potentially reduce the incidence of severe complications, lowering long-term healthcare costs and improving patient quality of life.
*   **ML Workflow Demonstration:** Showcases a robust and adaptable machine learning pipeline applicable to other predictive healthcare analytics problems.

## Project Structure

```
Project_04/
│
├── .gitignore
├── Project Proposal - Harnessing Machine Learning for Proactive Diabetes Risk Prediction - Empowering Early Detection and Management.pdf # Initial proposal
├── diabetes_binary_health_indicators_BRFSS2015.csv # Dataset file
├── Tensorflow_Keras_installation.ipynb     # Code for installing Python modules
├── Project_04_Diabetes_Prediction.ipynb    # Main Jupyter Notebook with all code
├── Notes - Harnessing Machine Learning for Proactive Diabetes Risk Prediction - Empowering Early Detection and Management.pdf # Project summary & methodology notes
├── Optimization_Metrics.pdf                  # Table tracking model performance per version
├── Harnessing Machine Learning for Proactive Diabetes Risk Prediction - Empowering Early Detection and Management.pdf      # PDF export of the work presentation (includes outputs/plots)
├── README.md                               # This file
├── main.py                                 # Script for using pre-trained models (if implemented)
└── requirements.txt                        # Python dependencies
```

## Disclaimer

These are machine learning models developed for educational and research purposes to showcase data science techniques applied to health data. **They are not intended for medical diagnosis.** Predictions should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for any health concerns or before making any decisions related to your health or treatment.

## Contributors

*   Camden Beck
*   Alex Gerwer
*   Sylvester Gold
*   Sarah Gutierrez

## Acknowledgements

*   **Dataset:** Centers for Disease Control and Prevention (CDC) - Behavioral Risk Factor Surveillance System (BRFSS) 2015.
*   **Tools:** Python Software Foundation, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, TensorFlow, Keras, Keras Tuner, Jupyter, Statsmodels communities.
