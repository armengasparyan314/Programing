# Premier League Player Clustering & Analysis Pipeline

A comprehensive Machine Learning and Exploratory Data Analysis (EDA) system designed to segment Premier League football players based on their performance metrics and tactical attributes. This project implements an end-to-end data science workflow—from data collection and Principal Component Analysis (PCA) feature reduction to unsupervised clustering and an interactive web dashboard deployed via Streamlit Cloud.

# Live Demo & Application

**Streamlit Web App**: [pl-player-clustering.streamlit.app](https://pl-player-clustering.streamlit.app)

# Project Overview & Objective

Understanding player roles, tactical profiles, and performance similarities is crucial for modern football scouting, tactical analysis, and team composition. 

**Objective**: Apply unsupervised machine learning techniques to group Premier League players into distinct performance clusters based on their statistical profiles.

**Impact**: Enables scouts, analysts, and football enthusiasts to identify similar player profiles, evaluate tactical contributions, and visualize player groupings dynamically.

# Dataset & Data Analysis (EDA)
**Preprocessing & Feature Engineering steps**: 
- Cleaned and normalized player match statistics from Premier League data sources.
- Handled missing values and standardized numerical features so that different metric scales (e.g., goals vs. passing accuracy) contribute evenly.
- Explored data distributions and correlations using interactive visualizations.

# Machine Learning Approach
- **Dimensionality Reduction (PCA)**: Applied Principal Component Analysis (`Player_Clustering_Pca.ipynb`) to reduce high-dimensional player statistics into principal components while retaining maximum variance.
- **Unsupervised Clustering**: Grouped players into meaningful clusters representing distinct tactical roles and playing styles.
- **Evaluation & Visualization**: Utilized `matplotlib` and `seaborn` to generate feature loading plots, cluster scatter plots, and performance comparisons.

# Interactive Dashboard (Streamlit)
- Built a multi-page interactive web application using **Streamlit** (`app.py`).
- Allows users to filter players, explore cluster characteristics, and visualize PCA components dynamically in real time.

# Tech Stack
- **Language**: Python
- **Framework**: Streamlit
- **ML & Data Science Libraries**: Scikit-Learn (PCA, Clustering), Pandas, NumPy, Matplotlib, Seaborn
- **Deployment**: Streamlit Cloud, Git & GitHub

# Local Installation & Running

## Prerequisites
Make sure you have Python installed, then clone the repository and set up your environment.

## Run Locally
1. Activate your virtual environment:
   ```bash
   cd "Football_Player_Clustering"
   ../.my_env/bin/streamlit run app.py