# Bank Customer Churn Prediction API & ML Pipeline

A production-ready Machine Learning system designed to predict bank customer churn, packaged into a high-performance FastAPI backend, containerized via Docker, and deployed serverlessly to Google Cloud Run. This project outlines an end-to-end machine learning engineering workflow—from exploratory data analysis and ensemble modeling to cloud deployment and scaling.

# Live Demo & Documentation

**Cloud Run Service URL**: https://bank-churn-service-699539648647.us-central1.run.app

**Interactive API Docs (Swagger UI)**: https://bank-churn-service-699539648647.us-central1.run.app/docs

**Alternative API Docs (ReDoc)**: https://bank-churn-service-699539648647.us-central1.run.app/redoc


# Project Overview & Objective

Customer churn (attrition) is a critical metric for retail banking institutions. Retaining existing customers is significantly more cost-effective than acquiring new ones.

**Objective**: Build an accurate predictive classification model to identify bank customers who are at high risk of closing their accounts.

**Impact**: Enables proactive retention strategies, targeted marketing discounts, and optimized customer relationship management.

# Dataset & Data Analysis (EDA)
**Preprocessing steps**: 
  - Handled missing values and encoded categorical features.
  - Scaled numerical features so that different value ranges don't confuse the models.
  - Kept an eye on class imbalance since churned customers are usually fewer than active ones.

# Machine Learning Approach
- **Models tested**: Started with simpler models like Logistic Regression and Random Forest, then moved to gradient boosting algorithms (XGBoost and CatBoost).
- **Final Model**: Built an ensemble pipeline (`ensemble_package.pkl`) combining the best-performing models to get a more reliable prediction.
- **Evaluation**: Used cross-validation and tuned hyperparameters to make sure the model doesn't just memorize the training data but actually performs well on new data.
- **Deployment**: Saved the final pipeline using `joblib` so FastAPI can load it instantly for real-time predictions.

# API & Deployment
- Built a simple and fast REST API using **FastAPI**.
- Used **Pydantic** to validate incoming JSON data so the app won't crash on bad inputs.
- Wrapped everything inside a **Docker** container and deployed it to **Google Cloud Run** so it's accessible online via a live URL.

# Tech Stack
* **Language**: Python 3.12.3
* **Framework**: FastAPI, Uvicorn, Pydantic
* **ML Libraries**: CatBoost, XGBoost, Lightgbm Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, Joblib
* **DevOps**: Doker, Google Cloud Run


# Bank Customer Churn Prediction API

## Local Run
`../.my_env/bin/python -m uvicorn main:app --reload --port 8001`
- Docs: http://127.0.0.1:8001/docs

## Docker Run
- Build: `docker build -t bank-churn-api .`
- Run: `docker run -d -p 8002:8000 --name bank-churn-container bank-churn-api`
- Docs: http://localhost:8002/docs