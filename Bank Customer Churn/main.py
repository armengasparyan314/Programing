import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 0. Ապահովում ենք բացարձակ ուղի, որ ֆայլը միշտ անխափան գտնվի
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "ensemble_package.pkl")

# 1. Բեռնում ենք մեր պահպանված փաթեթը
package = joblib.load(model_path)

cat_model = package["cat_model"]
xgb_model = package["xgb_model"]
cat_weight = package["cat_weight"]
xgb_weight = package["xgb_weight"]
threshold = package["threshold"]

# 2. Ստեղծում ենք FastAPI հավելվածը
app = FastAPI(title="Ensemble ML API (CatBoost + XGBoost)")


# 3. Սահմանում ենք մուտքային տվյալների սխեման (Pydantic)
class CustomerFeatures(BaseModel):
    Age: float
    CreditScore: float
    EstimatedSalary: float
    Balance: float
    NumOfProducts: int
    Tenure: int
    Gender: int
    IsActiveMember: int
    Geography_Germany: bool
    HasCrCard: int
    Geography_Spain: bool
    Geography_France: bool
    old_customers: bool


# 4. Ստեղծում ենք կանխատեսման endpoint-ը
@app.post("/predict")
def predict_ensemble(data: CustomerFeatures):
    # Տվյալները սարքում ենք DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # Ստանում ենք հավանականությունները առանձին-առանձին
    cat_prob = cat_model.predict_proba(input_df)[:, 1][0]
    xgb_prob = xgb_model.predict_proba(input_df)[:, 1][0]

    # Կիրառում ենք անսամբլի կշիռները (weighted average)
    ensemble_prob = cat_weight * cat_prob + xgb_weight * xgb_prob

    # Կիրառում ենք մեր գտած լավագույն շեմը (Threshold)
    prediction = int(ensemble_prob >= threshold)

    return {
        "ensemble_probability": float(ensemble_prob),
        "prediction": prediction,
        "used_threshold": float(threshold),
        "status": "success",
    }