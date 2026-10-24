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
    CreditScore: float
    Gender: int
    Age: float
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geography_France: bool 
    Geography_Germany: bool
    Geography_Spain: bool
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

# սկզբում joblib.load միջոցով քաշում ենք "ensemble_package.pkl" ֆայլը։
# ինչից հետո մեր արդեն ստղծած dictionary ից առանձնացնում ենք բոլոր պարամետրերը։
# Գրում ենք մուտքային տվյալների սխեման, որ սխալ տվյալ մուտքագրելու դեպքում կոդը չաշխատի։
# Երբ հաճախորդը POST հարցում է ուղարկում /predict հասցեին,
# ֆունկցիան ընդունում է տվյալները CustomerFeatures Pydantic սխեմայի միջոցով։
# Սա ավտոմատ կերպով ստուգում է տվյալների տիպերը և կանխում սերվերի խափանումը սխալ տվյալների դեպքում։
# data.model_dump() հրամանի միջոցով ստացված տվյալները վերածվում են բառարանի, 
# որից հետո pd.DataFrame[...]-ի միջոցով ստեղծվում է Pandas աղյուսակ (input_df), 
# քանի որ մեր CatBoost և XGBoost մոդելները աշխատում են հենց DataFrame-ների հետ։
# Մոդելների վրա կանչվում է .predict_proba(input_df) մեթոդը։ [:, 1][0] գրելաձևի միջոցով
# մենք մատրիցայից վերցնում ենք բոլոր տողերի միայն երկրորդ սյան արժեքը 
# (դրական դասի՝ հաճախորդի հեռանալու հավանականությունը) 
# և այն մաքուր float թվի տեսքով վերագրում cat_prob-ին ու xgb_prob-ին։
# Վերջում միասնական հավանականությունը համեմատվում է մեր գտած օպտիմալ շեմի հետ
# (ensemble_prob >= threshold), որոշվում է վերջնական որոշումը (0 կամ 1),
# և սերվերը հաճախորդին վերադարձնում է հստակ JSON պատասխան՝ ներառելով հավանականությունը,
#  կանխատեսումն ու օգտագործված շեմը։»