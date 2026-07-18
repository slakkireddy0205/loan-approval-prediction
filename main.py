from fastapi import FastAPI
from pydantic import BaseModel , Field
from typing import Literal
import joblib
import json
import pandas as pd


app = FastAPI()

model = joblib.load('loan_model.joblib')
with open('feature_columns.json','r') as f:
    feature_columns = json.load(f)
@app.get("/")
def read_root():
    return{'message':"Loan Approval  prediction API is running"}
class LoanAppilication(BaseModel):
    no_of_dependents : int = Field(...,ge=0,description="Number of dependents")
    education :Literal['Graduate',"Not Graduate"]
    self_employed :Literal["Yes","No"]
    income_annum : int = Field(...,ge=0, description="Annual income")
    loan_amount: int = Field(..., gt=0, description="Requested loan amount")
    loan_term: int = Field(..., gt=0, description="Loan term in years")
    cibil_score: int = Field(..., ge=300, le=900, description="CIBIL credit score")
    residential_assets_value: int = Field(..., ge=0)
    commercial_assets_value: int = Field(..., ge=0)
    luxury_assets_value: int = Field(..., ge=0)
    bank_asset_value: int = Field(..., ge=0)
@app.post("/predict")
def predict_loan(application : LoanAppilication):
    input_dict = application.model_dump()

    input_dict['education'] = 1 if input_dict['education'] == 'Graduate' else 0
    input_dict['self_employed'] = 1 if input_dict['self_employed'] == 'Yes'else 0

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)[0]
    result = "Approved" if prediction == 1 else "Rejected"

    return {"prediction": result}