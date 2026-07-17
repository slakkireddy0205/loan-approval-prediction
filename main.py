from fastapi import FastAPI
import joblib
import json

app = FastAPI()

model = joblib.load('loan_model.joblib')
with open('feature_columns.json','r') as f:
    feature_columns = json.load(f)
@app.get("/")
def read_root():
    return{'message':"Loan Approval  prediction API is running"}