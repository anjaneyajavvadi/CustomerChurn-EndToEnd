from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

from src.pipelines.predict_pipeline import PredictPipeline
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig

app = FastAPI()

# Templates folder
templates = Jinja2Templates(directory="templates")

# Initialize your PredictPipeline
predictor = PredictPipeline(
    model_path=ModelTrainerConfig().trained_model_file_path,
    preprocessor_path=DataTransformationConfig().preprocessor_obj_file_path
)


@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Age: float = Form(...),
    Tenure: float = Form(...),
    Usage_Frequency: float = Form(...),
    Support_Calls: float = Form(...),
    Payment_Delay: float = Form(...),
    Total_Spend: float = Form(...),
    Last_Interaction: float = Form(...),
    Gender: str = Form(...),
    Contract_Length: str = Form(...),
    Subscription_Type: str = Form(...)
):
    try:
        input_df = pd.DataFrame([{
            "Age": Age,
            "Tenure": Tenure,
            "Usage Frequency": Usage_Frequency,
            "Support Calls": Support_Calls,
            "Payment Delay": Payment_Delay,
            "Total Spend": Total_Spend,
            "Last Interaction": Last_Interaction,
            "Gender": Gender,
            "Contract Length": Contract_Length,
            "Subscription Type": Subscription_Type
        }])

        preds, probs = predictor.predict(input_df)
        result = "Will Churn" if preds[0] == 1 else "Will Not Churn"
        prob_text = f"{probs[0]*100:.2f}%" if probs is not None else "N/A"

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": result,
            "probability": prob_text
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": f"Error: {e}",
            "probability": "N/A"
        })

