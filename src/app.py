import os
import threading
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from src.utils.logger import logging
from src.pipelines.predict_pipeline import PredictPipeline
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.pipelines.trainer_pipeline import TrainerPipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model_path = ModelTrainerConfig().trained_model_file_path
preprocessor_path = DataTransformationConfig().preprocessor_obj_file_path

training_in_progress = False
predictor = None

def run_training():
    global predictor, training_in_progress
    training_in_progress = True
    logging.info("⚙️ Background training started...")
    pipeline = TrainerPipeline()
    artifacts = pipeline.run()
    logging.info(f"✅ Training finished. Best model: {artifacts['best_model_name']} ({artifacts['best_score']:.4f})")
    predictor = PredictPipeline(
        model_path=artifacts['model_path'],
        preprocessor_path=artifacts['preprocessor_path']
    )
    training_in_progress = False

# If no model exists, start training in the background
if not os.path.exists(model_path):
    logging.info("⚠️ No trained model found — training will start in background.")
    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()
else:
    predictor = PredictPipeline(model_path=model_path, preprocessor_path=preprocessor_path)
    logging.info("✅ Pretrained model found — ready for predictions.")

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    """Render the input form or training message."""
    if training_in_progress:
        return templates.TemplateResponse("loading.html", {"request": request})
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
    if training_in_progress:
        return templates.TemplateResponse("loading.html", {"request": request})

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


@app.get("/status")
async def status():
    return {"training_in_progress": training_in_progress}