import os
import sys
import pandas as pd
from src.utils.utils import load_object
from src.utils.exceptionhandler import CustomException
from src.utils.logger import logging

class PredictPipeline:
    def __init__(self, model_path, preprocessor_path):
        try:
            logging.info("Loading saved model and preprocessor...")
            self.model = load_object(model_path)
            self.preprocessor = load_object(preprocessor_path)
            logging.info("✅ Artifacts loaded successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_df: pd.DataFrame):
        try:
            logging.info("Transforming input data for prediction...")
            X_transformed = self.preprocessor.transform(input_df)

            # Predict class
            preds = self.model.predict(X_transformed)

            # Predict probability if model supports it
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_transformed)[:, 1]  # probability of class 1
            else:
                probs = None

            logging.info("✅ Prediction complete.")
            return preds, probs
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Example usage
    model_path = os.path.join("src","artifacts", "models", "best_model.pkl")
    preprocessor_path = os.path.join("src","artifacts", "models", "preprocessor.pkl")

    predictor = PredictPipeline(model_path=model_path, preprocessor_path=preprocessor_path)

    # Load new data to predict
    new_data = pd.read_csv("src/artifacts/new_data.csv")
    predictions = predictor.predict(new_data)

    print("Predictions:", predictions)
