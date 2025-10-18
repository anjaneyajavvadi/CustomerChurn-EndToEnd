import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.exceptionhandler import CustomException
from src.utils.logger import logging

class TrainerPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run(self):
        try:
            # Step 1: Data Ingestion
            logging.info("Starting Data Ingestion...")
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Train Path: {train_path}, Test Path: {test_path}")

            # Step 2: Data Transformation
            logging.info("Starting Data Transformation...")
            train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(train_path, test_path)
            logging.info(f"Preprocessor saved at: {preprocessor_path}")

            # Step 3: Model Training
            logging.info("Starting Model Training...")
            best_model_name, best_score, model_path = self.model_trainer.initiate_model_training()
            logging.info(f"Best Model: {best_model_name} with Accuracy: {best_score}")
            logging.info(f"Model saved at: {model_path}")

            return {
                "train_path": train_path,
                "test_path": test_path,
                "preprocessor_path": preprocessor_path,
                "model_path": model_path,
                "best_model_name": best_model_name,
                "best_score": best_score
            }

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainerPipeline()
    artifacts = pipeline.run()
    print("✅ Training Pipeline finished successfully!")
    print("Artifacts:", artifacts)
