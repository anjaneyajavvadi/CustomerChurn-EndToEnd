import os
import sys
import shutil
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi
from src.utils.logger import logging
from src.utils.exceptionhandler import CustomException


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("src", "artifacts", "data", "raw")         
    train_data_path: str = os.path.join("src", "artifacts", "ingested", "train.csv") 
    test_data_path: str = os.path.join("src", "artifacts", "ingested", "test.csv")   


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def download_from_kaggle(self, dataset_name, download_dir):
        try:
            logging.info("Connecting to Kaggle API...")
            os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
            os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

            api = KaggleApi()
            api.authenticate()

            os.makedirs(download_dir, exist_ok=True)
            logging.info(f"Downloading dataset: {dataset_name}")
            api.dataset_download_files(dataset_name, path=download_dir, unzip=True)
            logging.info("Dataset downloaded and extracted successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion process")
        try:
        
            dataset_name = "muhammadshahidazeem/customer-churn-dataset"
            raw_data_dir = os.path.dirname(self.ingestion_config.raw_data_path)
            self.download_from_kaggle(dataset_name, raw_data_dir)

        
            train_csv_path = os.path.join(raw_data_dir, "customer_churn_dataset-training-master.csv")
            test_csv_path = os.path.join(raw_data_dir, "customer_churn_dataset-testing-master.csv")

            if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
                raise FileNotFoundError("Train or test CSV file not found in raw dataset folder.")

        
            train_data = pd.read_csv(train_csv_path)
            test_data = pd.read_csv(test_csv_path)

            logging.info(f"Original train shape: {train_data.shape}")
            logging.info(f"Original test shape: {test_data.shape}")

        
            combined_data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
            logging.info(f"Combined dataset shape: {combined_data.shape}")

        
            train_split, test_split = train_test_split(
                combined_data, test_size=0.2, random_state=42, shuffle=True
            )

        
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

        
            train_split.to_csv(self.ingestion_config.train_data_path, index=False)
            test_split.to_csv(self.ingestion_config.test_data_path, index=False)

        
            if os.path.exists(raw_data_dir):
                shutil.rmtree(raw_data_dir)
                logging.info(f"Deleted raw data folder: {raw_data_dir}")

            logging.info("Data Ingestion (with new 80/20 split) completed successfully.")
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print("✅ Train Path:", train_path)
    print("✅ Test Path:", test_path)
