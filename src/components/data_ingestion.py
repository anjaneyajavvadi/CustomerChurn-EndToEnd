import os
import sys
import shutil
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi
from src.utils.logger import logging
from src.utils.exceptionhandler import CustomException



# ✅ Updated config class — saves inside src/artifacts
@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("src", "artifacts", "data", "raw")          # raw downloaded dataset
    train_data_path: str = os.path.join("src", "artifacts", "ingested", "train.csv")  # final train
    test_data_path: str = os.path.join("src", "artifacts", "ingested", "test.csv")    # final test


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


            # Step 2: Find downloaded CSVs
            downloaded_files = [f for f in os.listdir(raw_data_dir) if f.endswith(".csv")]
            if not downloaded_files:
                raise FileNotFoundError("No CSV files found in downloaded dataset folder.")

            train_csv_path = os.path.join(raw_data_dir, "customer_churn_dataset-training-master.csv")
            test_csv_path = os.path.join(raw_data_dir, "customer_churn_dataset-testing-master.csv")

            train_data = pd.read_csv(train_csv_path)
            test_data = pd.read_csv(test_csv_path)

            # Step 3: Ensure ingested dir exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Step 4: Save as train.csv and test.csv in ingested folder
            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            # Step 5: Delete raw data folder
            if os.path.exists(raw_data_dir):
                shutil.rmtree(raw_data_dir)
                logging.info(f"Deleted raw data folder: {raw_data_dir}")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print("✅ Train Path:", train_path)
    print("✅ Test Path:", test_path)
