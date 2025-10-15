import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder

from src.utils.exceptionhandler import CustomException
from src.utils.logger import logging
from src.utils.utils import save_object


# ✅ Configuration class — defines where to save preprocessor object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('src','artifacts', 'models', 'preprocessor.pkl')
    transformed_train_path: str = os.path.join('src','artifacts', 'transformed', 'train.npy')
    transformed_test_path: str = os.path.join('src','artifacts', 'transformed', 'test.npy')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates a preprocessing pipeline:
        - Scales numeric features with appropriate scalers
        - One-hot encodes categorical features
        """
        try:
            numerical_columns = ["Age", "Tenure", "Usage Frequency", "Support Calls",
                                 "Payment Delay", "Total Spend", "Last Interaction"]
            categorical_columns = ["Gender", "Contract Length", "Subscription Type"]

            preprocessor = ColumnTransformer(
                transformers=[
                    ('age', StandardScaler(), ['Age']),
                    ('tenure', MinMaxScaler(), ['Tenure']),
                    ('usage', MinMaxScaler(), ['Usage Frequency']),
                    ('support', RobustScaler(), ['Support Calls']),
                    ('payment', MinMaxScaler(), ['Payment Delay']),
                    ('spend', StandardScaler(), ['Total Spend']),
                    ('interaction', MinMaxScaler(), ['Last Interaction']),
                    ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
                ],
                remainder='drop'
            )

            logging.info("✅ Data Transformation pipeline created successfully.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # ✅ Step 1: Load train & test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test datasets loaded successfully.")

            # ✅ Step 2: Separate target and features
            target_column_name = 'Churn'

            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name])
            y_test = test_df[target_column_name]

            logging.info("Separated input features and target column.")

            # ✅ Step 3: Create preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # ✅ Step 4: Apply transformations
            X_train_scaled = preprocessing_obj.fit_transform(X_train)
            X_test_scaled = preprocessing_obj.transform(X_test)

            logging.info("Applied preprocessing transformations on train and test data.")

            # ✅ Step 5: Combine scaled data with target
            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            # ✅ Step 6: Create transformed data folder if not exists
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_path), exist_ok=True)

            # ✅ Step 7: Save transformed data
            np.save(self.data_transformation_config.transformed_train_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_path, test_arr)

            logging.info("Transformed data saved successfully.")

            # ✅ Step 8: Save preprocessor object
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor object saved successfully.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Example paths (adjust if needed)
    train_path = os.path.join("src", "artifacts", "ingested", "train.csv")
    test_path = os.path.join("src", "artifacts", "ingested", "test.csv")


    transformer = DataTransformation()
    train_data, test_data, preprocessor_path = transformer.initiate_data_transformation(train_path, test_path)

    print("✅ Transformation Complete")
    print("Train Data Shape:", train_data.shape)
    print("Test Data Shape:", test_data.shape)
    print("Preprocessor Saved At:", preprocessor_path)
