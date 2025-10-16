import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from src.utils.exceptionhandler import CustomException
from src.utils.logger import logging
from src.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("src", "artifacts", "models", "preprocessor.pkl")
    transformed_train_path: str = os.path.join("src", "artifacts", "transformed", "train.csv")
    transformed_test_path: str = os.path.join("src", "artifacts", "transformed", "test.csv")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                "Age", "Tenure", "Usage Frequency", "Support Calls",
                "Payment Delay", "Total Spend", "Last Interaction"
            ]
            categorical_columns = ["Gender", "Subscription Type", "Contract Length"]

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', ColumnTransformer(
                        transformers=[
                            ('age', StandardScaler(), ['Age']),
                            ('tenure', MinMaxScaler(), ['Tenure']),
                            ('usage', MinMaxScaler(), ['Usage Frequency']),
                            ('support', RobustScaler(), ['Support Calls']),
                            ('payment', MinMaxScaler(), ['Payment Delay']),
                            ('spend', StandardScaler(), ['Total Spend']),
                            ('interaction', MinMaxScaler(), ['Last Interaction']),
                        ],
                        remainder='drop'
                    ), numerical_columns),
                    ('cat', OneHotEncoder(drop=None, handle_unknown='ignore'), categorical_columns)
                ],
                remainder='drop'
            )

            logging.info("✅ Data Transformation pipeline created successfully.")
            return preprocessor, numerical_columns, categorical_columns

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test datasets loaded successfully.")

            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)

            target_column_name = 'Churn'

            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name]
            X_test = test_df.drop(columns=[target_column_name])
            y_test = test_df[target_column_name]

            preprocessing_obj, num_cols, cat_cols = self.get_data_transformer_object()

            X_train_scaled = preprocessing_obj.fit_transform(X_train)
            X_test_scaled = preprocessing_obj.transform(X_test)

            ohe = preprocessing_obj.named_transformers_['cat']
            ohe_features = list(ohe.get_feature_names_out(cat_cols))

            transformed_columns = num_cols + ohe_features

            train_transformed_df = pd.DataFrame(X_train_scaled, columns=transformed_columns)
            test_transformed_df = pd.DataFrame(X_test_scaled, columns=transformed_columns)

            train_transformed_df[target_column_name] = y_train.values
            test_transformed_df[target_column_name] = y_test.values

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_path), exist_ok=True)

            train_transformed_df.to_csv(self.data_transformation_config.transformed_train_path, index=False)
            test_transformed_df.to_csv(self.data_transformation_config.transformed_test_path, index=False)

            logging.info("✅ Transformed data saved successfully as CSV with column names.")

            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor object saved successfully.")

            return (
                train_transformed_df,
                test_transformed_df,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    train_path = os.path.join("src", "artifacts", "ingested", "train.csv")
    test_path = os.path.join("src", "artifacts", "ingested", "test.csv")

    transformer = DataTransformation()
    train_df, test_df, preprocessor_path = transformer.initiate_data_transformation(train_path, test_path)

    print("✅ Transformation Complete")
    print("Train Shape:", train_df.shape)
    print("Test Shape:", test_df.shape)
    print("Preprocessor Saved At:", preprocessor_path)
