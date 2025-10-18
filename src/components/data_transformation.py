import os
import sys
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder

from src.utils.exceptionhandler import CustomException
from src.utils.logger import logging
from src.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('src','artifacts', 'models', 'preprocessor.pkl')
    transformed_train_csv_path: str = os.path.join('src','artifacts', 'transformed', 'train.csv')
    transformed_test_csv_path: str = os.path.join('src','artifacts', 'transformed', 'test.csv')


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

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
            return preprocessor, numerical_columns, categorical_columns

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test datasets loaded successfully.")

            df=pd.concat([train_df,test_df])

            df.dropna(inplace=True)

            
            
            target_column = 'Churn'

            X_= df.drop(columns=[target_column])
            y_= df[target_column]

            X_train,X_test,y_train,y_test=train_test_split(X_,y_,test_size=0.2,random_state=42)

            # Get preprocessor and column lists
            preprocessor, num_cols, cat_cols = self.get_data_transformer_object()

            # Fit & transform
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            # Get OHE feature names
            ohe = preprocessor.named_transformers_['categorical']
            ohe_features = list(ohe.get_feature_names_out(cat_cols))
            transformed_columns = num_cols + ohe_features

            # Create DataFrames with target
            train_transformed_df = pd.DataFrame(X_train_scaled, columns=transformed_columns)
            train_transformed_df[target_column] = y_train.values

            test_transformed_df = pd.DataFrame(X_test_scaled, columns=transformed_columns)
            test_transformed_df[target_column] = y_test.values

            # Create folders
            os.makedirs(os.path.dirname(self.config.transformed_train_csv_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path), exist_ok=True)

            # Save CSVs
            train_transformed_df.to_csv(self.config.transformed_train_csv_path, index=False)
            test_transformed_df.to_csv(self.config.transformed_test_csv_path, index=False)

            logging.info("✅ Transformed data saved successfully as CSV.")

            # Save preprocessor
            save_object(file_path=self.config.preprocessor_obj_file_path, obj=preprocessor)
            logging.info(f"Preprocessor object saved at: {self.config.preprocessor_obj_file_path}")

            return train_transformed_df, test_transformed_df, self.config.preprocessor_obj_file_path

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
