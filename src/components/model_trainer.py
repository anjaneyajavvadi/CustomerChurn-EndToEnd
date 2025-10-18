import os
import sys
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils.logger import logging
from src.utils.exceptionhandler import CustomException
from src.utils.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("src","artifacts", "models", "best_model.pkl")
    transformed_train_path = os.path.join("src","artifacts", "transformed", "train.csv")
    transformed_test_path = os.path.join("src","artifacts", "transformed", "test.csv")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self):
        try:
            logging.info("📥 Loading transformed train and test data")

            train_df = pd.read_csv(self.config.transformed_train_path)
            test_df = pd.read_csv(self.config.transformed_test_path)

            X_train = train_df.drop(columns=['Churn'])
            y_train = train_df['Churn']
            X_test = test_df.drop(columns=['Churn'])
            y_test = test_df['Churn']

            X_train = X_train.values
            X_test = X_test.values
            y_train = y_train.values
            y_test = y_test.values


            logging.info(f"✅ Training shape: {X_train.shape}, Test shape: {X_test.shape}")

            models = {
                "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
                "Ridge Classifier": RidgeClassifier(random_state=42),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
                "Random Forest Classifier": RandomForestClassifier(random_state=42),
                "SVM Classifier": LinearSVC(max_iter=10000, random_state=42),
                "XGBClassifier": XGBClassifier( eval_metric='logloss', random_state=42),
                "CatBoost Classifier": CatBoostClassifier(verbose=False, random_seed=42),
                "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
                "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
            }


            best_model_name = None
            best_model = None
            best_score = 0
            results = []

            logging.info("🚀 Training all models...")

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds)
                rec = recall_score(y_test, preds)
                f1 = f1_score(y_test, preds)

                results.append([name, acc, prec, rec, f1])

                logging.info(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}")

                if acc > best_score:
                    best_score = acc
                    best_model_name = name
                    best_model = model

            results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
            results_path = os.path.join("artifacts", "models", "model_results.csv")
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            results_df.to_csv(results_path, index=False)

            logging.info(f" All model results saved to: {results_path}")
            logging.info(f" Best Model: {best_model_name} with Accuracy={best_score:.4f}")

            save_object(file_path=self.config.trained_model_file_path, obj=best_model)

            return best_model_name, best_score, self.config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer()
    best_model, score, path = trainer.initiate_model_training()
    print(f"Best Model: {best_model} | Accuracy: {score:.4f}")
    print(f"Model saved at: {path}")
