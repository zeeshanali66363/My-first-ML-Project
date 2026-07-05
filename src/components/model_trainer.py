# importing Machine learning models

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# trees
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

# More Libraries
import sys, os
from src.exception import CustomException
from src.logger import logging  
from src.utils import save_object, evaluate_models
from dataclasses import dataclass

# Metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        try:
            logging.info("Assigning input and output variables to both train and test data")

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models={
                "Linear Regression": LinearRegression(),
                "SVR": SVR(),   
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "KNN": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
                }
            
            params = {
                "Linear Regression": {},
                "SVR": {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
                "Lasso": {"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10], "max_iter": [1000, 5000]},
                "Ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
                "KNN": {"n_neighbors": [3,5,7,9], "weights": ["uniform","distance"], "p": [1,2]},
                "Decision Tree": {"max_depth": [None, 5, 10, 20], "min_samples_split": [2,5,10], "min_samples_leaf": [1,2,4]},
                "Random Forest": {"n_estimators": [50,100,200], "max_depth": [None,5,10], "min_samples_split": [2,5]},
                "Gradient Boosting": {"n_estimators": [100,200], "learning_rate": [0.01,0.1], "max_depth": [3,5]},
                "XGBRegressor": {"n_estimators": [100,200], "learning_rate": [0.01,0.1], "max_depth": [3,5]},
                "AdaBoost Regressor": {"n_estimators": [50,100,200], "learning_rate": [0.01,0.1,1]}
            }

            logging.info(f"Hyperparameter grids prepared for models: {list(params.keys())}")

            model_report_on_train_data, model_report_on_test_data=evaluate_models(X_train, y_train, X_test, y_test, models, params=params)
            logging.info("Completed hyperparameter tuning and evaluation for all models.")
            logging.info("Model evaluation completed on both train and test data")

            gaps = [abs(train_score - test_score)
                    for train_score, test_score in zip(
                    model_report_on_train_data.values(),
                    model_report_on_test_data.values())]
            
            models_with_less_than_2_gap = {model_name: gap for model_name, gap in zip(models.keys(), gaps) if gap < 0.02}

            logging.info(f"Models that have less than 2% gap in both training and testing data: {models_with_less_than_2_gap}")

            maximum_r2_score_on_test_with_less_than_2_gap = max(list(models_with_less_than_2_gap))
            best_score=model_report_on_test_data[maximum_r2_score_on_test_with_less_than_2_gap]
            
            best_model_name=[model for model in models_with_less_than_2_gap.keys() if model_report_on_test_data[model] == best_score][0]

            logging.info(f"The best model found is {best_model_name} with {best_score}")
            
                

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=models[best_model_name]
            )
            logging.info(f"Saved best model '{best_model_name}' to {self.model_trainer_config.trained_model_file_path}")
          
            return best_score
        except Exception as e:
            raise CustomException(e, sys)




