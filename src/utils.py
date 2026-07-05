import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, params=None, n_iter_search=10, cv=3, random_state=42):

        try:

            report_on_train_data={}
            report_on_test_data={}

            for i, (name, model) in enumerate(models.items()):
                logging.info(f"Evaluating model: {name}")
                if params and name in params and params[name]:
                    param_dist = params[name]
                    try:
                        total_combinations = int(np.prod([len(v) for v in param_dist.values()]))
                    except Exception:
                        total_combinations = None
                    logging.info(f"Starting RandomizedSearchCV for {name} with param dist keys: {list(param_dist.keys())} and approx combos: {total_combinations}")

                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_dist,
                        n_iter=min(n_iter_search, max(1, int(total_combinations))) 

                        if total_combinations else n_iter_search,
                        cv=cv,
                        scoring='r2',
                        n_jobs=-1,
                        random_state=random_state,
                        verbose=0
                    )
                    
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    logging.info(f"Best params for {name}: {search.best_params_}")
                    logging.info(f"Best CV score for {name}: {search.best_score_}")
                else:
                    logging.info(f"No hyperparameter grid provided for {name}; fitting default estimator.")
                    model.fit(X_train, y_train)
                    best_model = model

                # replace the model in the dict with the fitted best model
                models[name] = best_model

                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                r2_train = r2_score(y_train, y_train_pred)
                r2_test = r2_score(y_test, y_test_pred)

                logging.info(f"{name} -> Train R2: {r2_train:.4f}, Test R2: {r2_test:.4f}")

                report_on_train_data[name] = r2_train
                report_on_test_data[name] = r2_test

            return report_on_train_data, report_on_test_data
        
        except Exception as e:
             raise CustomException(e, sys)


