import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import os
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        
        try:
            if not os.path.exists('artifacts'):
                obj=DataIngestion()
                train_data, test_data=obj.initiate_data_ingestion()

                data_transformation=DataTransformation()
                train_arr, test_arr, preprocessor_path=data_transformation.initiate_data_transformation(train_data, test_data)  

                model_trainer=ModelTrainer()
                model_trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path)

            else:
                model_path = 'artifacts/model.pkl'
                preprocessor_path = 'artifacts/preprocessor.pkl'

                logging.info("Loading preprocessor and model for prediction")
                preprocessor = load_object(preprocessor_path)
                model = load_object(model_path)

                logging.info("Transforming features using preprocessor")
                transformed_features = preprocessor.transform(features)

                logging.info("Making predictions using the trained model")
                predictions = model.predict(transformed_features)

                logging.info(f"The data predicted Successfully.")
                return predictions
            
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int 
        ):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score]
            }

            logging.info("Custom data input dictionary created successfully.")
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)