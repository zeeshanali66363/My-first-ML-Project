import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This Function is used to handle missing values, standarize numerical features, and to encode categorical features
        '''

        try:
            numerical_columns= ['reading score', 'writing score']
            categorical_columns= ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder())
                ]
            )
            logging.info("Numerical columns Standarized and Categorical pipelines seperated.")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipline", num_pipeline, numerical_columns),
                    ('categorical_pipeline', categorical_pipeline, categorical_columns)
                ]
            )

            logging.info("ColumnTransformer obj created to Standarize Numerical and encode Categorical features.")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Trianing and testing data read")

            logging.info("Obtaining proprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name= "math score"
            numerical_columns = ['reading score', 'writing score']

            X_train_df=train_df.drop(columns=[target_column_name])
            y_train_df=train_df[target_column_name]

            X_test_df=test_df.drop(columns=[target_column_name])
            y_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing on train and test dataset.")

            X_train_df_arr=preprocessing_obj.fit_transform(X_train_df)
            X_test_df_arr=preprocessing_obj.transform(X_test_df)

            train_arr= np.c_[X_train_df_arr, np.array(X_train_df)]
            test_arr= np.c_[X_test_df_arr, np.array(X_test_df)]


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)
        
