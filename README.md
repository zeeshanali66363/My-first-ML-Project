# My First ML Project

## Overview

This repository contains my first end-to-end machine learning project, built with guidance from Krishnaik06. The project predicts a student math score using demographic and academic features from a cleaned student performance dataset.

The project is implemented as a data pipeline with ingestion, transformation, model training, and a Flask web app for prediction.

## Key Features

- Data ingestion with train/test split
- Data transformation with imputation, scaling, and one-hot encoding
- Model training and hyperparameter optimization across multiple regressors
- Model serialization using `dill`
- Flask-based web app to predict math scores from user inputs
- Logging and custom exception handling for robustness

## Project Structure

- `app.py` - Flask app to serve the prediction interface
- `src/components/data_ingestion.py` - loads raw CSV data and creates train/test datasets
- `src/components/data_transformation.py` - builds preprocessing pipeline and transforms data
- `src/components/model_trainer.py` - trains and evaluates regression models
- `src/pipeline/predict_pipeline.py` - prediction logic and model loading
- `src/utils.py` - helper utilities for saving/loading objects and model evaluation
- `artifacts/` - pre-trained model and preprocessor files (`model.pkl`, `preprocessor.pkl`) committed to version control for deployment
- `notebook/` - exploratory data analysis and model training notebooks
- `templates/` - HTML templates for the Flask web interface
- `requirements.txt` - Python dependencies
- `setup.py` - package installation configuration

## Dataset

The project uses the `StudentsPerformance_cleaned.csv` dataset located in `notebook/data/`. It contains student profiles and scores:

- `gender`
- `race/ethnicity`
- `parental level of education`
- `lunch`
- `test preparation course`
- `reading score`
- `writing score`
- `math score` (target)

## How It Works

### Development (Training)
1. `DataIngestion` reads the cleaned dataset and saves raw, train, and test CSV files under `artifacts/`.
2. `DataTransformation` applies preprocessing:
   - median imputation and standard scaling for numeric features
   - most frequent imputation and one-hot encoding for categorical features
3. `ModelTrainer` evaluates several regressors and selects the best model using randomized search and R² scoring.
4. The best model and preprocessing pipeline are serialized to `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.

### Deployment (Prediction)
5. Pre-trained models are included in the repository (`artifacts/model.pkl` and `artifacts/preprocessor.pkl`).
6. `app.py` loads the pre-trained model and preprocessor, accepts user inputs from the web form, and returns a predicted math score.
7. No training occurs on the deployed application - predictions are instant.

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install the package:

```bash
pip install -e .
```

## Run the Flask App

```bash
python app.py
```

Open the browser at `http://127.0.0.1:5000/` to use the prediction form.

## Deployment Notes

- **Pre-trained Models**: The `artifacts/` folder contains pre-trained `model.pkl` and `preprocessor.pkl` files committed to version control. This ensures instant predictions without any training delay on deployment.
- **Production Ready**: When deployed, the Flask app loads the pre-trained model and serves predictions immediately.
- **To Retrain**: Delete the `artifacts/` folder, and on next run, the pipeline will automatically retrain the model.
- The Flask app is designed to accept form inputs for all features except `math score`, which the model predicts.
- Logging output is saved to `logs/` with timestamped log files.

## Acknowledgements

- Project developed as a first end-to-end ML pipeline with help from Krishnaik06.
- Inspired by machine learning project tutorials and best practices for production-style model pipelines.
