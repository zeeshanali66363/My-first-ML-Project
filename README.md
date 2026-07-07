# My First ML Project

## Overview

This repository contains my first end-to-end machine learning project, built with guidance from Krishnaik06. The project predicts a student math score using demographic and academic features from a cleaned student performance dataset.

The project is implemented as a data pipeline with ingestion, transformation, model training, and a Flask web app for prediction.

## Key Features

# My First ML Project

## Overview

This repository is an end-to-end machine learning pipeline that predicts a student math score from demographic and academic features. It includes data ingestion, transformation, model training (for development), and a Flask web app for serving predictions.

## Key Features

- Reproducible data ingestion and train/test split
- Preprocessing pipeline with imputation, scaling, and encoding
- Model training and selection across multiple regressors
- Model and preprocessor serialization using `dill`
- Flask web app for live predictions
- Logging and custom exception handling

## Project Structure

- `app.py` - Flask app to serve the prediction interface
- `src/components/data_ingestion.py` - loads raw CSV data and creates train/test datasets
- `src/components/data_transformation.py` - builds preprocessing pipeline and transforms data
- `src/components/model_trainer.py` - trains and evaluates regression models
- `src/pipeline/predict_pipeline.py` - loads pre-trained artifacts and performs prediction
- `src/utils.py` - helper utilities for saving/loading objects and model evaluation
- `artifacts/` - pre-trained model and preprocessor files (`model.pkl`, `preprocessor.pkl`) and CSV snapshots
- `notebook/` - exploratory data analysis and training notebooks
- `templates/` - HTML templates for the Flask web interface
- `requirements.txt` - Python dependencies
- `setup.py` - package installation configuration

## Dataset

The cleaned dataset used for development is `StudentsPerformance_cleaned.csv` located at `notebook/data/StudentsPerformance_cleaned.csv` (the code reads it using a workspace-relative path; Windows path separators are accepted).

Fields:

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
2. `DataTransformation` builds and applies the preprocessing pipeline (imputation, scaling, encoding).
3. `ModelTrainer` evaluates multiple regressors and saves the best model and preprocessor to `artifacts/`.

### Deployment (Prediction)
1. The repository includes pre-trained `artifacts/model.pkl` and `artifacts/preprocessor.pkl` so the app can serve predictions instantly.
2. `app.py` loads the pre-trained model and preprocessor, accepts user inputs via the web form, and returns a predicted math score.
3. No training occurs during prediction in the deployed application.

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install the package (optional):

```bash
pip install -e .
```

## Run the Flask App (local)

```bash
python app.py
```

Open the browser at `http://127.0.0.1:5000/` to use the prediction form.

## Deployment Notes

- **Pre-trained Models**: This repository already contains trained artifacts (`artifacts/model.pkl` and `artifacts/preprocessor.pkl`). The Flask app loads these on startup for immediate predictions.
- **If you want to deploy with fresh training**: delete the `artifacts/` folder; on next run the pipeline will recreate `artifacts/` and retrain the model (development only).
- **Committing artifacts**: If you want artifacts tracked in git, ensure `artifacts/` is not ignored in your `.gitignore`. For large binary models consider using Git LFS:

```bash
git lfs install
git lfs track "artifacts/*.pkl"
```

- **Dataset path**: The ingestion code reads `notebook/data/StudentsPerformance_cleaned.csv` (relative path). Ensure that file exists when retraining.

## Notes

- The Flask app accepts form inputs for the features listed above and returns the predicted `math score`.
- Logs are saved under `logs/` with timestamped filenames.
- If you want to retrain the model you can uncomment the code bellow in src/components/data_ingestion.py and run it.

## Acknowledgements

- Project developed as a learning exercise following tutorials and best practices for ML pipelines.
