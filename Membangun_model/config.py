import os
# initialize global constants
feature_cols = ['region', 'category', 'parameter', 'mode', 'powertrain', 'year']
target_col = 'value_scaled'
dataset_path = 'IEA_Global_EV_Data_2024_cleaned.csv' 

# initialize DagsHub parameters
dagshub_repo_owner='nbintang'
dagshub_repo_name='IEA_EV_Prediction_MLflow_DagsHub'
dagshub_token = os.getenv('DAGSHUB_TOKEN')
dagshub_repo_url = f'https://dagshub.com/{dagshub_repo_owner}/{dagshub_repo_name}.mlflow'

# initialize model constants
experiment_name='IEA_EV_Prediction_Basic'
model_autolog_name = "RandomForest_Autolog"

# initialize model tuning constants
experiment_tuning_name = "IEA_EV_Prediction_DagsHub"
model_tuning_type = "RandomForestRegressor"
model_tuning_storage = "DagsHub"
model_tuning_name = "RandomForest_DagsHub"