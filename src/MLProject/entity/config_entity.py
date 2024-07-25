from dataclasses import dataclass
from pathlib import Path

"""NOTE: Delete or replace any class as you need
and don't forget to import this class in
'../config/configuration.py' or 
'src/MLProject/config/configuration.py'
"""

@dataclass(frozen=True)
class DataIngestionSQLConfig:
    root_dir: Path
    source_URI: str
    data_table: str
    data_path: Path
    
@dataclass(frozen=True)
class DataDumpConfig:
    root_dir: Path
    data_path: Path
    input_train_path: Path
    input_test_path: Path
    input_valid_path: Path
    output_train_path: Path
    output_test_path: Path
    output_valid_path: Path
    params_test_size: float
    params_valid_size: float

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    input_train_path: Path
    input_test_path: Path
    input_valid_path: Path
    scaled_train_path: Path
    scaled_test_path: Path
    scaled_valid_path: Path
    model_dir: Path
    scaler_model_path: Path

@dataclass(frozen=True)
class TrainingConfig:
    input_train_path: Path
    output_train_path: Path
    output_test_path: Path
    scaled_train_path: Path
    scaled_test_path: Path
    model_path: Path
    params_C: list
    params_solver: list
    params_n_trials: list

@dataclass(frozen=True)
class TrainEvaluationConfig:
    root_dir: Path
    input_train_path: Path
    input_test_path: Path
    input_valid_path: Path
    output_train_path: Path
    output_test_path: Path
    output_valid_path: Path
    scaled_train_path: Path
    scaled_test_path: Path
    scaled_valid_path: Path
    scaler_model_path: Path
    model_path: Path
    train_score_path: Path
    test_score_path: Path
    valid_score_path: Path
    mlflow_dataset_path: Path
    mlflow_dataset_column: list
    minio_endpoint_url: str
    minio_access_key_id: str
    minio_secret_access_key: str
    mlflow_tracking_uri: str
    mlflow_exp_name: str
    mlflow_dataset_bucket: str
    mlflow_run_name: str

@dataclass(frozen=True)
class PredictionConfig:
    root_dir: Path
    mlflow_tracking_uri: str
    mlflow_model_name: str
    mlflow_deploy_model_alias: Path
    mlflow_scaler_model_path: Path
    
    # for development (debug)
    input_valid_path: Path
    output_valid_path: Path

@dataclass(frozen=True)
class UnitTestConfig:
    root_dir: Path
    mlflow_tracking_uri: str
    mlflow_model_name: str
    mlflow_deploy_model_alias: str
    mlflow_input_example_path: Path
    app_endpoint: str