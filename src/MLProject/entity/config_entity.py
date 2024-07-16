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
    output_train_path: Path
    output_test_path: Path
    params_test_size: float

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    input_train_path: Path
    input_test_path: Path
    scaled_train_path: Path
    scaled_test_path: Path
    model_dir: Path
    scaler_model_path: Path

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    input_train_path: Path
    output_train_path: Path
    scaled_train_path: Path
    model_path: Path
    params_max_iter: int
    params_solver: str
    params_n_jobs: int

@dataclass(frozen=True)
class TrainEvaluationConfig:
    root_dir: Path
    input_train_path: Path
    input_test_path: Path
    output_train_path: Path
    output_test_path: Path
    model_path: Path
    score_path: Path
    mlflow_dataset_path: Path
    mlflow_tracking_uri: str
    mlflow_exp_name: str
    mlflow_run_name: str