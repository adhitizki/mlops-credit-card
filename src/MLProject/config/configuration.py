import os

from pathlib import Path
from MLProject.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from MLProject.utils.common import read_yaml, create_directories
from MLProject.entity.config_entity import (DataIngestionSQLConfig,
                                            DataDumpConfig,
                                            DataPreprocessingConfig,
                                            TrainingConfig,
                                            TrainEvaluationConfig)

"""NOTE: Delete or replace any function as you need
and don't forget to import each class config from
'../config/configuration.py' or
'src/MLProject/config/configuration.py'
"""

class ConfigurationManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([Path(self.config.artifacts_root)])
    
    def get_data_ingestion_sql_config(self) -> DataIngestionSQLConfig:
        """read data ingestion config file and store as config entity
        then apply the dataclasses
        
        Returns:
            config: DataIngestionConfig type
        """
        data_ingest_config = self.config.ingest_from_sql

        create_directories([data_ingest_config.root_dir])

        config = DataIngestionSQLConfig(
            root_dir=data_ingest_config.root_dir,
            source_URI=os.environ["POSTGRES_URI"],
            data_table=data_ingest_config.data_table,
            data_path=Path(data_ingest_config.data_path),
        )

        return config
    
    def get_dump_data_config(self) -> DataDumpConfig:
        """read data dump config file and store as config entity
        then apply the dataclasses
        
        Returns:
            config: PreprocessingConfig type
        """
        dump_config = self.config.dump_data
        ingest_config = self.config.ingest_from_sql
        dataset_params = self.params

        create_directories([dump_config.root_dir])

        config = DataDumpConfig(
            root_dir=dump_config.root_dir,
            data_path=ingest_config.data_path,
            input_train_path=dump_config.input_train_path,
            input_test_path=dump_config.input_test_path,
            output_train_path=dump_config.output_train_path,
            output_test_path=dump_config.output_test_path,
            params_test_size=dataset_params.TEST_SIZE
        )

        return config
    
    def get_preprocessing_data_config(self) -> DataPreprocessingConfig:
        """read preprocessing config file and store as config entity
        then apply the dataclasses
        
        Returns:
            config: PreprocessingConfig type
        """
        dump_config = self.config.dump_data
        scaler_config = self.config.scaler_data
        train_config = self.config.train_model

        create_directories([scaler_config.root_dir])

        config = DataPreprocessingConfig(
            root_dir=scaler_config.root_dir,
            input_train_path=Path(dump_config.input_train_path),
            input_test_path=Path(dump_config.input_test_path),
            scaled_train_path=Path(scaler_config.scaled_train_path),
            scaled_test_path=Path(scaler_config.scaled_test_path),
            model_dir=train_config.root_dir,
            scaler_model_path=Path(scaler_config.scaler_model_path)
        )

        return config
    
    def get_training_config(self) -> TrainingConfig:
        """read training config file and store as config entity
        then apply the dataclasses
        
        Returns:
            config: TrainingConfig type
        """
        data_dump_config = self.config.dump_data
        scaler_config = self.config.scaler_data
        train_config = self.config.train_model
        train_params = self.params

        create_directories([train_config.root_dir])

        config = TrainingConfig(
            root_dir=train_config.root_dir,
            input_train_path=Path(data_dump_config.input_train_path),
            output_train_path=Path(data_dump_config.output_train_path),
            scaled_train_path=Path(scaler_config.scaled_train_path),
            model_path=Path(train_config.model_path),
            params_max_iter=train_params.MAX_ITER,
            params_solver=train_params.SOLVER,
            params_n_jobs=train_params.N_JOBS
        )

        return config
    
    def get_train_eval_config(self) -> TrainEvaluationConfig:
        """read evaluation config file and store as config entity
        then apply the dataclasses
        
        Returns:
            config: TrainEvaluationConfig type
        """
        data_ingestion_config = self.config.data_ingestion
        preprocessing_config = self.config.preprocessing
        training_config = self.config.train_model
        evaluation_config = self.config.evaluation
        
        create_directories([Path(evaluation_config.root_dir)])

        config = TrainEvaluationConfig(
            root_dir=Path(data_ingestion_config.root_dir),
            input_train_path=Path(data_ingestion_config.input_train_path),
            input_test_path=Path(data_ingestion_config.input_test_path),
            output_train_path=Path(preprocessing_config.output_train_path),
            output_test_path=Path(preprocessing_config.output_test_path),
            model_path=Path(training_config.model_path),
            score_path=Path(evaluation_config.score_path),
            mlflow_dataset_path=Path(evaluation_config.mlflow_dataset_path),
            mlflow_tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
            mlflow_exp_name=evaluation_config.mlflow_exp_name,
            mlflow_run_name=evaluation_config.mlflow_run_name
        )

        return config