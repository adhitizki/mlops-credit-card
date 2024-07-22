import os
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
from MLProject.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from MLProject.utils.common import read_yaml, create_directories
from MLProject.entity.config_entity import (DataIngestionSQLConfig,
                                            DataDumpConfig,
                                            DataPreprocessingConfig,
                                            TrainingConfig,
                                            TrainEvaluationConfig,
                                            PredictionConfig)

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
        data_ingest_config = self.config.data_ingestion

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
        ingest_config = self.config.data_ingestion
        dataset_params = self.params

        create_directories([dump_config.root_dir])

        config = DataDumpConfig(
            root_dir=dump_config.root_dir,
            data_path=ingest_config.data_path,
            input_train_path=dump_config.input_train_path,
            input_test_path=dump_config.input_test_path,
            input_valid_path=dump_config.input_valid_path,
            output_train_path=dump_config.output_train_path,
            output_test_path=dump_config.output_test_path,
            output_valid_path=dump_config.output_valid_path,
            params_test_size=dataset_params.TEST_SIZE,
            params_valid_size=dataset_params.VALID_SIZE
        )

        return config
    
    def get_preprocessing_data_config(self) -> DataPreprocessingConfig:
        """read preprocessing config file and store as config entity
        then apply the dataclasses
        
        Returns:
            config: PreprocessingConfig type
        """
        dump_config = self.config.dump_data
        scaler_config = self.config.scale_data
        train_config = self.config.train_model

        create_directories([scaler_config.root_dir, train_config.root_dir])

        config = DataPreprocessingConfig(
            root_dir=scaler_config.root_dir,
            input_train_path=Path(dump_config.input_train_path),
            input_test_path=Path(dump_config.input_test_path),
            input_valid_path=Path(dump_config.input_valid_path),
            scaled_train_path=Path(scaler_config.scaled_train_path),
            scaled_test_path=Path(scaler_config.scaled_test_path),
            scaled_valid_path=Path(scaler_config.scaled_valid_path),
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
        scaler_config = self.config.scale_data
        train_config = self.config.train_model
        train_params = self.params

        config = TrainingConfig(
            input_train_path=Path(data_dump_config.input_train_path),
            output_train_path=Path(data_dump_config.output_train_path),
            output_test_path=Path(data_dump_config.output_test_path),
            scaled_train_path=Path(scaler_config.scaled_train_path),
            scaled_test_path=Path(scaler_config.scaled_test_path),
            model_path=Path(train_config.model_path),
            params_C=train_params.C,
            params_solver=train_params.SOLVER,
            params_n_trials=train_params.N_TRIALS,
        )

        return config
    
    def get_train_eval_config(self) -> TrainEvaluationConfig:
        """read training evaluation config file and store as 
        config entity then apply the dataclasses
        
        Returns:
            config: TrainEvaluationConfig type
        """
        data_dump_config = self.config.dump_data
        scaler_config = self.config.scale_data
        train_config = self.config.train_model
        eval_config = self.config.evaluation

        create_directories([eval_config.root_dir])

        config = TrainEvaluationConfig(
            root_dir=eval_config.root_dir,
            input_train_path=Path(data_dump_config.input_train_path),
            input_test_path=Path(data_dump_config.input_test_path),
            input_valid_path=Path(data_dump_config.input_valid_path),
            output_train_path=Path(data_dump_config.output_train_path),
            output_test_path=Path(data_dump_config.output_test_path),
            output_valid_path=Path(data_dump_config.output_valid_path),
            scaled_train_path=Path(scaler_config.scaled_train_path),
            scaled_test_path=Path(scaler_config.scaled_test_path),
            scaled_valid_path=Path(scaler_config.scaled_valid_path),
            scaler_model_path=Path(scaler_config.scaler_model_path),
            model_path=Path(train_config.model_path),
            train_score_path=Path(eval_config.train_score_path),
            test_score_path=Path(eval_config.test_score_path),
            valid_score_path=Path(eval_config.valid_score_path),
            mlflow_dataset_path=Path(eval_config.mlflow_dataset_path),
            mlflow_dataset_column=eval_config.mlflow_dataset_column,
            minio_endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'],
            minio_access_key_id=os.environ['MINIO_ACCESS_KEY'],
            minio_secret_access_key=os.environ['MINIO_SECRET_ACCESS_KEY'],
            mlflow_tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
            mlflow_exp_name=eval_config.mlflow_exp_name,
            mlflow_dataset_bucket=os.environ["PROJECT_BUCKET"],
            mlflow_run_name=eval_config.mlflow_run_name
        )

        return config
    
    def get_prediction_config(self) -> PredictionConfig:
        """read training evaluation config file and store as 
        config entity then apply the dataclasses
        
        Returns:
            config: PredictionConfig type
        """
        predict_config = self.config.predict
        
        # for development (debug)
        dump_data_config = self.config.dump_data

        create_directories([predict_config.root_dir])

        config = PredictionConfig(
            root_dir=predict_config.root_dir,
            mlflow_tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
            mlflow_model_name=predict_config.mlflow_model_name,
            mlflow_deploy_model_alias=os.environ["MLFLOW_DEPLOY_MODEL_ALIAS"],
            mlflow_scaler_model_path=predict_config.mlflow_scaler_model_path,
            
            # for development (debug)
            input_valid_path=dump_data_config.input_valid_path,
            output_valid_path=dump_data_config.output_valid_path
        )

        return config