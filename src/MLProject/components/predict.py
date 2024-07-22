from MLProject import logger
from MLProject.entity.config_entity import PredictionConfig

from mlflow.artifacts import download_artifacts
from mlflow import MlflowClient
from mlflow import pyfunc
from pathlib import Path

import joblib

class Predict:
    def __init__(self, config: PredictionConfig):
        self.config = config

    def run(self, data: list) -> list:
        """predict the data with linear regression model
        
        Args:
            data (pd.Dataframe): input data to predict

        Raises:
            client_error: error when access mlflow to get deployed model
            download_error: error when download scaler from mlflow artifact
            load_error: scaler error
        
        Returns:
            y_predict: list type
        """
        try:
            logger.info("Set MLflow Client.")
            client = MlflowClient(tracking_uri=self.config.mlflow_tracking_uri)
            
            logger.info("Select the deployed model from MLflow.")
            selected_model = client.get_model_version_by_alias(
                self.config.mlflow_model_name, 
                self.config.mlflow_deploy_model_alias
            )
            
            logger.info("Get the deployed model run id.")
            selected_run_id = selected_model.run_id
        except Exception as client_error:
            logger.error(client_error)
            raise client_error
        
        root_dir = self.config.root_dir
        mlflow_scaler_model_path = self.config.mlflow_scaler_model_path
        scaler_model_path = Path(f"{root_dir}/{mlflow_scaler_model_path}")
        
        try:
            logger.info("Downloading scaler from MLflow's artifacts.")
            download_artifacts(
                run_id=selected_run_id,
                artifact_path=self.config.mlflow_scaler_model_path,
                dst_path=self.config.root_dir
            )
        except Exception as download_error:
            logger.error(download_error)
            raise download_error
        
        try:
            logger.info("Load the scaler model.")
            scaler = joblib.load(scaler_model_path)
            
            logger.info("Transform the data.")
            X_test_scaled = scaler.transform(data)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=data.columns)

        except Exception as load_error:
            logger.error(load_error)
            raise load_error
        
        logger.info("Predict the data.")
        loaded_model = pyfunc.load_model(model_uri=selected_model.source)
        y_predict = loaded_model.predict(X_test_scaled).tolist()
        
        return y_predict