from MLProject import logger
from MLProject.entity.config_entity import (DataDumpConfig, 
                                            DataPreprocessingConfig)

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DumpData:
    def __init__(self, config: DataDumpConfig):
        self.config = config

    def dump_data(self) -> None:
        """dump the splited dataset to data training and testing
        """
        logger.info(f"Read reviews file.")
        dataset = pd.read_csv(self.config.data_path)
        dataset = dataset.drop(columns=['id']).copy()
        dataset.dropna(inplace=True)
        
        logger.info(f"Split data file to data train and test.")
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.drop(columns=["Class"]), 
            dataset["Class"], 
            test_size=self.config.params_test_size,
            stratify=dataset["Class"],
        )
        
        # NOTE: data save as pandas dataframe and y as series
        logger.info(f"Dump data train into {self.config.root_dir} directory.")
        X_train.to_pickle(self.config.input_train_path)
        X_test.to_pickle(self.config.input_test_path)
        
        # NOTE: data save as pandas dataframe and y as serie
        logger.info(f"Dump data test into {self.config.root_dir} directory.")
        y_train.to_pickle(self.config.output_train_path)
        y_test.to_pickle(self.config.output_test_path)

class Preprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def scaling_data(self) -> None:
        """scaling the splited dataset and dump vectorizer model
        """
        scaler = StandardScaler()
        
        logger.info(f"Load data train in {self.config.input_train_path}.")
        X_train = joblib.load(self.config.input_train_path)
        
        logger.info(f"Load data test in {self.config.input_test_path}.")
        X_test = joblib.load(self.config.input_test_path)
        
        logger.info(f"Vectorize the data.")
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"Dump the scaled data.")
        joblib.dump(X_train_scaled, self.config.scaled_train_path)
        joblib.dump(X_test_scaled, self.config.scaled_test_path)
        
        logger.info(f"Creating {self.config.model_dir} directory.")
        model_dir = str(self.config.model_dir)
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Save the scaler model.")
        joblib.dump(scaler, self.config.scaler_model_path)  
