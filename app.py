from MLProject.config.configuration import ConfigurationManager
from MLProject.components.predict import Predict
from MLProject import logger

import joblib

try:
    config = ConfigurationManager()
    predict_config = config.get_prediction_config()
    data = joblib.load(predict_config.input_valid_path)
    predict = Predict(config=predict_config)
    result = predict.run(data)
except Exception as e:
    logger.error(e)
    raise e