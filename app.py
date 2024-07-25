import os
import logging

from MLProject.config.configuration import ConfigurationManager
from MLProject.components.predict import Predict

from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = FastAPI()
config = ConfigurationManager()

class Item(BaseModel):
    columns: List[str]
    data : List[List[float]]

@app.get("/")
def read_root():
    return {"message": "Up!"}

@app.post("/predict")
def predict(item: Item):
    logging.info("Access data")
    columns = item.columns
    data = item.data

    logging.info("Load configuration.")
    data = pd.DataFrame(data, columns=columns)
    predict_config = config.get_prediction_config()
    predict = Predict(config=predict_config)
    
    logging.info("Make prediction.")
    result = predict.run(data)
    
    return result