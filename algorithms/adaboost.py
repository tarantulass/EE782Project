#### system import
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

#### local import
from utils.logmodule import logsetup

def adaboost_regressor(data: Path, n_estimators=50, learning_rate=1.0):
    logger = logsetup()
    logger.info("Starting AdaBoost Regressor training...")

    #### loading the dataset

    model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(X_train, y_train)
    logger.info("AdaBoost Regressor training completed.")
    predictions = model.predict(X_test)
    return predictions
