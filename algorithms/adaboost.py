#### system import
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

#### local import
from utils.logmodule import logsetup
import config

def adaboost_regressor(data: Path, n_estimators=20, learning_rate=1.0):
    logger = logsetup()
    logger.info("Starting AdaBoost Regressor training...")

    #### loading the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        pd.read_csv(data).drop(columns=['Freq_Hz','Substrate']),
        pd.read_csv(data)['Freq_Hz'],
        test_size=0.3,
        random_state=42
    )

    #### scaling the target variable as it is in 10^9 order 
    y_train = y_train/1e9
    y_test = y_test/1e9

    #### model initialization and training
    model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(X_train, y_train)
    logger.info("AdaBoost Regressor training completed.")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    logger.info(f"MAE: {mae}")
    logger.info(f"MSE: {mse}")
    logger.info(f"RMSE: {rmse}")
    logger.info(f"R² Score: {r2}")

    #### Save metrics to text file
    results_dir = Path(config.ADABOOST_DIR)
    results_dir.mkdir(exist_ok=True)

    metrics_path = results_dir / "adaboost_metrics.txt"

    with open(metrics_path, "w") as f:
        f.write("AdaBoost Regressor Metrics\n")
        f.write("==========================\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"R² Score: {r2}\n")

    logger.info(f"Metrics saved to: {metrics_path}")

    #### saving the feature importance

    feature_importances = model.feature_importances_

    fi_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": feature_importances
    }).sort_values("importance", ascending=False)

    fi_path = results_dir / "adaboost_feature_importance.txt"

    with open(fi_path, "w") as f:
        f.write("AdaBoost Feature Importances\n")
        f.write("============================\n")
        for feature, importance in zip(fi_df["feature"], fi_df["importance"]):
            f.write(f"{feature}: {importance}\n")

    logger.info(f"Feature importances saved to: {fi_path}")


    return predictions, {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

if __name__ == "__main__":
    data_path = Path("datasets/Patch_data_insetfed.csv")  
    adaboost_regressor(data_path)
