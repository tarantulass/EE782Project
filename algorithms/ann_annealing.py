#### system import
import os,sys
# The sys.path manipulation is included based on your request, but requires 'utils/logmodule.py' and 'config.py' to run correctly outside of this environment.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import warnings

# Suppress ConvergenceWarning from MLPRegressor for clean output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

#### local import
# Assuming utils.logmodule and config are available in the target environment
try:
    from utils.logmodule import logsetup
    import config
except ImportError:
    # Define minimal mocks if running outside the intended environment
    class MockLogger:
        def info(self, msg): print(f"INFO - {msg}")
    def logsetup(): return MockLogger()
    class MockConfig:
        MLP_DIR = "results/ann_optimization"
    config = MockConfig()
    
def ann_regressor(data: Path, hidden_layer_sizes=(200, 200), max_iter=1000):
    """
    Trains an Artificial Neural Network (MLPRegressor) for **Inverse Design**.
    
    Inputs (X): Antenna performance metrics (Gain, VSWR, etc.)
    Outputs (Y): Antenna geometrical parameters (Sub_W, Patch_L, etc.)
    
    This model acts as the inverse surrogate function for the Simulated Annealing (SA) phase,
    mapping target performance requirements back to a physical design.
    """
    logger = logsetup()
    logger.info("Starting ANN Regressor (Inverse Surrogate Model) training...")

    #### Load and prepare the dataset
    df = pd.read_csv(data)
    
    # 1. Define Input (X - Performance Metrics) and Output (Y - Geometrical Parameters + Frequency)
    # This is the INVERSE configuration for design prediction.
    X_cols = ['Gain', 'Directivity', 'S11(dB)', 'Rad_eff', 'Total_eff', 'VSWR']
    Y_cols = ['Sub_W', 'Sub_L', 'Sub_H', 'Patch_W', 'Patch_L', 'Feed_W', 
              'Slot1_W', 'Slot1_L', 'Slot2_W', 'Slot2_L', 'Freq_GHz']
    
    # Filter out incomplete rows
    df = df.dropna(subset=X_cols + Y_cols)
    
    X = df[X_cols]
    Y = df[Y_cols].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    # NOTE: Log transformation is removed as geometrical outputs (Y) are typically linear and well-scaled.
    # The number of input features is now 6, and output features is now 11.
    
    # 2. Split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )

    # 3. Scaling (CRITICAL for ANN performance)
    # Scaling X (inputs) and Y (outputs) helps the ANN converge faster and more reliably.
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_Y = StandardScaler()
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    Y_test_scaled = scaler_Y.transform(Y_test)

    # 4. Model Initialization and Training (BPNN)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes, 
        activation='relu', 
        solver='adam', 
        max_iter=max_iter, 
        random_state=42
    )
    logger.info(f"Training MLP with {len(X_cols)} inputs and {len(Y_train.columns)} outputs...")
    model.fit(X_train_scaled, Y_train_scaled)
    logger.info("ANN (Inverse Surrogate Model) training completed.")

    # 5. Prediction and Inverse Scaling
    predictions_scaled = model.predict(X_test_scaled)
    # Directly inverse-transform the scaled predictions back to the original geometrical dimensions
    predictions_eval = pd.DataFrame(scaler_Y.inverse_transform(predictions_scaled), 
                                     columns=Y_cols, 
                                     index=Y_test.index)
    
    Y_test_eval = Y_test.copy()


    # 6. Evaluation (RMSE, R2 for each output - Geometrical Parameter)
    metrics = {}
    total_rmse = 0
    
    # Define the output columns for logging (now the 11 geometrical/frequency parameters)
    Original_Y_cols = Y_cols 

    for col in Original_Y_cols:
        # Calculate metrics using the original, unscaled target data
        mse = mean_squared_error(Y_test_eval[col], predictions_eval[col])
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_test_eval[col], predictions_eval[col])
        
        metrics[col] = {"RMSE": rmse, "R2": r2}
        total_rmse += rmse
        
        logger.info(f"--- Metrics for {col} ---")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"R² Score: {r2:.4f}")

    logger.info(f"Average RMSE across all outputs: {total_rmse / len(Original_Y_cols):.4f}")

    #### Save metrics to text file
    results_dir = Path(config.MLP_DIR)
    results_dir.mkdir(exist_ok=True, parents=True)

    metrics_path = results_dir / "ann_inverse_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("ANN Regressor (Inverse Surrogate Model) Metrics\n")
        f.write("=========================================\n")
        f.write(f"Model Structure: {hidden_layer_sizes}, Max Iter: {max_iter}\n")
        f.write("\nIndividual Output Metrics (Prediction Error for Geometry):\n")
        for col, m in metrics.items():
            f.write(f"  {col}:\n")
            f.write(f"    RMSE: {m['RMSE']:.4f}\n")
            f.write(f"    R² Score: {m['R2']:.4f}\n")
        f.write(f"\nAverage RMSE: {total_rmse / len(Original_Y_cols):.4f}\n")

    logger.info(f"Metrics saved to: {metrics_path}")

    # The trained model and scalers are what you need for the SA phase
    return model, scaler_X, scaler_Y, metrics


if __name__ == "__main__":
    data_path = Path("datasets/Antenna_s11.csv") 
    
    # The SA model depends heavily on the model's predictive quality, so we run a larger, 2-layer ANN.
    ann_regressor(data_path, hidden_layer_sizes=(200, 200), max_iter=1000) 
