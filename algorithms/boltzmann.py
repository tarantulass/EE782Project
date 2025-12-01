"""
Restricted Boltzmann Machine for Patch Antenna Inverse Design (Regression)
THE FOLLOWING TAKES INTO ACCOUNT THE USUAL DESIGNER PRACTICES WHERE CERTAIN 
PARAMETERS ARE KNOWN AND OTHERS ARE OBTAINED BY FEM USING HFSS EM simulations
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional
import os,sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
### local imports
from utils.logmodule import logsetup

logger = logsetup("results/boltzmann/boltzmann.log")

# Gaussian RBM for continuous values (regression)
class GaussianRBM():
    def __init__(self, visible, hidden):
        self.W = torch.randn(hidden, visible) * 0.01
        self.h_bias = torch.zeros(1, hidden)
        self.v_bias = torch.zeros(1, visible)
        
    def sample_hidden(self, v):
        """Probability that hidden node is activated given visible nodes"""
        v_dotW = torch.mm(v, self.W.T)
        activation = v_dotW + self.h_bias.expand_as(v_dotW)
        prob_h = torch.sigmoid(activation)
        return prob_h ,torch.bernoulli(prob_h)
    
    def sample_visible(self, h):
        """Sample visible nodes given hidden nodes (Gaussian for continuous)"""
        h_dotW = torch.mm(h, self.W)
        mean_v = h_dotW + self.v_bias.expand_as(h_dotW)
        return mean_v, mean_v
    
    def train(self, v0, vk, ph0, phk):
        """Contrastive Divergence update"""
        self.W += (torch.mm(v0.T, ph0) - torch.mm(vk.T, phk)).T * 0.001
        self.v_bias += torch.sum((v0 - vk), 0) * 0.001
        self.h_bias += torch.sum((ph0 - phk), 0) * 0.001


def load_and_preprocess_data(csv_path: str, 
                              feature_cols: List[str],
                              train_split: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor, StandardScaler]:
  
    df = pd.read_csv(csv_path)
    data = df[feature_cols].values
    
    # Normalize the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Convert to torch tensors
    data_tensor = torch.FloatTensor(data_normalized)
    
    # Split into train and test
    train_size = int(train_split * len(data_tensor))
    training_set = data_tensor[:train_size]
    testing_set = data_tensor[train_size:]
    
    logger.info(f"Training samples: {len(training_set)}")
    logger.info(f"Testing samples: {len(testing_set)}")
    
    return training_set, testing_set, scaler


def train_rbm(rbm: GaussianRBM, 
              training_set: torch.Tensor,
              epochs: int = 100,
              batch_size: int = 32,
              gibbs_steps: int = 10,
              log_interval: int = 10,
              logger: Optional[object] = None) -> GaussianRBM:
    
    log_func = logger.info if logger else print
    log_func("\n=== Training RBM ===")
    
    for epoch in range(1, epochs + 1):
        training_loss = 0
        s = 0
        
        for id in range(0, len(training_set) - batch_size, batch_size):
            vk = training_set[id:id + batch_size]
            v0 = training_set[id:id + batch_size]
            ph0, _ = rbm.sample_hidden(v0)
            
            # Gibbs sampling (k-step CD)
            for k in range(gibbs_steps):
                _, hk = rbm.sample_hidden(vk)
                _, vk = rbm.sample_visible(hk)
            
            phk, _ = rbm.sample_hidden(vk)
            rbm.train(v0, vk, ph0, phk)
            training_loss += torch.mean(torch.abs(v0 - vk))
            s += 1
        
        if epoch % log_interval == 0:
            log_func(f'Epoch {epoch}/{epochs} - Training Loss: {training_loss/s:.6f}')
    
    return rbm


def test_rbm(rbm: GaussianRBM,
             testing_set: torch.Tensor,
             scaler: StandardScaler,
             feature_cols: List[str],
             known_indices: List[int],
             unknown_indices: List[int],
             reconstruction_steps: int = 20,
             logger: Optional[object] = None) -> Dict:
  
    log_func = logger.info if logger else print
    log_func("\n=== Testing RBM with Missing Parameters ===")
    
    testing_loss = 0
    mae_per_feature = torch.zeros(len(unknown_indices))
    predictions_list = []
    actuals_list = []
    
    for id in range(len(testing_set)):
        v_test = testing_set[id:id + 1].clone()
        v_original = testing_set[id:id + 1].clone()
        
        # Mask unknown parameters with zeros initially
        v_input = v_test.clone()
        v_input[0, unknown_indices] = 0
        
        # Reconstruct through RBM multiple times for better prediction
        vk = v_input.clone()
        for k in range(reconstruction_steps):
            _, hk = rbm.sample_hidden(vk)
            _, vk = rbm.sample_visible(hk)
            # Keep known values fixed
            vk[0, known_indices] = v_input[0, known_indices]
        
        # Calculate error only on unknown parameters
        error = torch.abs(v_original[0, unknown_indices] - vk[0, unknown_indices])
        mae_per_feature += error
        testing_loss += torch.mean(error)
        
        # Store for denormalization
        predictions_list.append(vk[0].detach().numpy())
        actuals_list.append(v_original[0].detach().numpy())
    
    avg_testing_loss = testing_loss / len(testing_set)
    mae_per_feature /= len(testing_set)
    
    # Denormalize predictions to original scale
    predictions_denorm = scaler.inverse_transform(predictions_list)
    actuals_denorm = scaler.inverse_transform(actuals_list)
    
    # Calculate real-world MAE
    mae_real = np.mean(np.abs(predictions_denorm[:, unknown_indices] - 
                              actuals_denorm[:, unknown_indices]), axis=0)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((predictions_denorm[:, unknown_indices] - 
                           actuals_denorm[:, unknown_indices]) / 
                          (actuals_denorm[:, unknown_indices] + 1e-10)) * 100, axis=0)
    
    # Get parameter names for unknown indices
    param_names = [feature_cols[i] for i in unknown_indices]
    
    # Log results
    log_func(f'\nOverall MAE (normalized): {avg_testing_loss:.6f}')
    log_func('\nMAE per predicted parameter (normalized):')
    for i, name in enumerate(param_names):
        log_func(f'  {name}: {mae_per_feature[i]:.6f}')
    
    logger.info('Real-world MAE (original units)')
    for i, name in enumerate(param_names):
        logger.info(f'  {name}: {mae_real[i]:.6f}')

    logger.info(' MAPE (%) ')
    for i, name in enumerate(param_names):
        logger.info(f'  {name}: {mape[i]:.2f}%')

    # Show sample predictions
    logger.info(' Sample Predictions (first 3 test cases) ')
    for i in range(min(3, len(testing_set))):
        logger.info(f'\nTest Case {i+1}:')
        logger.info(f"  Freq_Hz (given): {actuals_denorm[i, 0]:.2e}")
        logger.info(f"  EpsilonR (given): {actuals_denorm[i, len(feature_cols)-1]:.2f}")
        logger.info('\n  Predicted vs Actual:')
        for j, idx in enumerate(unknown_indices):
            pred_val = predictions_denorm[i, idx]
            actual_val = actuals_denorm[i, idx]
            error_pct = abs(pred_val - actual_val) / (actual_val + 1e-10) * 100
            logger.info(f'    {feature_cols[idx]}: {pred_val:.6f} vs {actual_val:.6f} (error: {error_pct:.2f}%)')
    
    return {
        'predictions': predictions_denorm,
        'actuals': actuals_denorm,
        'mae_normalized': avg_testing_loss.item(),
        'mae_per_feature_normalized': mae_per_feature.numpy(),
        'mae_real': mae_real,
        'mape': mape,
        'param_names': param_names
    }


def run_antenna_inverse_design(csv_path: str,
                                hidden_nodes: int = 50,
                                epochs: int = 100,
                                batch_size: int = 32,
                                train_split: float = 0.8,
                                logger: Optional[object] = None) -> Tuple[GaussianRBM, Dict]:
 
    # # Feature columns for insetfed
    # feature_cols = ['Freq_Hz', 'PatchLength', 'PatchWidth', 'PatchHeight', 
    #                 'Striplinewidth', 'FeedOffset', 'Notchlength', 'Notchwidth',
    #                 'Gndlength', 'Gndwidth', 'EpsilonR']
    # known_indices = [0, 3, 5, 10]
    # unknown_indices = [1, 2, 4, 6, 7, 8, 9]

    # Feature columns for coaxialfed
    feature_cols = ['Freq_Hz', 'PatchLength', 'PatchWidth', 'PatchHeight', 
                    'FeedPosX', 'FeedPosY', 'EpsilonR']
    known_indices = [0, 3, 6]
    unknown_indices = [1, 2, 4, 5]

    # Known and unknown parameter indices
    # Known: Freq_Hz (0), PatchHeight (3), FeedOffset (5), EpsilonR (10)
    # Unknown: PatchLength (1), PatchWidth (2), Striplinewidth (4), 
    #          Notchlength (6), Notchwidth (7), Gndlength (8), Gndwidth (9)
 
    
    # Load and preprocess data
    training_set, testing_set, scaler = load_and_preprocess_data(
        csv_path, feature_cols, train_split
    )
    
    # Initialize RBM
    visible = len(training_set[0])
    rbm = GaussianRBM(visible, hidden_nodes)
    logger.info(f"\nRBM Architecture: {visible} visible nodes, {hidden_nodes} hidden nodes")
    
    # Train RBM
    rbm = train_rbm(rbm, training_set, epochs, batch_size, logger=logger)
    
    # Test RBM
    results = test_rbm(rbm, testing_set, scaler, feature_cols, 
                       known_indices, unknown_indices, logger=logger)
    
    return rbm, results


# Example usage
if __name__ == "__main__":
    data_path = Path("datasets/patch_data_coaxial.csv")  
    logger.info("\n Running Antenna Inverse Design with RBM ")

    rbm_model, test_results = run_antenna_inverse_design(
        csv_path=data_path,
        hidden_nodes=70,
        epochs=600,
        batch_size=32,
        train_split=0.8,
        logger=logger
    )

    logger.info("\n=== Accessing Results ===")
    logger.info(f"MAE Real: {test_results['mae_real']}")
    logger.info(f"MAPE: {test_results['mape']}")
