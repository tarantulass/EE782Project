"""
Restricted Boltzmann Machine for Patch Antenna Inverse Design (Regression)
THE FOLLOWING TAKES INTO ACCOUNT THE USUAL DESIGNER PRACTICES WHERE CERTAIN PARAMETERS ARE KNOWN AND OTHERS ARE OBTAINED BY FEM USING HFSS EM simulations
"""

from camelot import logger
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


### local imports
from utils.logmodule import logsetup


# Load your patch antenna dataset
# Replace with your actual CSV file path
df = pd.read_csv('patch_antenna_data.csv')

# Feature columns
feature_cols = ['Freq_Hz', 'PatchLength', 'PatchWidth', 'PatchHeight', 
                'Striplinewidth', 'FeedOffset', 'Notchlength', 'Notchwidth',
                'Gndlength', 'Gndwidth', 'EpsilonR']

# Separate features
data = df[feature_cols].values

# Normalize the data (important for regression RBM)
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Convert to torch tensors
data_tensor = torch.FloatTensor(data_normalized)

# Split into train and test
train_size = int(0.8 * len(data_tensor))
training_set = data_tensor[:train_size]
testing_set = data_tensor[train_size:]

print(f"Training samples: {len(training_set)}")
print(f"Testing samples: {len(testing_set)}")

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
        return prob_h, torch.bernoulli(prob_h)
    
    def sample_visible(self, h):
        """Sample visible nodes given hidden nodes (Gaussian for continuous)"""
        h_dotW = torch.mm(h, self.W)
        mean_v = h_dotW + self.v_bias.expand_as(h_dotW)
        # For Gaussian RBM, we return the mean directly (can add noise if needed)
        return mean_v, mean_v
    
    def train(self, v0, vk, ph0, phk):
        """Contrastive Divergence update"""
        self.W += (torch.mm(v0.T, ph0) - torch.mm(vk.T, phk)).T * 0.001
        self.v_bias += torch.sum((v0 - vk), 0) * 0.001
        self.h_bias += torch.sum((ph0 - phk), 0) * 0.001

# Initialize RBM
visible = len(training_set[0])  # 11 features
hidden = 50  # Hidden units to capture patterns
batch_size = 32
rbm = GaussianRBM(visible, hidden)

logger.info(f"\nRBM Architecture: {visible} visible nodes, {hidden} hidden nodes")

# Training the RBM
epochs = 100
logger.info("\n=== Training RBM ===")

for epoch in range(1, epochs + 1):
    training_loss = 0
    s = 0
    
    for id in range(0, len(training_set) - batch_size, batch_size):
        vk = training_set[id:id + batch_size]
        v0 = training_set[id:id + batch_size]
        ph0, _ = rbm.sample_hidden(v0)
        
        # Gibbs sampling (k-step CD)
        for k in range(10):
            _, hk = rbm.sample_hidden(vk)
            _, vk = rbm.sample_visible(hk)
        
        phk, _ = rbm.sample_hidden(vk)
        rbm.train(v0, vk, ph0, phk)
        training_loss += torch.mean(torch.abs(v0 - vk))
        s += 1
    
    if epoch % 10 == 0:
        logger.info(f'Epoch {epoch}/{epochs} - Training Loss: {training_loss/s:.6f}')

logger.info("\n=== Testing RBM with Missing Parameters ===")

# For testing: mask out parameters to predict
# Known: Freq_Hz (0), PatchHeight (3), FeedOffset (5), EpsilonR (10)
# Unknown: PatchLength (1), PatchWidth (2), Striplinewidth (4), 
#          Notchlength (6), Notchwidth (7), Gndlength (8), Gndwidth (9)

known_indices = [0, 3, 5, 10]
unknown_indices = [1, 2, 4, 6, 7, 8, 9]

testing_loss = 0
mae_per_feature = torch.zeros(len(unknown_indices))
predictions_list = []
actuals_list = []

for id in range(len(testing_set)):
    # Create masked input (known parameters only)
    v_test = testing_set[id:id + 1].clone()
    v_original = testing_set[id:id + 1].clone()
    
    # Mask unknown parameters with zeros initially
    v_input = v_test.clone()
    v_input[0, unknown_indices] = 0
    
    # Reconstruct through RBM multiple times for better prediction
    vk = v_input.clone()
    for k in range(20):  # More iterations for better reconstruction
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

logger.info(f'\nOverall MAE (normalized): {avg_testing_loss:.6f}')
logger.info('\nMAE per predicted parameter (normalized):')
param_names = ['PatchLength', 'PatchWidth', 'Striplinewidth', 
               'Notchlength', 'Notchwidth', 'Gndlength', 'Gndwidth']
for i, name in enumerate(param_names):
    logger.info(f'  {name}: {mae_per_feature[i]:.6f}')

# Denormalize predictions to original scale
predictions_denorm = scaler.inverse_transform(predictions_list)
actuals_denorm = scaler.inverse_transform(actuals_list)

# Calculate real-world MAE
mae_real = np.mean(np.abs(predictions_denorm[:, unknown_indices] - 
                          actuals_denorm[:, unknown_indices]), axis=0)

logger.info('\n=== Real-world MAE (original units) ===')
for i, name in enumerate(param_names):
    logger.info(f'  {name}: {mae_real[i]:.6f}')

# Calculate MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((predictions_denorm[:, unknown_indices] - 
                       actuals_denorm[:, unknown_indices]) / 
                      (actuals_denorm[:, unknown_indices] + 1e-10)) * 100, axis=0)

logger.info('\n=== MAPE (%) ===')
for i, name in enumerate(param_names):
    logger.info(f'  {name}: {mape[i]:.2f}%')

# Show sample predictions
logger.info('\n=== Sample Predictions (first 3 test cases) ===')
for i in range(min(3, len(testing_set))):
    logger.info(f'\nTest Case {i+1}:')
    logger.info(f"  Freq_Hz (given): {actuals_denorm[i, 0]:.2e}")
    logger.info(f"  EpsilonR (given): {actuals_denorm[i, 10]:.2f}")
    logger.info('\n  Predicted vs Actual:')
    for j, idx in enumerate(unknown_indices):
        pred_val = predictions_denorm[i, idx]
        actual_val = actuals_denorm[i, idx]
        error_pct = abs(pred_val - actual_val) / (actual_val + 1e-10) * 100
        logger.info(f'    {feature_cols[idx]}: {pred_val:.6f} vs {actual_val:.6f} (error: {error_pct:.2f}%)')