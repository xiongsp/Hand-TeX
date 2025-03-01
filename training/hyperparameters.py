# Optimized with Optuna.
learning_rate = 0.0005364683338857024
weight_decay = 1.4868633572534654e-06
step_size = 3
gamma = 0.15235718695861067
batch_size = 64
num_epochs = 8
db_path = "database/handtex.db"

# With 1024 nodes for fully connected layer, 10 epochs:
# Training Loss: 13.9550, Training Accuracy: 93.18%
# Validation Loss: 9.9632, Validation Accuracy: 94.94%
# Results felt very bad.

# With 2048 nodes for fully connected layer, 10 epochs:
# Training Loss: 13.7973, Training Accuracy: 93.23%
# Validation Loss: 10.0225, Validation Accuracy: 94.91%

# With 4096 nodes for fully connected layer and a 4th convolutional layer, 10 epochs:
# Training Loss: 13.8536, Training Accuracy: 93.51%
# Validation Loss: 10.6074, Validation Accuracy: 95.14%
# Epoch 20:
# Training Loss: 10.6421, Training Accuracy: 94.76%
# Validation Loss: 9.0769, Validation Accuracy: 95.70%

# Training Loss: 13.9664, Training Accuracy: 93.09%
# Validation Loss: 12.2665, Validation Accuracy: 93.89%

# 12 epochs:
# Training Loss: 12.3984, Training Accuracy: 94.48%
# Validation Loss: 10.2381, Validation Accuracy: 95.72%

# 18 epochs:
# Training Loss: 9.0018, Training Accuracy: 95.92%
# Validation Loss: 9.2644, Validation Accuracy: 96.31%

# Noise deformation augmentation:

# Simplified model 10 epochs:
# Validation accuracy: 85.31%

# 3 epochs:
# Actually pretty nice.
# Training Loss: 38.9639, Training Accuracy: 84.40%
# Validation Loss: 45.1656, Validation Accuracy: 83.03%

# Deep model 10 epochs:
# Training Loss: 7.7815, Training Accuracy: 96.63%
# Validation Loss: 38.0330, Validation Accuracy: 89.54%

# Training Loss: 6.7798, Training Accuracy: 97.03%
# Validation Loss: 27.7906, Validation Accuracy: 90.05%

# Optuna best
# Training Loss: 8.7202, Training Accuracy: 96.35%
# Validation Loss: 27.8282, Validation Accuracy: 90.06%
# Test Loss: 27.7608, Test Accuracy: 90.27%, Test F1 (macro): 0.9016
# Checkpoint saved at best_model_checkpoint.chkpt

# Optuna best with more training data
# Training Loss: 5.3208, Training Accuracy: 97.73%
# Validation Loss: 20.4027, Validation Accuracy: 92.00%
# Test Loss: 19.1761, Test Accuracy: 92.09%, Test F1 (macro): 0.9196
