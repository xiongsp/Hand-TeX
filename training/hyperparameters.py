learning_rate = 0.001
batch_size = 64
num_epochs = 8
image_size = 64
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
