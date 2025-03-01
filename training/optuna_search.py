import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your modules.
import handtex.symbol_relations as sr
from training.data_loader import (
    StrokeDataset,
    DataSplit,
    recalculate_frequencies,
    build_stroke_cache,
)
from training.hyperparameters import db_path, batch_size, num_epochs
from handtex.detector.image_gen import image_size

# Device configuration.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare symbol data and label encoder.
symbol_data = sr.SymbolData()
num_classes = len(symbol_data.leaders)
label_encoder = LabelEncoder()
label_encoder.fit(symbol_data.leaders)

# Preprocess and build stroke cache.
recalculate_frequencies()
stroke_cache = build_stroke_cache(db_path)

class_limit_factor = 20
seed = 0


split_percentages = {
    DataSplit.TRAIN: 50,
    DataSplit.VALIDATION: 20,
    DataSplit.TEST: 30,
}
assert sum(split_percentages.values()) == 100, "Split points must sum to 100"


# Create training, validation, and test datasets.
train_dataset = StrokeDataset(
    db_path,
    symbol_data,
    image_size,
    label_encoder,
    random_seed=seed,
    split=DataSplit.TRAIN,
    split_percentages=split_percentages,
    class_limit=5 * class_limit_factor,
    stroke_cache=stroke_cache,
)
validation_dataset = StrokeDataset(
    db_path,
    symbol_data,
    image_size,
    label_encoder,
    random_seed=seed,
    split=DataSplit.VALIDATION,
    split_percentages=split_percentages,
    class_limit=2 * class_limit_factor,
    stroke_cache=stroke_cache,
)
test_dataset = StrokeDataset(
    db_path,
    symbol_data,
    image_size,
    label_encoder,
    random_seed=seed,
    split=DataSplit.TEST,
    split_percentages=split_percentages,
    class_limit=3 * class_limit_factor,
    stroke_cache=stroke_cache,
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

###############################################################################
# Define a dynamic CNN model function for NAS/hyperparameter search.
###############################################################################


class DynamicCNN(nn.Module):
    def __init__(self, num_classes: int, image_size: int, trial):
        """
        Build a CNN with a flexible number of convolutional layers (4–6).
        Each conv layer may optionally include dropout and pooling.
        Ends with a single fully connected layer.
        """
        super(DynamicCNN, self).__init__()
        self.image_size = image_size
        layers = []
        in_channels = 1  # assuming grayscale input

        # Determine number of conv layers (between 4 and 6)
        n_conv_layers = trial.suggest_int("n_conv_layers", 4, 6)

        # For the first conv layer, choose number of output channels.
        out_channels = trial.suggest_int("channels_0", 8, 32)
        for i in range(n_conv_layers):
            if i > 0:
                # For subsequent layers, allow increasing the number of channels.
                out_channels = trial.suggest_int(f"channels_{i}", out_channels, out_channels * 2)
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            bn = nn.BatchNorm2d(out_channels)
            relu = nn.ReLU(inplace=True)
            layers.extend([conv, bn, relu])

            # Optionally add dropout
            use_dropout = trial.suggest_categorical(f"use_dropout_{i}", [True, False])
            if use_dropout:
                dropout_rate = trial.suggest_float(f"dropout_rate_{i}", 0.1, 0.5)
                layers.append(nn.Dropout2d(dropout_rate))

            # Make pooling more likely by biasing choices (2/3 chance to pool)
            use_pooling = trial.suggest_categorical(f"use_pooling_{i}", [True, True, False])
            if use_pooling:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        # Global adaptive pooling to (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Single fully connected layer mapping from final channels to num_classes.
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


###############################################################################
# Define the objective function for Optuna.
###############################################################################
def objective(trial: optuna.trial.Trial) -> float:
    # Suggest optimizer hyperparameters.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # Build the model with the current trial's suggested architecture.
    model = DynamicCNN(num_classes=num_classes, image_size=image_size, trial=trial).to(device)

    # Check parameter count; prune if it exceeds 836,456.
    num_params = sum(p.numel() for p in model.parameters())
    if num_params > 850_000:
        raise optuna.exceptions.TrialPruned(f"Too many parameters: {num_params}")

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Learning Rate Scheduler ---
    # Here we use a StepLR scheduler with hyperparameters tuned by Optuna.
    step_size = trial.suggest_int("step_size", 3, 10)
    gamma = trial.suggest_float("gamma", 0.1, 0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    criterion = nn.CrossEntropyLoss()

    print(f"\nTrial {trial.number} started with {trial.params}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for data, targets in tqdm(
            train_dataloader, desc=f"Trial {trial.number} Epoch {epoch+1}/{num_epochs}"
        ):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)

        # Step the learning rate scheduler at the end of each epoch.
        scheduler.step()

        # Validation phase.
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in validation_dataloader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        val_loss = val_running_loss / len(validation_dataset)
        val_accuracy = 100 * correct / total

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


###############################################################################
# Run the Optuna study.
###############################################################################
if __name__ == "__main__":
    # Create or load a persistent study so you can interrupt with Ctrl+C and resume later.
    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
    )

    try:
        study.optimize(objective)
    except KeyboardInterrupt:
        print("Optimization interrupted by user. You can resume by re-running the script.")

    # Rebuild and evaluate the best model from the study.
    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Validation Loss: {best_trial.value:.4f}")
    print("  Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    ###############################################################################
    # Rebuild the final model using the best trial parameters.
    #
    # We create a DummyTrial object that returns exactly the saved values.
    ###############################################################################
    class DummyTrial:
        def __init__(self, params):
            self.params = params

        def suggest_int(self, name, low, high):
            # Expect the parameter to exist; otherwise, raise an error.
            if name in self.params:
                return int(self.params[name])
            raise KeyError(f"Missing integer hyperparameter: {name}")

        def suggest_float(self, name, low, high, log=False):
            if name in self.params:
                return float(self.params[name])
            raise KeyError(f"Missing float hyperparameter: {name}")

        def suggest_categorical(self, name, choices):
            if name in self.params:
                return self.params[name]
            # Fall back to the first choice if not present.
            return choices[0]

    dummy_trial = DummyTrial(best_trial.params)
    final_model = DynamicCNN(num_classes=num_classes, image_size=image_size, trial=dummy_trial).to(
        device
    )

    # Optionally, if you wish to retrain on a combined training+validation set,
    # do that here before evaluation. Otherwise, we directly evaluate on the test set.

    ###############################################################################
    # Final evaluation on the test set.
    ###############################################################################
    final_model.eval()
    criterion = nn.CrossEntropyLoss()
    test_running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in test_dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = final_model(data)
            loss = criterion(outputs, targets)
            test_running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)

    # Divide the total loss by the number of samples.
    test_loss = test_running_loss / len(test_dataset)
    print(f"Final Test Loss: {test_loss:.4f}")

#
# Validation Loss: 0.4935
#  Hyperparameters:
#    optimizer: Adam
#    lr: 0.00045306634736969894
#    weight_decay: 1.5907693530815136e-05
#    n_conv_layers: 6
#    channels_0: 31
#    use_dropout_0: True
#    dropout_rate_0: 0.390531541141081
#    use_pooling_0: False
#    channels_1: 58
#    use_dropout_1: False
#    use_pooling_1: True
#    channels_2: 115
#    use_dropout_2: False
#    use_pooling_2: False
#    channels_3: 226
#    use_dropout_3: False
#    use_pooling_3: False
#    channels_4: 226
#    use_dropout_4: False
#    use_pooling_4: True
#    channels_5: 226
#    use_dropout_5: False
#    use_pooling_5: False
#    step_size: 7
#    gamma: 0.6211629426770741


# Best trial:
#   Validation Loss: 0.3520
#   Hyperparameters:
#     optimizer: Adam
#     lr: 0.0005364683338857024
#     weight_decay: 1.4868633572534654e-06
#     n_conv_layers: 5
#     channels_0: 31
#     use_dropout_0: False
#     use_pooling_0: True
#     channels_1: 52
#     use_dropout_1: False
#     use_pooling_1: True
#     channels_2: 92
#     use_dropout_2: False
#     use_pooling_2: True
#     channels_3: 122
#     use_dropout_3: False
#     use_pooling_3: True
#     channels_4: 178
#     use_dropout_4: False
#     use_pooling_4: True
#     step_size: 3
#     gamma: 0.15235718695861067
# Final Test Loss: 7.4833
# {
#     "optimizer": "Adam",
#     "lr": 0.0005364683338857024,
#     "weight_decay": 1.4868633572534654e-06,
#     "n_conv_layers": 5,
#     "channels_0": 31,
#     "use_dropout_0": False,
#     "use_pooling_0": True,
#     "channels_1": 52,
#     "use_dropout_1": False,
#     "use_pooling_1": True,
#     "channels_2": 92,
#     "use_dropout_2": False,
#     "use_pooling_2": True,
#     "channels_3": 122,
#     "use_dropout_3": False,
#     "use_pooling_3": True,
#     "channels_4": 178,
#     "use_dropout_4": False,
#     "use_pooling_4": True,
#     "step_size": 3,
#     "gamma": 0.15235718695861067,
# }
