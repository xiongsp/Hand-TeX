from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from training.hyperparameters import image_size
from training.model import CNN


def load_decoder(path: Path) -> dict[int, str]:
    with open(path, "r") as file:
        decoder = {i: symbol.strip() for i, symbol in enumerate(file)}
    return decoder


def load_model_and_decoder(model_path: Path, encodings_path: Path):
    # The decoder was created alongside the model, it knows
    # how many symbols to build the model with.

    # Load label encoder
    label_decoder = load_decoder(encodings_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model state
    model = CNN(num_classes=len(label_decoder), image_size=image_size)
    model.load_state_dict(load_file(model_path))
    model.to(device)
    model.eval()  # Set to evaluation mode

    return model, label_decoder


def predict(
    tensor: torch.Tensor, model: nn.Module, label_decoder: dict[int, str], max_results: int = 20
) -> list[tuple[str, float]]:
    """
    Predict the class of a given image using the trained model.

    :param tensor: The input image tensor (should be of shape [1, 1, image_size, image_size]).
    :param model: The trained neural network model.
    :param label_decoder: The label encoder used to encode the labels.
    :param max_results: The maximum number of results to return.
    :return: A list of tuples containing the predicted label and confidence score
    """
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        tensor = tensor.to(device)
        output = model(tensor)
        _, predicted_class = output.max(1)
        # Return the first 10 results and their confidences, sorted by confidence.
        # Example: [('A', 0.99), ('B', 0.01), ..., ('J', 0.00)]
        results = []
        for i in range(len(label_decoder)):
            confidence = F.softmax(output, dim=1)[0][i].item()
            results.append((i, confidence))
        results.sort(key=lambda x: x[1], reverse=True)
        # Prune the results to the top 20 or less, discarding results with tiny confidence.
        final_results = []
        for label, confidence in results:
            if len(final_results) >= max_results:
                break
            if final_results and confidence < 0.005:
                break
            final_results.append((label_decoder[label], confidence))
        return final_results
