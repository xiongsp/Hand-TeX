from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
import numpy as np

from handtex.detector.image_gen import IMAGE_SIZE
import handtex.detector.image_gen as ig
from handtex.detector.model import CNN


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
    model = CNN(num_classes=len(label_decoder), image_size=IMAGE_SIZE)
    model.load_state_dict(load_file(model_path))
    model.to(device)
    model.eval()  # Set to evaluation mode
    # num_params = sum(p.numel() for p in model.parameters())
    # print(f"Model has {num_params} parameters")

    return model, label_decoder


def predict_strokes(
    strokes: list[list[tuple[int, int]]],
    model: nn.Module,
    label_decoder: dict[int, str],
    max_results: int = 20,
) -> list[tuple[str, float]]:
    """
    Predict the class of a given image using the trained model.

    :param strokes: The strokes of the image to predict.
    :param model: The trained neural network model.
    :param label_decoder: The label encoder used to encode the labels.
    :param max_results: The maximum number of results to return.
    :return: A list of tuples containing the predicted label and confidence score
    """
    tensor = ig.tensorize_strokes(strokes, IMAGE_SIZE)
    return predict(tensor, model, label_decoder, max_results)


def predict_image(
    image_data: np.ndarray,
    model: nn.Module,
    label_decoder: dict[int, str],
    max_results: int = 20,
) -> list[tuple[str, float]]:
    """
    Predict the class of a given image using the trained model.
    The image must be an 8-bit grayscale image.
    The background should be white and the symbol black.

    :param image_data: The image data to predict.
    :param model: The trained neural network model.
    :param label_decoder: The label encoder used to encode the labels.
    :param max_results: The maximum number of results to return.
    :return: A list of tuples containing the predicted label and confidence score
    """
    # Turn the array into a tensor
    assert image_data.dtype == np.uint8
    assert image_data.ndim == 2
    assert image_data.shape == (IMAGE_SIZE, IMAGE_SIZE)
    tensor = torch.from_numpy(image_data).unsqueeze(0).unsqueeze(0).float()
    return predict(tensor, model, label_decoder, max_results)


def predict(
    tensor: torch.Tensor,
    model: nn.Module,
    label_decoder: dict[int, str],
    max_results: int = 20,
) -> list[tuple[str, float]]:
    """
    Predict the class of a given image using the trained model.

    :param tensor: The tensor of the image to predict.
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
                # break
                pass
            final_results.append((label_decoder[label], confidence))
        return final_results
