import torch
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
from training.image_gen import tensorize_strokes, load_decoder
from training.train import CNN, num_classes, device, image_size

from safetensors.torch import load_file
import handtex.utils as ut


# Load the model for inference
def load_model_and_decoder(model_path: Path, num_classes: int, encodings_path: Path):

    # Load model state
    model = CNN(num_classes=num_classes)
    model.load_state_dict(load_file(model_path))
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Load label encoder
    label_decoder = load_decoder(encodings_path)

    return model, label_decoder


# Inference function
def predict(
    tensor: torch.Tensor, model: nn.Module, label_decoder: dict[int, str]
) -> list[tuple[str, float]]:
    """
    Predict the class of a given image using the trained model.

    :param tensor: The input image tensor (should be of shape [1, 1, image_size, image_size]).
    :param model: The trained neural network model.
    :param label_decoder: The label encoder used to encode the labels.
    :return: A list of tuples containing the predicted label and confidence score
    """
    model.eval()  # Set model to evaluation mode
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
        # Prune the results to the top 10 or less, discarding results with tiny confidence.
        final_results = []
        for label, confidence in results:
            if final_results and confidence < 0.01:
                break
            final_results.append((label_decoder[label], confidence))
        return final_results


def main():
    # Test usage.
    import time

    start = time.time()
    loaded_model, label_decoder = load_model_and_decoder(
        ut.get_model_path(), num_classes, ut.get_encodings_path()
    )
    print("Model and label encoder loaded for inference.")
    print(f"Model loaded in {time.time() - start:.2f} seconds.")

    start = time.time()
    sum1_tensor = tensorize_strokes(sum1, image_size)
    sum2_tensor = tensorize_strokes(sum2, image_size)
    alpha_tensor = tensorize_strokes(alpha, image_size)
    neq1_tensor = tensorize_strokes(neq1, image_size)
    neq2_tensor = tensorize_strokes(neq2, image_size)
    print(f"Images tensorized in {time.time() - start:.2f} seconds.")

    start = time.time()
    sum1_prediction = predict(sum1_tensor, loaded_model, label_decoder)
    sum2_prediction = predict(sum2_tensor, loaded_model, label_decoder)
    alpha_prediction = predict(alpha_tensor, loaded_model, label_decoder)
    neq1_prediction = predict(neq1_tensor, loaded_model, label_decoder)
    neq2_prediction = predict(neq2_tensor, loaded_model, label_decoder)
    print(f"Predictions made in {time.time() - start:.2f} seconds.")

    print("Predictions:")
    for name, prediction in [
        ("Sum1", sum1_prediction),
        ("Sum2", sum2_prediction),
        ("Alpha", alpha_prediction),
        ("Neq1", neq1_prediction),
        ("Neq2", neq2_prediction),
    ]:
        print(f"{name}:")
        for label, confidence in prediction:
            print(f"  {label}: {confidence:.10f}")
        print("")


sum1 = [
    [
        [949, 72],
        [906, 59],
        [891, 59],
        [873, 55],
        [855, 54],
        [779, 41],
        [730, 41],
        [700, 37],
        [652, 36],
        [631, 29],
        [340, 29],
        [325, 32],
        [292, 32],
        [274, 37],
        [199, 37],
        [184, 41],
        [141, 41],
        [129, 44],
        [83, 44],
        [75, 47],
        [19, 47],
        [19, 50],
        [32, 72],
        [54, 101],
        [72, 116],
        [93, 141],
        [111, 159],
        [141, 180],
        [171, 207],
        [202, 231],
        [228, 259],
        [365, 350],
        [429, 398],
        [468, 419],
        [501, 444],
        [577, 495],
        [616, 516],
        [671, 552],
        [689, 570],
        [722, 592],
        [736, 606],
        [740, 606],
        [740, 610],
        [730, 610],
        [725, 613],
        [697, 618],
        [643, 639],
        [509, 685],
        [462, 707],
        [412, 725],
        [361, 740],
        [274, 782],
        [231, 800],
        [159, 837],
        [95, 876],
        [83, 881],
        [75, 894],
        [62, 899],
        [44, 914],
        [41, 917],
        [39, 924],
        [26, 932],
        [18, 945],
        [14, 949],
        [8, 950],
        [8, 953],
        [4, 953],
        [4, 957],
        [0, 960],
        [4, 963],
        [23, 967],
        [95, 967],
        [138, 970],
        [325, 970],
        [412, 963],
        [455, 963],
        [524, 953],
        [585, 953],
        [618, 949],
        [675, 949],
        [710, 945],
        [764, 945],
        [815, 939],
        [833, 939],
        [848, 932],
        [902, 930],
        [916, 927],
        [945, 927],
        [952, 924],
        [993, 924],
        [996, 921],
        [1000, 921],
    ]
]

sum2 = [
    [
        [885, 53],
        [823, 53],
        [803, 50],
        [146, 50],
        [133, 53],
        [87, 53],
        [82, 57],
        [16, 58],
        [12, 62],
        [0, 62],
        [8, 76],
        [16, 82],
        [25, 96],
        [39, 105],
        [55, 121],
        [75, 132],
        [87, 151],
        [180, 207],
        [212, 223],
        [242, 246],
        [282, 266],
        [321, 289],
        [403, 328],
        [446, 358],
        [489, 382],
        [533, 403],
        [571, 430],
        [653, 473],
        [775, 562],
        [780, 571],
        [794, 583],
        [794, 587],
        [751, 617],
        [721, 626],
        [548, 716],
        [494, 739],
        [439, 766],
        [391, 794],
        [289, 841],
        [242, 857],
        [203, 876],
        [169, 889],
        [146, 903],
        [126, 908],
        [107, 926],
        [98, 928],
        [85, 935],
        [82, 939],
        [78, 939],
        [157, 950],
        [633, 950],
        [666, 946],
        [760, 946],
        [780, 942],
        [876, 942],
        [885, 939],
        [1000, 939],
    ]
]

alpha = [
    [
        [817, 210],
        [817, 241],
        [813, 256],
        [813, 269],
        [781, 363],
        [772, 384],
        [758, 404],
        [747, 429],
        [726, 453],
        [717, 476],
        [698, 497],
        [681, 525],
        [630, 591],
        [606, 608],
        [591, 625],
        [566, 644],
        [474, 689],
        [450, 691],
        [418, 702],
        [387, 706],
        [280, 706],
        [214, 698],
        [182, 689],
        [107, 653],
        [58, 619],
        [48, 598],
        [28, 566],
        [16, 553],
        [0, 491],
        [0, 455],
        [3, 438],
        [11, 425],
        [24, 389],
        [37, 372],
        [62, 348],
        [79, 335],
        [161, 293],
        [177, 290],
        [235, 290],
        [263, 293],
        [297, 301],
        [331, 306],
        [374, 325],
        [412, 331],
        [446, 348],
        [546, 389],
        [598, 438],
        [619, 455],
        [647, 483],
        [694, 538],
        [719, 574],
        [768, 632],
        [809, 694],
        [851, 736],
        [868, 747],
        [892, 768],
        [913, 781],
        [924, 785],
        [930, 785],
        [937, 789],
        [949, 789],
        [952, 785],
        [952, 777],
        [958, 772],
        [962, 764],
        [975, 751],
        [979, 743],
        [1000, 719],
    ]
]


neq1 = [
    [
        [47, 348],
        [60, 348],
        [74, 351],
        [222, 351],
        [260, 354],
        [580, 354],
        [630, 360],
        [774, 360],
        [895, 364],
        [924, 370],
        [1000, 372],
    ],
    [
        [0, 627],
        [60, 632],
        [131, 632],
        [171, 638],
        [412, 638],
        [462, 632],
        [842, 632],
        [877, 630],
        [964, 630],
    ],
    [
        [257, 921],
        [257, 914],
        [270, 880],
        [300, 830],
        [347, 758],
        [367, 720],
        [400, 671],
        [420, 622],
        [447, 574],
        [467, 520],
        [538, 364],
        [575, 272],
        [641, 138],
        [664, 88],
        [664, 81],
        [667, 78],
    ],
]
neq2 = [
    [
        [0, 327],
        [578, 328],
        [608, 334],
        [641, 337],
        [667, 343],
        [719, 348],
        [744, 354],
        [788, 357],
        [847, 364],
        [877, 373],
        [889, 373],
        [922, 379],
        [972, 381],
        [977, 384],
        [1000, 384],
    ],
    [
        [67, 615],
        [555, 615],
        [616, 620],
        [648, 626],
        [747, 639],
        [771, 648],
        [794, 654],
        [814, 656],
        [830, 662],
        [873, 671],
        [889, 672],
        [894, 675],
        [900, 675],
        [903, 679],
        [907, 679],
    ],
    [
        [217, 154],
        [228, 162],
        [238, 174],
        [273, 207],
        [323, 248],
        [350, 277],
        [380, 298],
        [410, 327],
        [478, 384],
        [513, 420],
        [548, 450],
        [583, 484],
        [619, 520],
        [652, 559],
        [814, 725],
        [836, 757],
        [877, 801],
        [922, 845],
    ],
]

if __name__ == "__main__":
    main()
