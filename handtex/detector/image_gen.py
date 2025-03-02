import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

IMAGE_SIZE = 64


def strokes_to_grayscale_image(stroke_data: list[list[tuple[int, int]]], image_size: int):
    # Create the Pillow version
    img_pil = Image.new("L", (image_size, image_size), 255)  # 'L' mode for grayscale
    draw = ImageDraw.Draw(img_pil)

    # Scale down the strokes to fit the image size, adding an extra 5% padding.
    padding = 0.10
    scale = (image_size * (1 - padding)) / 1000
    offset = (image_size * padding) / 2
    stroke_data = [
        [(round(point[0] * scale + offset), round(point[1] * scale + offset)) for point in stroke]
        for stroke in stroke_data
    ]

    # Clear out duplicate consecutive points, so that they can be detected as single points.
    # Otherwise, a short segment could be scaled down to a single point, and then
    # not get the point handling treatment, resulting in nothing being drawn.
    for stroke in stroke_data:
        for i in range(len(stroke) - 1, 0, -1):
            if stroke[i] == stroke[i - 1]:
                stroke.pop(i)

    # Iterate through each stroke and draw it
    for stroke in stroke_data:
        if len(stroke) == 1:
            # Handle single point by adding a second point offset by 1
            point = stroke[0]
            stroke.append((point[0] + 1, point[1] + 1))

        draw.line(stroke, fill=0, width=1)

    img_pil_np = np.array(img_pil)

    return img_pil_np


def rotation_matrix(angle: float, image_size: int = 1000) -> np.ndarray:
    """
    Rotate the stroke data by the given angle around the center of the image.
    Positive angles rotate counter-clockwise, negative angles rotate clockwise.
    The center is at coordinates (image_size / 2, image_size / 2), not specific to
    the image, since the image is expected to be centered in the frame.
    After rotation, the image may need to be scaled down to fit the original image size.

    :param angle: Angle in degrees to rotate the strokes.
    :param image_size: Size of the image the strokes are drawn on.
    :return: Rotation matrix.
    """
    angle_rad = np.radians(-angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    x_offset = image_size / 2
    y_offset = image_size / 2

    transformation = np.array(
        [
            [cos_theta, -sin_theta, -x_offset * cos_theta + y_offset * sin_theta + x_offset],
            [sin_theta, cos_theta, -x_offset * sin_theta - y_offset * cos_theta + y_offset],
            [0, 0, 1],
        ]
    )
    return transformation


def reflection_matrix(angle: float, image_size: int = 1000) -> np.ndarray:
    """
    Reflect the stroke data along an axis that passes through the center of the image
    at a specified angle. The center is at coordinates (image_size / 2, image_size / 2).
    Positive angles rotate the axis counter-clockwise, negative angles rotate clockwise.

    :param angle: Angle in degrees defining the axis to reflect across.
    :param image_size: Size of the image the strokes are drawn on.
    :return: Reflection matrix.
    """
    angle_rad = np.radians(-angle)
    two_theta = 2 * angle_rad
    cos_2theta = np.cos(two_theta)
    sin_2theta = np.sin(two_theta)
    x_offset = image_size / 2
    y_offset = image_size / 2

    transformation = np.array(
        [
            [cos_2theta, sin_2theta, x_offset - cos_2theta * x_offset - sin_2theta * y_offset],
            [sin_2theta, -cos_2theta, y_offset - sin_2theta * x_offset + cos_2theta * y_offset],
            [0, 0, 1],
        ]
    )
    return transformation


def scale_matrix(scale_x: float, scale_y: float, image_size: int = 1000) -> np.ndarray:
    """
    Scale the stroke data along the x and y axes, relative to the center of the image.
    The center is at coordinates (image_size / 2, image_size / 2).

    :param scale_x: Scaling factor for the x-axis. Values less than 1 will shrink the x-axis.
    :param scale_y: Scaling factor for the y-axis. Values less than 1 will shrink the y-axis.
    :param image_size: Size of the image the strokes are drawn on.
    :return: Scaling matrix.
    """
    x_offset = image_size / 2
    y_offset = image_size / 2

    transformation = np.array(
        [
            [scale_x, 0, x_offset - scale_x * x_offset],
            [0, scale_y, y_offset - scale_y * y_offset],
            [0, 0, 1],
        ]
    )
    return transformation


def translation_matrix(
    translate_x: float, translate_y: float, image_size: int = 1000
) -> np.ndarray:
    """
    Translate the stroke data by the given x and y offsets, in proportion to the image size.

    :param translate_x: Offset to translate the strokes along the x-axis.
    :param translate_y: Offset to translate the strokes along the y-axis.
    :param image_size: Size of the image the strokes are drawn on.
    :return: Translation matrix.
    """
    x_offset = image_size * translate_x
    y_offset = image_size * translate_y
    transformation = np.array(
        [
            [1, 0, x_offset],
            [0, 1, y_offset],
            [0, 0, 1],
        ]
    )
    return transformation


def skew_matrix(skew_x: float, skew_y: float, image_size: int = 1000) -> np.ndarray:
    """
    Skew the stroke data along the x and y axes, relative to the center of the image.
    The center is at coordinates (image_size / 2, image_size / 2).
    If the skewed strokes exceed the image boundaries, they are scaled down to fit.

    :param skew_x: Skew factor along the x-axis. Positive values shear to the right.
    :param skew_y: Skew factor along the y-axis. Positive values shear upwards.
    :param image_size: Size of the image the strokes are drawn on.
    :return: Skew matrix.
    """
    x_offset = image_size / 2
    y_offset = image_size / 2

    transformation = np.array(
        [
            [1, skew_x, -skew_x * y_offset],
            [skew_y, 1, -skew_y * x_offset],
            [0, 0, 1],
        ]
    )
    return transformation


def apply_transformations(
    stroke_data: list[list[tuple[int, int]]],
    transformations: list[np.ndarray] | np.ndarray,
    shuffle_transformations: bool = False,
    image_size: int = 1000,
) -> list[list[tuple[int, int]]]:
    """
    Apply a sequence of transformation matrices to the stroke data.
    If the transformed strokes exceed the image boundaries, they are scaled down to fit.
    Transformations are applied in the ascending order they are provided, unless shuffled.

    :param stroke_data: List of strokes, each stroke being a list of points.
    :param transformations: List of 3x3 transformation matrices to apply.
    :param shuffle_transformations: [Optional] If True, shuffle the order of transformations.
    :param image_size: [Optional] Size of the image the strokes are drawn on.
    :return: Transformed stroke data.
    """
    if not isinstance(transformations, list):
        transformations = [transformations]

    # Shuffle transformations if required
    if shuffle_transformations:
        generator = np.random.default_rng()
        generator.shuffle(transformations)

    # Combine all transformation matrices into a single matrix
    total_transformation = np.eye(3)
    for transformation in transformations:
        total_transformation = transformation @ total_transformation

    # Flatten all strokes into a single array and keep track of stroke lengths.
    all_points = []
    stroke_lengths = []
    for stroke in stroke_data:
        all_points.extend(stroke)
        stroke_lengths.append(len(stroke))

    points_array = np.array(all_points)
    ones = np.ones((points_array.shape[0], 1))
    homogeneous_points = np.hstack([points_array, ones])

    transformed_points = homogeneous_points @ total_transformation.T
    x_new, y_new = transformed_points[:, 0], transformed_points[:, 1]

    # Calculate the bounding box of the transformed points.
    # We may need to scale the image down to fit again.
    min_x, min_y = x_new.min(), y_new.min()
    max_x, max_y = x_new.max(), y_new.max()

    width, height = max_x - min_x, max_y - min_y
    if width == 0 or height == 0:
        scale = 1
    else:
        scale = min(image_size / width, image_size / height)

    if scale < 1:
        tx = (1 - scale) * image_size / 2
        ty = (1 - scale) * image_size / 2
        scaling_matrix = np.array(
            [
                [scale, 0, tx],
                [0, scale, ty],
                [0, 0, 1],
            ]
        )

        ones = np.ones((transformed_points.shape[0], 1))
        transformed_points = np.hstack([transformed_points[:, :2], ones])
        scaled_points = transformed_points @ scaling_matrix.T
        x_scaled = np.round(scaled_points[:, 0]).astype(int)  # Replace with x_new??
        y_scaled = np.round(scaled_points[:, 1]).astype(int)
    else:
        x_scaled = np.round(x_new).astype(int)
        y_scaled = np.round(y_new).astype(int)

    # Split the transformed points back into strokes.
    transformed_strokes = []
    idx = 0
    for length in stroke_lengths:
        stroke_points = list(zip(x_scaled[idx : idx + length], y_scaled[idx : idx + length]))
        transformed_strokes.append(stroke_points)
        idx += length

    return transformed_strokes


def tensorize_strokes(stroke_data: list[list[tuple[int, int]]], image_size: int):
    img = strokes_to_grayscale_image(stroke_data, image_size)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts the image to a PyTorch tensor (C, H, W) with values in [0, 1]
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to range [-1, 1]
        ]
    )
    img_tensor = transform(img)  # Output shape will be [1, H, W]

    # Expand the tensor to add a batch dimension.
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor


def augment_strokes_with_perlin(
    strokes: list[list[tuple[int, int]]],
    scale: float = 1000.0,  # Controls the "zoom" of the noise
    amplitude: float = 150.0,  # Controls the intensity of the distortion
    octaves: int = 3,  # Number of detail levels in the noise
    persistence: float = 0.5,  # How each octave contributes to the overall noise
    lacunarity: float = 2.0,  # Frequency multiplier for each octave
    seed: int = 42,  # Random seed for reproducibility
) -> list[list[tuple[int, int]]]:
    """
    Augments stroke data with Perlin noise distortions.

    Parameters:
    - strokes: List of strokes, where each stroke is a list of 2D coordinates.
    - scale: Scale factor for the Perlin noise.
    - amplitude: Amplitude of the noise distortion.
    - octaves: Number of noise octaves for detail.
    - persistence: Controls the contribution of each octave.
    - lacunarity: Frequency multiplier between octaves.
    - seed: Seed for the Perlin noise generator.

    Returns:
    - List of augmented strokes with distorted coordinates.
    """
    # This dependency is only needed for training.
    # This way the file can still be used for inference without the dependency.
    from noise import pnoise2

    distorted_strokes = []

    for stroke in strokes:
        distorted_stroke = []
        for x, y in stroke:
            # Normalize the coordinates to the scale of the noise
            noise_x = pnoise2(
                x / scale,
                y / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1000,  # Match canvas size for seamless noise
                repeaty=1000,
                base=seed,
            )

            noise_y = pnoise2(
                y / scale,
                x / scale,  # Swap x and y for variety in distortion
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1000,
                repeaty=1000,
                base=seed + 1,  # Slightly different seed for y-axis noise
            )

            # Apply the noise as a distortion
            new_x = x + amplitude * noise_x
            new_y = y + amplitude * noise_y

            distorted_stroke.append([int(new_x), int(new_y)])

        distorted_strokes.append(distorted_stroke)

    return distorted_strokes
