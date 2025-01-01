import numpy as np
import matplotlib.pyplot as plt

from noise import pnoise2
from training.image_gen import augment_strokes_with_perlin


def visualize_strokes_with_augmentation(
    strokes: list[list[list[int]]],
    scale: float = 1000.0,
    amplitude: float = 150.0,
    octaves: int = 3,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 42,
):
    """
    Visualizes original and augmented strokes side by side, with Perlin noise overlay.

    Parameters:
    - strokes: List of strokes, where each stroke is a list of 2D coordinates.
    - scale, amplitude, octaves, persistence, lacunarity, seed: Parameters for Perlin noise.
    """
    augmented_strokes = augment_strokes_with_perlin(
        strokes, scale, amplitude, octaves, persistence, lacunarity, seed
    )

    # Generate Perlin noise for overlay
    noise_grid = np.zeros((1000, 1000))
    for y in range(1000):
        for x in range(1000):
            noise_grid[y, x] = pnoise2(
                x / scale,
                y / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1000,
                repeaty=1000,
                base=seed,
            )

    # Normalize noise for visualization
    noise_grid = (noise_grid - np.min(noise_grid)) / (np.max(noise_grid) - np.min(noise_grid))

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Stroke Augmentation Visualization")

    # Plot original strokes
    axes[0, 0].set_title("Original Strokes (With Points)")
    for stroke in strokes:
        stroke_x = [point[0] for point in stroke]
        stroke_y = [point[1] for point in stroke]
        axes[0, 0].plot(stroke_x, stroke_y, marker="o")

    axes[1, 0].set_title("Original Strokes (Without Points)")
    for stroke in strokes:
        stroke_x = [point[0] for point in stroke]
        stroke_y = [point[1] for point in stroke]
        axes[1, 0].plot(stroke_x, stroke_y)

    # Plot augmented strokes
    axes[0, 1].set_title("Augmented Strokes (With Points)")
    for stroke in augmented_strokes:
        stroke_x = [point[0] for point in stroke]
        stroke_y = [point[1] for point in stroke]
        axes[0, 1].plot(stroke_x, stroke_y, marker="o")

    axes[1, 1].set_title("Augmented Strokes (Without Points)")
    for stroke in augmented_strokes:
        stroke_x = [point[0] for point in stroke]
        stroke_y = [point[1] for point in stroke]
        axes[1, 1].plot(stroke_x, stroke_y)

    # Overlay Perlin noise
    for i in range(2):
        for j in range(2):
            axes[i, j].imshow(noise_grid, cmap="gray", extent=(0, 1000, 1000, 0), alpha=0.3)
            axes[i, j].axis("equal")
            axes[i, j].set_xlabel("X Coordinate")
            axes[i, j].set_ylabel("Y Coordinate")
            axes[i, j].set_xlim(0, 1000)
            axes[i, j].set_ylim(1000, 0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Example usage:
# strokes = [[[100, 200], [150, 250], [200, 300]], [[400, 500], [450, 550]]]
strokes = [
    [[282, 583], [389, 568], [580, 568], [590, 558], [634, 558], [634, 548]],
    [[380, 710], [463, 710], [541, 690], [732, 690]],
    [
        [267, 431],
        [267, 387],
        [282, 377],
        [389, 377],
        [389, 387],
        [409, 397],
        [531, 397],
        [580, 358],
        [615, 289],
    ],
]
seed = 0
while True:
    visualize_strokes_with_augmentation(strokes, seed=seed)
    seed += 1
