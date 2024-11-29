import handtex.data.symbol_metadata
from importlib import resources
from matplotlib import pyplot as plt


def main():

    with resources.path(handtex.data.symbol_metadata, "symbol_frequency.csv") as path:
        frequencies_path = path

    with resources.path(handtex.data.symbol_metadata, "augmented_symbol_frequency.csv") as path:
        augmented_frequencies_path = path

    # Load the symbol frequencies.
    frequencies = {}
    with open(frequencies_path, "r") as f:
        for line in f:
            symbol, frequency = line.strip().split(",")
            frequencies[symbol] = int(frequency)

    augmented_frequencies = {}
    with open(augmented_frequencies_path, "r") as f:
        for line in f:
            symbol, frequency = line.strip().split(",")
            augmented_frequencies[symbol] = int(frequency)

    # Draw the boxplot.
    fig, ax = plt.subplots()
    ax.boxplot([list(frequencies.values()), list(augmented_frequencies.values())])
    ax.set_xticklabels(["Original", "Augmented"])
    ax.set_ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    main()
