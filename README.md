# Hand TeX

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/voxelcubes/Hand-TeX?logo=GitHub)](https://github.com/voxelcubes/Hand-TeX/releases)
[![PyPI version](https://img.shields.io/pypi/v/handtex)](https://pypi.org/project/handtex/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Draw the symbol to find out what LaTeX command to use.

![Example](https://raw.githubusercontent.com/VoxelCubes/Hand-TeX/master/media/demo.gif)


### Features
- Over 3500 symbols and their variations, of which over 1700 are unique
- Browse a list of all symbols
- Fully offline
- Supported packages: amsmath, amssymb, bbold, cmll, dsfont, esint, fdsymbol, gensymb, halloweenmath, latex2e, latexsym, logix, marvosym, mathabx, mathdots, mathrsfs, MnSymbol, stix, stix2, stmaryrd, textcomp, tipa, txfonts, upgreek, wasysym

This was inspired by the [Detexify](http://detexify.kirelabs.org/) project and uses
some of the same training data, now with new symbols and additional training data.

<div align="center">
    <img src="https://raw.githubusercontent.com/VoxelCubes/Hand-TeX/master/media/classification.png" width="45%">
    <img src="https://raw.githubusercontent.com/VoxelCubes/Hand-TeX/master/media/classification_light.png" width="45%">
    <br>
    <img src="https://raw.githubusercontent.com/VoxelCubes/Hand-TeX/master/media/symbol_list.png" width="60%">
</div>

### Installation

| Platform | Format     | Link                                                                  |
| --- |------------|-----------------------------------------------------------------------|
| Linux | Flatpak | Coming soon                                                           |
| | AUR | Coming soon                                                           |
| Windows | Executable | [HandTeX.exe](https://github.com/VoxelCubes/Hand-TeX/releases/latest) |
| All Platforms | Python package | `pip install handtex`                                                 |

The Python package requires Python 3.10 or later. Ensure Python is in your PATH.

If installed with Python, run `handtex` to start the program.
If that doesn't work, try `python -m handtex.main`.


### Does this work with CUDA?

Yes, but for using Hand TeX, this is not necessary. The model is small enough to
run very quickly on a CPU, there is no noticeable difference.


### Can I help make more training data?

Yes! In Hand TeX, go to the hamburger menu and select "Help symbol training".
This will present you with a symbol to draw. Then just submit your drawing
and it will be saved to a .json file. The location for this can be configured.
Once you are satisfied with the drawings you have collected, send all of the 
.json files in one .zip archive to [voxel.aur@gmail.com](mailto:voxel.aur@gmail.com?subject=Hand%20TeX%20training%20data).
Thanks!

![Training example](https://raw.githubusercontent.com/VoxelCubes/Hand-TeX/master/media/training.png)

### Can I help in other ways?

Yes, help is always appreciated. If you know a thing or two about training
models, I would like to know how to improve it further.

If you'd like to suggest new symbols, please open an issue and include drawings
you made with the "Help symbol training" feature. To do this for new symbols, 
just manually enter the symbol name you want to suggest and press skip in the 
interface. Then draw the symbol and submit it. It would also be nice to
include a .tex file that shows the symbol in use, so that it compiles with 
pdflatex or xelatex.


### What is the difference between Hand TeX and Detexify?

Hand TeX supports the same symbols and many more, as it uses
a larger dataset. The Detexify model uses time information to know
what order you drew the strokes in, and what direction too.
This can be useful for common symbols that are drawn in a specific way,
but this approach struggles on more complex symbols that don't have 
a common way to draw them.

Hand TeX renders your strokes to a flat image and uses a convolutional
neural network to classify the symbol. This approach ignores
the order you draw the strokes in, focusing solely on the shape of the
symbol.

The expanded, modified dataset is available under the same license
as the original Detexify dataset [here](https://github.com/VoxelCubes/Hand-TeX/releases/tag/database).


### Running from source

These instructions assume you have Python 3.10 or later installed, as well as a collection of terminal utilities.
This will work on any Linux system, or other OS with the appropriate tools installed.

To run from source, clone the repository
```bash
git clone https://github.com/VoxelCubes/Hand-TeX.git
cd Hand-TeX
```

Optionally, create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

Install the dependencies
```bash
pip install -r requirements.txt
```
Optionally, if you wish to train and develop, also install the dev dependencies
```bash
pip install -r requirements_training.txt
```

You need a model to run the program. You have two options:

1. Download the model from the releases page
```bash
mkdir -p handtex/data/model
curl -o handtex/data/model/handtex.safetensors https://github.com/VoxelCubes/Hand-TeX/releases/download/model/handtex.safetensors
curl -o handtex/data/model/encodings.txt https://github.com/VoxelCubes/Hand-TeX/releases/download/model/encoding.txt
```
2. Train the model yourself. This requires the dev dependencies and a lot of time.
```bash
mkdir -p training/database
curl -o training/database/handtex.db.tar.xz https://github.com/VoxelCubes/Hand-TeX/releases/download/database/handtex.db.tar.xz
tar -xf training/database/handtex.db.tar.xz -C training/database
PYTHONPATH=. python training/train.py
```

Finally, run the program
```bash
PYTHONPATH=. python handtex/main.py
```
or
```bash
make run
```

If you have changed anything with the symbols or Qt ui files, you will need to regenerate the resources
```bash
make refresh-assets
```
