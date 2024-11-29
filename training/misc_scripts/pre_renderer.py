import os
import json
import subprocess
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# This process was inspired by the way TeX Match solved the same problem.
# Running latex at runtime for each symbol wouldn't work,
# especially considering how much shit it took to render all of them.
# So we're pre-rendering all symbols as svg files, that way nothing can go wrong.
# The 1.3MB of space this takes up is waaay less than including 3 GB of latex
# compiler and packages for this to all render, not to mention the time
# wasted if re-rendering the latex upon each startup (especially for the symbol
# list, which would need all symbols).

# Running this properly requires a shit-ton of texlive packages.
# You have been warned.

import handtex.utils as ut
import handtex.data


# Might need to re-run the script a few times in case one of the inkscapes gets killed.
# Only an issue if multi-threading. But you really should multithread with 1000 symbols...
# CPUS = os.cpu_count()
CPUS = 1
with ut.resource_path(handtex.data) as path:
    SYMBOLS_FILE = path / "symbol_metadata" / "symbols.json"
    OUTPUT_DIR = path / "symbols"


# 1. Load symbols from symbols.json
def load_symbols(file_path):
    with open(file_path, "r") as file:
        symbols = json.load(file)
    return symbols


# 2. Generate LaTeX files
def generate_latex_file(symbol):
    # Parse package, encoding, and command from symbol id
    package, encoding, command = symbol["key"].split("-", maxsplit=2)
    package = package.strip()
    encoding = encoding.strip()
    command = command.strip()

    # Construct LaTeX document
    tex_content = f"""
    \\documentclass[10pt]{{article}}
    \\usepackage[utf8]{{inputenc}}
    \\usepackage[{encoding}]{{fontenc}}
    {'\\usepackage{{' + package + '}}' if package != 'latex2e' else ''}
    \\pagestyle{{empty}}
    \\begin{{document}}
    {'$' + symbol['command'] + '$' if symbol['mathmode'] else symbol['command']}
    \\end{{document}}
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tex")
    with open(temp_file.name, "w") as file:
        file.write(tex_content)
    return temp_file.name


# 3. Render LaTeX to PDF using Tectonic
def render_latex_to_pdf(tex_path):
    pdf_path = tex_path.replace(".tex", ".pdf")
    subprocess.run(
        ["pdflatex", "-output-directory", os.path.dirname(tex_path), tex_path], check=True
    )
    return pdf_path


# 4. Convert PDF to SVG
def convert_pdf_to_svg(pdf_path):
    svg_path = pdf_path.replace(".pdf", ".svg")
    subprocess.run(["pdf2svg", pdf_path, svg_path], check=True)
    return svg_path


# 5. Crop SVG using Inkscape
def crop_svg(svg_path):
    cropped_svg_path = svg_path.replace(".svg", ".cropped.svg")
    subprocess.run(["inkscape", "-D", "-o", cropped_svg_path, svg_path], check=True)
    return cropped_svg_path


# 6. Resize SVG and add width/height attributes
def resize_svg(svg_path):
    with open(svg_path, "r") as file:
        svg_content = file.read()

    # Update the SVG with desired attributes and style
    svg_content = svg_content.replace("<svg ", '<svg width="64" height="64" ')

    with open(svg_path, "w") as file:
        file.write(svg_content)


# 7. Minimize SVG using Scour
def minimize_svg(svg_path):
    minimized_svg_path = svg_path.replace(".svg", ".min.svg")
    subprocess.run(
        [
            "scour",
            "--enable-viewboxing",
            "--enable-id-stripping",
            "--enable-comment-stripping",
            "--shorten-ids",
            "--set-precision=5",
            "--indent=none",
            "-i",
            svg_path,
            "-o",
            minimized_svg_path,
        ],
        check=True,
    )
    shutil.move(minimized_svg_path, svg_path)


# 8. Patch in a fill color
def patch_fill_color(svg_path):
    with open(svg_path, "r") as file:
        svg_content = file.read()

    # Some files have no fill but instead a stroke.
    # We need to check for the presence of fill="none" and stroke
    # For those we don't want to add a fill color.

    # if 'fill="none"' in svg_content or 'stroke=' in svg_content:
    #     return

    # Update the SVG with desired attributes and style
    svg_content = svg_content.replace("<svg ", '<svg fill="#000000" ')

    with open(svg_path, "w") as file:
        file.write(svg_content)


# 9. Rename SVGs
def rename_svg(symbol, svg_path, output_dir):
    output_path = os.path.join(output_dir, f"{symbol['filename']}.svg")
    shutil.move(svg_path, output_path)


# 10. Process a single symbol
def process_symbol(symbol):
    # Check if it already exists.
    if os.path.exists(os.path.join(OUTPUT_DIR, f"{symbol['filename']}.svg")):
        print(f"Symbol {symbol['command']} already exists. Skipping.")
        return
    try:
        tex_path = generate_latex_file(symbol)
        pdf_path = render_latex_to_pdf(tex_path)
        svg_path = convert_pdf_to_svg(pdf_path)
        cropped_svg_path = crop_svg(svg_path)
        resize_svg(cropped_svg_path)
        minimize_svg(cropped_svg_path)
        patch_fill_color(cropped_svg_path)
        rename_svg(symbol, cropped_svg_path, OUTPUT_DIR)
    except subprocess.CalledProcessError as e:
        print(f"Error processing symbol {symbol['command']}: {e}")
    finally:
        # Clean up temporary files
        for ext in [".tex", ".aux", ".log", ".pdf", ".svg"]:
            tmp_file = tex_path.replace(".tex", ext)
            if os.path.exists(tmp_file):
                os.remove(tmp_file)


# Main workflow
def main():
    symbols = load_symbols(SYMBOLS_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_symbol, symbol) for symbol in symbols]
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions caught during processing

    print(f"Expecting to find {len(symbols)} SVG files in the {OUTPUT_DIR} directory.")

    # Find out which symbols failed to render.
    missing_symbols = []
    for symbol in symbols:
        if not os.path.exists(os.path.join(OUTPUT_DIR, f"{symbol['filename']}.svg")):
            missing_symbols.append(symbol["command"])
    if missing_symbols:
        print(f"Failed to render the following symbols: {', '.join(missing_symbols)}")


if __name__ == "__main__":
    main()
