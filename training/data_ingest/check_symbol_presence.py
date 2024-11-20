import re

import handtex.utils as ut


def main():
    # Test data loading.
    symbols = ut.load_symbols()
    symbol_commands = set(s.command.replace(r"\\", "\\") for s in symbols.values())

    text_input = r"""
 | { \{ b \lfloor / / ⇑ \Uparrow x \llcorner
| \vert } \} c \rfloor \ \backslash ↑ \uparrow y \lrcorner
‖ \| 〈 \langle d \lceil [ [ ⇓ \Downarrow p \ulcorner
‖ \Vert 〉 \rangle e \rceil ] ] ↓ \downarrow q \urcorner
   \aleph
η \eta ω \omega υ \upsilon % \varrho Π \Pi i \beth
γ \gamma φ \phi ξ \xi ς \varsigma Ψ \Psi k \daleth
ι \iota π \pi ζ \zeta ϑ \vartheta Σ \Sigma ג \gimel 
    """

    # Parse out any commands.
    commands = re.findall(r"\\[a-zA-Z]+", text_input)
    print(f"Found {len(commands)} commands.")
    for command in commands:
        if command not in symbol_commands:
            print(f"Command {command} not found in the symbols database.")


if __name__ == "__main__":
    main()
