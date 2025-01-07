# The basic idea is to read the symbol.yaml file, but not parse it.
# Instead, read it line by line and check for duplicate commands, which
# likely indicate redundant symbols.
# Some symbol redundancy is ok, if the symbol it draws is actually different,
# or if it's only slightly different, it can be noted as a similar symbol.
# Otherwise it's just a waste of space.

import re
import handtex.symbol_relations as sr


def main():

    symbol_list_path = "../database/symbols.yaml"
    with open(symbol_list_path, "r") as f:
        lines = f.readlines()

    commands = set()
    duplicate_count = 0

    # - \infty
    pattern = re.compile(r"^\s+-\s+\\(?P<command>\S+)")
    # - package
    package_pattern = re.compile(r"^-\s+package:.*")

    first = True
    for line_no, line in enumerate(lines, 1):
        match = pattern.match(line)
        if match:
            command = match.group("command")
            if command in commands:
                # > lineno command
                # | lineno command
                if first:
                    print(f"> {line_no} {command}")
                    first = False
                else:
                    print(f"| {line_no} {command}")
                duplicate_count += 1

            else:
                commands.add(command)
        package_pattern_match = package_pattern.match(line)
        if package_pattern_match:
            first = True

    print(f"Found {duplicate_count} duplicate commands.")


def main2():
    """
    These aren't the same, despite the name.

    mathabx-_leftsquigarrow txfonts-_leftsquigarrow
    mathabx-_leftrightsquigarrow amssymb-_leftrightsquigarrow
    amssymb-_rightsquigarrow mathabx-_rightsquigarrow
    txfonts-_boxright mathabx-_boxright
    mathabx-_boxleft txfonts-_boxleft

    nnearrow isn't the same as nnearrow.
    same for nnwarrow and nnwarrow.

    ~ Add to similarity group ('esint-_landupint', 'MnSymbol-_landupint'):
    fdsymbol-_landupint
    """

    # Use symbol relations data to find duplicate commands and auto-suggest similarity groupings.
    symbol_data = sr.SymbolData()

    keys = set(symbol_data.all_keys)

    out_new = ""
    out_add = ""

    while keys:
        key = keys.pop()
        # Find keys with the same command.
        same_command = {k for k in keys if symbol_data[k].command == symbol_data[key].command}
        if not same_command:
            continue

        same_command_with_self = same_command | {key}
        # Check if all of them are already in the same similarity group.
        similarity_group = symbol_data.get_similarity_group(key)
        if all(k in similarity_group for k in same_command_with_self):
            # Remove all of them from keys.
            keys -= same_command
            continue
        elif all(len(symbol_data.get_similarity_group(k)) == 1 for k in same_command_with_self):
            # Suggest making a new similarity group.
            out_new += f"{' '.join(same_command_with_self)}\n"
            keys -= same_command
        else:
            # Check which one of them has a similarity group already then.
            for k in same_command_with_self:
                if len(symbol_data.get_similarity_group(k)) > 1:
                    similarity_group = symbol_data.get_similarity_group(k)
                    # Remove those that are already in it.
                    same_command_with_self -= set(similarity_group)
                    if not same_command_with_self:
                        print(f"WTF {similarity_group}, {same_command_with_self}???")
                        break
                    out_add += f"~ Add to similarity group {similarity_group}:\n{' '.join(same_command_with_self)}\n"
                    keys -= same_command
                    break

    print(out_new)
    print(out_add)


if __name__ == "__main__":
    main2()
