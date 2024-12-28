input_file = "../../handtex/data/symbol_metadata/inside.txt"

# Split each line into a list of symbols.
# Sort this list of symbols.
# Then sort the entire file by rows.

with open(input_file, "r") as file:
    lines = file.readlines()
    lines = [line.strip() for line in lines]

# Sort the entire file by rows.
lines.sort(key=lambda x: x.lower())
lines.sort(key=lambda x: x.startswith("latex2e"), reverse=True)


# Write the sorted file back out.
output_file = "../../handtex/data/symbol_metadata/inside_sorted.txt"

with open(output_file, "w") as file:
    file.write("\n".join(lines))
