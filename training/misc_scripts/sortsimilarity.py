# input_file = "../../handtex/data/symbol_metadata/similar_manual.txt"
input_file = "../../handtex/data/symbol_metadata/similar_greek.txt"

# Split each line into a list of symbols.
# Sort this list of symbols.
# Then sort the entire file by rows.

with open(input_file, "r") as file:
    lines = file.readlines()

# Split each line into a list of symbols.
lines = [line.strip().split() for line in lines]

for line in lines:
    # sort so that lines beginning with latex2e are first.
    line.sort()
    line.sort(key=lambda x: x.startswith("latex2e"), reverse=True)

# Merge back into strings per row.
lines = [" ".join(line) for line in lines]

# Sort the entire file by rows.
lines.sort()

# Write the sorted file back out.
output_file = "../../handtex/data/symbol_metadata/similar_manual_sorted.txt"

with open(output_file, "w") as file:
    file.write("\n".join(lines))
