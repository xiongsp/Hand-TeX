import re


def extract_commands(latex_text):
    # Regex to find LaTeX commands starting with a backslash and consisting of letters only
    command_pattern = r"\\[a-zA-Z]+"

    # Finding all commands
    commands = re.findall(command_pattern, latex_text)
    return commands


# Multiline string with LaTeX commands copied from PDF
latex_text = r"""
⦟
 \angdnr
 ⦨
 \measanglerutone
 ⦝
 \rightanglemdot
∠
 \angle
 ⦭
 \measangleultonw
 ⦜
 \rightanglesqr
⦞
 \angles
 ⦬
 \measangleurtone
 ∢
 \sphericalangle
⦤
 \angleubar
 ∡
 \measuredangle
 ⦡
 \sphericalangleup
⦠
 \gtlpar
 ⦛
 \measuredangleleft
 ⟀
 \threedangle
⦯
 \measangledltosw
 ⊾
 \measuredrightangle
 ⦢
 \turnangle
⦮
 \measangledrtose
 ⍼
 \rangledownzigzagarrow
 ⦦
 \wideangledown
⦫
 \measangleldtosw
 ⦣
 \revangle
 ⦧
 \wideangleup
⦩
 \measanglelutonw
 ⦥
 \revangleubar
⦪
 \measanglerdtose
 ∟
 \rightangle
 
 """

# Extracting the commands
commands = extract_commands(latex_text)

# Converting the list of commands to YAML-ready format
yaml_output = "    - " + "\n    - ".join(commands)

print(yaml_output)
