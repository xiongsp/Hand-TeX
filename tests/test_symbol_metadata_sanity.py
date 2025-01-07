import re
from time import time
from importlib import resources
from pathlib import Path

from loguru import logger
import networkx as nx
from matplotlib import pyplot as plt

import handtex.utils as ut
import handtex.symbol_relations as sr
import structures
from handtex.data import symbol_metadata
import handtex.structures as st


def test_similar_lists_disjunct() -> None:
    """
    Test that the symbol lists are disjunct.
    """
    with resources.path(symbol_metadata, "") as metadata_dir:
        metadata_dir = Path(metadata_dir)
    files = list(metadata_dir.glob("similar*"))

    # Check that each file is consistent on it's own.
    for file in files:
        line_symbols: list[set[str]]
        line_symbols = []

        with file.open("r") as f:
            lines = f.readlines()
        for line in lines:
            symbols = line.strip().split()
            line_symbol_set = set()
            for s in symbols:
                assert (
                    s not in line_symbol_set
                ), f"Symbol {s} is listed twice in {file}, line {line}"
                line_symbol_set.add(s)
            assert line_symbol_set not in line_symbols, f"Line {line} is listed twice in {file}"
            line_symbols.append(line_symbol_set)

        # Check that no two lines are a subset of each other.
        for i, line1 in enumerate(line_symbols, 1):
            for j, line2 in enumerate(line_symbols, 1):
                if i == j:
                    continue
                assert not line1.issubset(line2), f"Line {i} is a subset of line {j} in {file}"

        # Check that all lines are disjunct.
        for i, line1 in enumerate(line_symbols, 1):
            for j, line2 in enumerate(line_symbols, 1):
                if i == j:
                    continue
                assert not line1 & line2, f"Line {i} and line {j} share symbols in {file}"

    # There cannot be the same key listed twice in the file.
    global_keys = set()
    for file in files:
        # Just parse out all the keys with a regex.
        # A key is a string of 1 or more characters, no whitespace.
        keys = list(re.findall(r"\S+", file.read_text()))
        key_set = set(keys)
        for key in key_set:
            keys.remove(key)
        assert not keys, f"File {file} contains duplicate keys: {keys}"

        # The keys in this file must not be in the global set.
        intersection = global_keys & key_set
        assert (
            not intersection
        ), f"File {file} contains keys already in another file: {intersection}"
        global_keys |= key_set


def test_symbol_name_collisions() -> None:
    """
    Look for symbols that have the same command, but aren't
    in a similarity relation.
    """
    # Load symbols.
    symbol_data = sr.SymbolData()
    symbols = symbol_data.all_keys

    whitelist = [
        "fdsymbol-_landupint",
        "amssymb-_leftrightsquigarrow",
        "amssymb-_rightsquigarrow",
        "txfonts-_leftsquigarrow",
        "stmaryrd-_nnearrow",
        "stmaryrd-_nnwarrow",
        "mathabx-_boxright",
        "mathabx-_boxleft",
        "fdsymbol-_landdownint",
        "fdsymbol-_landupint",
        "mathabx-_nsucc",
        "mathabx-_nprec",
        "mathabx-_nsucceq",
        "mathabx-_npreceq",
        "mathabx-_nsubseteq",
        "mathabx-_nsupseteq",
        "mathabx-_nsubseteqq",
        "mathabx-_nsupseteqq",
        "MnSymbol-_gnapprox",
        "MnSymbol-_lnapprox",
        "MnSymbol-_gnsim",
        "MnSymbol-_lnsim",
        "txfonts-_varprod",
        "txfonts-_coloneq",
        "txfonts-_eqcolon",
        "mathabx-_nsuccapprox",
        "mathabx-_nprecapprox",
        "mathabx-_nsqsubset",
        "mathabx-_nsqsupset",
        "mathabx-_nsupset",
        "mathabx-_nsubset",
        "mathabx-_nsqsubseteq",
        "mathabx-_nsqsupseteq",
    ]

    # Check for collisions.
    for symbol1 in symbols:
        for symbol2 in symbols:
            if symbol1 == symbol2:
                continue
            if symbol_data[symbol1].command == symbol_data[symbol2].command:
                # Skip the whitelist.
                if symbol1 in whitelist or symbol2 in whitelist:
                    continue
                # Check if they are in a similarity relation.
                if symbol1 in symbol_data.get_similarity_group(symbol2):
                    continue
                if symbol2 in symbol_data.get_similarity_group(symbol1):
                    continue
                raise AssertionError(f"Symbols {symbol1} and {symbol2} have the same command.")


def test_self_symmetry_similarity_conflict() -> None:
    """
    Test that only the leader of a similarity group has self-symmetry.
    """

    similarity: dict[str, tuple[str, ...]] = sr.load_symbol_metadata_similarity()
    symbols: dict[str, st.Symbol] = sr.load_symbols()
    leaders: list[str] = sr.select_leader_symbols(list(symbols.keys()), similarity)

    symmetries: dict[str, list[st.Transformation]] = sr.load_symbol_metadata_self_symmetry()

    for leader in leaders:
        # Self symmetries must match the leader's.
        for similar in similarity.get(leader, []):
            if similar not in symmetries:
                continue
            # If the leader has no self-symmetry, whatever.
            if not symmetries.get(leader, []):
                continue
            assert set(symmetries.get(leader, [])) == set(
                symmetries.get(similar, [])
            ), f"Similar {similar} has different self-symmetry from leader {leader}."


def test_leader_mapping() -> None:
    """
    Test that the leader mapping is correct.
    """
    similarity_groups = sr.load_symbol_metadata_similarity_groups()
    to_leader = sr.construct_to_leader_mapping(similarity_groups)
    # We do not want any leaders in this mapping. Meaning, a key may not map to itself.
    for key, leader in to_leader.items():
        assert key != leader, f"Key {key} is it's own leader."


def test_self_symmetry_normalization() -> None:
    """
    Test that the self-symmetry transformations are normalized.
    """
    self_symmetries = sr.load_symbol_metadata_self_symmetry()
    similarity_groups = sr.load_symbol_metadata_similarity_groups()
    to_leader = sr.construct_to_leader_mapping(similarity_groups)
    normalized = sr.normalize_self_symmetry_to_leaders(self_symmetries, to_leader)
    # Print out differences.
    # for key, value in symmetries.items():
    #     if key not in normalized:
    #         print(f"Key {key} is not in normalized.")
    #         continue
    #     if value != normalized[key]:
    #         print(f"Key {key} has different value: {value} vs {normalized[key]}")

    # We only want leaders in this normalized dictionary.
    for key in normalized:
        assert key not in to_leader.keys(), f"Key {key} is not a leader."


def test_other_symmetry_normalization() -> None:
    """
    Test that the other-symmetry transformations are normalized.
    """
    self_symmetries = sr.load_symbol_metadata_self_symmetry()
    other_symmetries = sr.load_symbol_metadata_other_symmetry()
    similarity_groups = sr.load_symbol_metadata_similarity_groups()
    to_leader = sr.construct_to_leader_mapping(similarity_groups)
    normalized = sr.normalize_other_symmetry_to_leaders(
        other_symmetries, to_leader, self_symmetries
    )
    # Print out differences.
    # for key, value in symmetries.items():
    #     if key not in normalized:
    #         print(f"Key {key} is not in normalized.")
    #         continue
    #     if value != normalized[key]:
    #         print(f"Key {key} has different value: {value} vs {normalized[key]}")

    # We only want leaders in this normalized dictionary.
    for key in normalized:
        assert key not in to_leader.keys(), f"Key {key} is not a leader."

        for target, _ in normalized[key]:
            assert target not in to_leader.keys(), f"Target {target} is not a leader."


def test_graph_transitivity() -> None:
    # Create a small test graph.
    symbol_keys = ["leftarrow", "rightarrow", "uparrow", "downarrow", "box", "circle"]
    similarity_groups = []
    self_symmetries = {
        "leftarrow": [st.Transformation.mir(0)],
        "rightarrow": [st.Transformation.mir(0)],
        "uparrow": [st.Transformation.mir(90)],
        "downarrow": [st.Transformation.mir(90)],
        "box": [
            st.Transformation.rot(90),
            st.Transformation.rot(180),
            st.Transformation.rot(270),
            st.Transformation.mir(0),
        ],
        "circle": [st.Transformation.mir(0), st.Transformation.mir(90)],
    }
    other_symmetries = {
        "leftarrow": [
            ("rightarrow", [st.Transformation.mir(0), st.Transformation.rot(180)]),
            ("uparrow", [st.Transformation.mir(135), st.Transformation.rot(270)]),
            ("downarrow", [st.Transformation.mir(45), st.Transformation.rot(90)]),
        ],
        "rightarrow": [("leftarrow", [st.Transformation.mir(0), st.Transformation.rot(180)])],
        "uparrow": [("leftarrow", [st.Transformation.mir(135), st.Transformation.rot(90)])],
        "downarrow": [("leftarrow", [st.Transformation.mir(45), st.Transformation.rot(270)])],
    }

    graph = sr.build_pure_transformation_graph(
        symbol_keys, similarity_groups, self_symmetries, other_symmetries
    )
    # # Draw the graph, just print the edge labels.
    # pos = nx.spring_layout(graph)
    # nx.draw_networkx(graph, pos, with_labels=True)
    # edge_labels = nx.get_edge_attributes(graph, "transformations")
    # print(edge_labels)
    # plt.show()

    # Apply transitivity.
    graph = sr.apply_transitivity(graph)
    # # Draw the graph, just print the edge labels.
    # pos = nx.spring_layout(graph)
    # nx.draw_networkx(graph, pos, with_labels=True)
    # edge_labels = nx.get_edge_attributes(graph, "transformations")
    # print(edge_labels)
    # plt.show()
    assert len(graph.edges) == 18

    # raw graph.
    # {('leftarrow', 'leftarrow'): ((<Transformation.mir(0): 'mir0'>,),), ('leftarrow', 'rightarrow'): ((<Transformation.mir(0): 'mir0'>, <Transformation.rot(180): 'rot180'>),), ('leftarrow', 'uparrow'): ((<Transformation.mir(135): 'mir135'>, <Transformation.rot(270): 'rot270'>),), ('leftarrow', 'downarrow'): ((<Transformation.mir(45): 'mir45'>, <Transformation.rot(90): 'rot90'>),), ('rightarrow', 'rightarrow'): ((<Transformation.mir(0): 'mir0'>,),), ('uparrow', 'uparrow'): ((<Transformation.mir(90): 'mir90'>,),), ('downarrow', 'downarrow'): ((<Transformation.mir(90): 'mir90'>,),), ('box', 'box'): ((<Transformation.rot(90): 'rot90'>, <Transformation.rot(180): 'rot180'>, <Transformation.rot(270): 'rot270'>, <Transformation.mir(0): 'mir0'>),), ('circle', 'circle'): ((<Transformation.mir(0): 'mir0'>, <Transformation.mir(90): 'mir90'>),)}
    #
    # Transitive graph.
    # {('leftarrow', 'leftarrow'): ((< Transformation.mir(0): 'mir0' >,),),
    #  ('leftarrow', 'rightarrow'): ((< Transformation.mir(0): 'mir0' >, < Transformation.rot(180): 'rot180' >),),
    #  ('leftarrow', 'uparrow'): ((< Transformation.mir(135): 'mir135' >, < Transformation.rot(270): 'rot270' >),),
    #  ('leftarrow', 'downarrow'): ((< Transformation.mir(45): 'mir45' >, < Transformation.rot(90): 'rot90' >),),
    #  ('rightarrow', 'rightarrow'): ((< Transformation.mir(0): 'mir0' >,),),
    #  ('rightarrow', 'leftarrow'): ((< Transformation.mir(0): 'mir0' >, < Transformation.rot(180): 'rot180' >),),
    #  ('rightarrow', 'uparrow'): ((< Transformation.mir(0): 'mir0' >, < Transformation.rot(180): 'rot180' >),
    #                              (< Transformation.mir(135): 'mir135' >, < Transformation.rot(270): 'rot270' >)),
    #  ('rightarrow', 'downarrow'): ((< Transformation.mir(0): 'mir0' >, < Transformation.rot(180): 'rot180' >),
    #                                (< Transformation.mir(45): 'mir45' >, < Transformation.rot(90): 'rot90' >)),
    #  ('uparrow', 'uparrow'): ((< Transformation.mir(90): 'mir90' >,),),
    #  ('uparrow', 'leftarrow'): ((< Transformation.mir(135): 'mir135' >, < Transformation.rot(90): 'rot90' >),),
    #  ('uparrow', 'rightarrow'): ((< Transformation.mir(135): 'mir135' >, < Transformation.rot(90): 'rot90' >),
    #                              (< Transformation.mir(0): 'mir0' >, < Transformation.rot(180): 'rot180' >)),
    #  ('uparrow', 'downarrow'): ((< Transformation.mir(135): 'mir135' >, < Transformation.rot(90): 'rot90' >),
    #                             (< Transformation.mir(45): 'mir45' >, < Transformation.rot(90): 'rot90' >)),
    #  ('downarrow', 'downarrow'): ((< Transformation.mir(90): 'mir90' >,),),
    #  ('downarrow', 'leftarrow'): ((< Transformation.mir(45): 'mir45' >, < Transformation.rot(270): 'rot270' >),),
    #  ('downarrow', 'rightarrow'): ((< Transformation.mir(45): 'mir45' >, < Transformation.rot(270): 'rot270' >),
    #                                (< Transformation.mir(0): 'mir0' >, < Transformation.rot(180): 'rot180' >)),
    #  ('downarrow', 'uparrow'): ((< Transformation.mir(45): 'mir45' >, < Transformation.rot(270): 'rot270' >),
    #                             (< Transformation.mir(135): 'mir135' >, < Transformation.rot(270): 'rot270' >)),
    #  ('box', 'box'): ((
    #                       < Transformation.rot(90): 'rot90' >, < Transformation.rot(180): 'rot180' >, < Transformation.rot(270): 'rot270' >, < Transformation.mir(0): 'mir0' >),),
    #  ('circle', 'circle'): ((< Transformation.mir(0): 'mir0' >, < Transformation.mir(90): 'mir90' >),)}


def test_graph_creation() -> None:

    start = time()
    # g = sr.load_graph()
    # print(f"Graph creation took {1000 * (time() - start):.2f} ms.")
    #
    # start = time()
    # g = sr.apply_transitivity(g)
    symbol_data = sr.SymbolData()
    g = symbol_data.graph
    print(f"Transitivity took {1000 * (time() - start):.2f} ms.")

    return
    # Try to emulate finding leaders.
    start = time()
    leader_mapping = {}
    # If a node has an outgoing edge with no transformation, that is it's leader.
    for node in g.nodes:
        if g.out_degree(node) >= 1 and g.in_degree(node) == 0:
            # Search for the target node.
            for target in g.successors(node):
                if "leader" in g[node][target]:
                    leader_mapping[node] = target
                    break
    print(f"Leader mapping took {1000 * (time() - start):.2f} ms.")
    similarity_groups = sr.load_symbol_metadata_similarity_groups()
    assert leader_mapping == sr.construct_to_leader_mapping(similarity_groups)

    # # Find the edge with the most transformations.
    # # The most is the product of the lengths of the transformations.
    # max_edge = None
    # max_product = 0
    # max_layer_edge = None
    # max_layers = 0
    # for edge in g.edges(data=True):
    #     product = 1
    #     for transformation in edge[2]["transformations"]:
    #         product *= len(transformation)
    #     if product > max_product:
    #         max_edge = edge
    #         max_product = product
    #     if len(edge[2]["transformations"]) > max_layers:
    #         max_layer_edge = edge
    #         max_layers = len(edge[2]["transformations"])
    # print(f"Max edge: {max_edge}, product: {max_product}")
    # # Plot the distribution of transformation lengths.
    # transformation_lengths = []
    # for edge in g.edges(data=True):
    #     product = 1
    #     for transformation in edge[2]["transformations"]:
    #         product *= len(transformation)
    #     transformation_lengths.append(product)
    #
    # # Calculate frequency of each length.
    # from collections import Counter
    #
    # print(Counter(transformation_lengths))
    # # Make a bar chart of the frequencies.
    # sorted_by_product = sorted(Counter(transformation_lengths).items())
    # x, y = zip(*sorted_by_product)
    # plt.bar(x, y)
    # plt.show()
    #
    # print(f"Max layer edge: {max_layer_edge}, layers: {max_layers}")
    # transformation_layers = []
    # for edge in g.edges(data=True):
    #     transformation_layers.append(len(edge[2]["transformations"]))
    # print(Counter(transformation_layers))
    # # Make a bar chart of the frequencies.
    # sorted_by_layers = sorted(Counter(transformation_layers).items())
    # x, y = zip(*sorted_by_layers)
    # plt.bar(x, y)
    # plt.show()

    # return
    # Trim out all nodes that have no incoming edges to simplify the structure.
    # for node in list(g.nodes):
    #     if not g.edges(node):
    #         g.remove_node(node)
    # Display the graph.
    pos = nx.spring_layout(g)
    # Move MnSymbol-_backapprox to the corner
    pos["MnSymbol-_backapprox"] = (-1, -1)
    pos["MnSymbol-_nbackapprox"] = (-1, -1.1)
    # Color the nodes not in the leader mapping.
    colors = []
    for node in g.nodes:
        if node in leader_mapping:
            colors.append("blue")
        else:
            colors.append("red")
    # Make the edges with the leader property dashed
    # Max edge: ('latex2e-_bigcirc', 'latex2e-_bigcirc', {'transformations': (
    # (<Transformation.rot(45): 'rot45'>, <Transformation.rot(90): 'rot90'>, <Transformation.rot(135): 'rot135'>, <Transformation.rot(180): 'rot180'>, <Transformation.rot(225): 'rot225'>, <Transformation.rot(270): 'rot270'>, <Transformation.rot(315): 'rot315'>, <Transformation.mir(0): 'mir0'>, <Transformation.mir(45): 'mir45'>, <Transformation.mir(90): 'mir90'>, <Transformation.mir(135): 'mir135'>)
    # ,)}), product: 11

    edge_styles = []
    edge_colors = []
    edge_alpha = []
    for edge in g.edges:
        if "leader" in g[edge[0]][edge[1]]:
            edge_styles.append("dashed")
            edge_colors.append("black")
            edge_alpha.append(0.0)
        elif "negation" in g[edge[0]][edge[1]]:
            edge_styles.append("dotted")
            edge_colors.append("green")
            edge_alpha.append(1)
        else:
            edge_styles.append("solid")
            edge_colors.append("black")
            edge_alpha.append(0.0)
    # nx.draw_networkx(g, pos, node_color=colors, label=True)
    nx.draw_networkx_edges(g, pos, edge_color=edge_colors, style=edge_styles, alpha=edge_alpha)
    nx.draw_networkx_labels(g, pos)
    nx.draw_networkx_nodes(g, pos, node_color=colors)

    plt.show()


def test_simplify_transform() -> None:
    """
    Test the transformation simplification.
    """
    i = st.Transformation.identity()
    r90 = st.Transformation.rot(90)
    r180 = st.Transformation.rot(180)
    r270 = st.Transformation.rot(270)
    m0 = st.Transformation.mir(0)
    m90 = st.Transformation.mir(90)
    m135 = st.Transformation.mir(135)

    assert i.merge(i) == i
    assert i.merge(r90) == r90
    assert m0.merge(i) == m0

    assert r90.merge(r90) == r180
    assert r90.merge(r180) == r270
    assert r90.merge(r270) == i

    assert m0.merge(m0) == i
    assert m0.merge(m90) == [m0, m90]

    assert st.simplify_transformations((i, r90, r180, r270, m0, m90, m135)) == (r180, m0, m90, m135)
    assert st.simplify_transformations((i, r90, r180, i, i, r270, r180, m0, m0)) == ()
    assert st.simplify_transformations((r90, r90, m0, m0, i)) == (r180,)
    assert st.simplify_transformations((i,)) == ()


def test_for_accidental_similarity() -> None:
    """
    Look for symbols that have an empty transformation path to another symbol.
    These should've been in a similarity relation instead.
    Such relations are in the graph, but they have the leader property set to true.
    """
    graph = sr.SymbolData().graph
    for source, dest, data in graph.edges(data=True):
        if source == dest:
            continue
        if not data.get("transformations", False):
            assert (
                "leader" in data or "negation" in data or "inside" in data
            ), f"Edge {source}->{dest} has no transformations."


def test_negation_similarity_disjunct() -> None:
    symbol_data = sr.SymbolData()
    # Look at the negations and make sure none of them
    # are in a similarity relation.
    # We do this by checking that no node has an incoming negation edge
    # at the same time as having an outgoing leader edge to a node,
    # which also has an incoming negation edge.
    for node in symbol_data.graph.nodes:
        leader = None
        for pred in symbol_data.graph.predecessors(node):
            if "leader" in symbol_data.graph[pred][node]:
                leader = pred
                break
        else:
            continue

        # We can't have them both posses negations.
        for pred in symbol_data.graph.predecessors(leader):
            if "negation" in symbol_data.graph[pred][leader]:
                break
        else:
            continue

        # Ok, now this one doesn't get to have a negation.
        for pred in symbol_data.graph.predecessors(node):
            assert (
                "negation" not in symbol_data.graph[pred][node]
            ), f"Node {node} has a negation and a leader with a negation."


def test_symbol_data_class() -> None:
    start = time()
    symbol_data = sr.SymbolData()
    print(f"SymbolData creation took {1000 * (time() - start):.2f} ms.")
    for symbol in symbol_data.leaders:
        options = symbol_data.all_paths_to_symbol(symbol)
        assert options, f"Symbol {symbol} has no paths."
        ancestors = symbol_data.all_symbols_to_symbol(symbol)
        assert len(set(ancestors)) == len(ancestors), f"Symbol {symbol} has duplicate ancestors."
        assert symbol in ancestors, f"Symbol {symbol} can't reach itself!"
        # No transformation list should contain identity.
        for source, trans, neg in options:
            assert not any(
                t.is_identity for t in trans
            ), f"Symbol {source}->{symbol} contains identity transformation in {trans}"
            # Negations can't be self loops.
            if neg:
                assert source != symbol, f"Symbol {source} is a negation of itself."
