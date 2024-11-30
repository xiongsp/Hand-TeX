# networkx reimplementation of utils stuff.
import re
from itertools import product
from collections import defaultdict
from collections import deque
from functools import cache
from importlib import resources
from pathlib import Path

import networkx as nx
from loguru import logger

import handtex.data.symbol_metadata
import handtex.data.symbols
import handtex.structures as st
from handtex.utils import resource_path

# TODO add hat/widehat= and give it symbol metadata


class SymbolData:
    """
    A class to hold and process all symbol data.
    """

    symbol_data: dict[str, st.Symbol]
    symbol_keys: list[str]
    to_leader: dict[str, str]
    leaders: list[str]
    graph: nx.DiGraph
    symbols_grouped_by_similarity: list[tuple[str, ...]]
    symbols_with_self_symmetry: list[str]
    symbols_grouped_by_transitive_symmetry: list[tuple[str, ...]]
    symmetric_symbols: list[str]
    asymmetric_symbols: list[str]

    def __init__(self):
        self.symbol_data = load_symbols()
        self.symbol_keys = list(self.symbol_data.keys())
        similarity_groups = load_symbol_metadata_similarity_groups()
        self_symmetries = load_symbol_metadata_self_symmetry()
        other_symmetries = load_symbol_metadata_other_symmetry()
        self.to_leader = construct_to_leader_mapping(similarity_groups)
        self.leaders = [key for key in self.symbol_keys if key not in self.to_leader]
        normalized_self_symmetries = normalize_self_symmetry_to_leaders(
            self_symmetries, self.to_leader
        )
        normalized_other_symmetries = normalize_other_symmetry_to_leaders(
            other_symmetries, self.to_leader, self_symmetries
        )
        graph = build_graph(
            self.symbol_keys,
            similarity_groups,
            normalized_self_symmetries,
            normalized_other_symmetries,
        )
        self.graph = apply_transitivity(graph)

        self.symbols_grouped_by_similarity = self._compute_symbols_grouped_by_similarity()
        self.symbols_with_self_symmetry = list(normalized_self_symmetries.keys())
        self.symbols_grouped_by_transitive_symmetry = (
            self._compute_symbols_grouped_by_transitive_symmetry()
        )
        self.asymmetric_symbols = self._compute_asymmetric_symbols()
        self.symmetric_symbols = [
            key for key in self.symbol_keys if key not in self.asymmetric_symbols
        ]

    def __getitem__(self, item):
        return self.symbol_data[item]

    def __contains__(self, item):
        return item in self.symbol_data

    @property
    def all_keys(self):
        return self.symbol_keys

    @cache
    def has_self_symmetry(self, symbol_key: str) -> bool:
        """
        Check if a symbol has self-symmetry.
        """
        return self.graph.has_edge(symbol_key, symbol_key)

    @cache
    def has_other_symmetry(self, symbol_key: str) -> bool:
        """
        Check if a symbol has other-symmetry.
        We define this as having an outgoing edge to another symbol
        which has a non-empty transformations attribute.
        """
        for source, dest, data in self.graph.out_edges(symbol_key, data=True):
            if source != dest and data["transformations"]:
                return True
        return False

    def _compute_symbols_grouped_by_similarity(self) -> list[tuple[str, ...]]:
        """
        Get a full list of symbols grouped by similarity.
        Those inside a group are similar to each other.

        :return: A list of tuples of symbol keys.
        """
        groups = []
        for node in self.leaders:
            # Find all nodes that are reachable from this node with a leader edge.
            similar_symbols = filter(
                lambda n: "leader" in self.graph[n][node], nx.ancestors(self.graph, node)
            )
            groups.append(tuple((node, *similar_symbols)))
        return groups

    def _compute_symbols_grouped_by_transitive_symmetry(self) -> list[tuple[str, ...]]:
        """
        Get a full list of symbols grouped by transitive symmetry.
        Transitive symmetry is a weakly connected component of the graph.
        """
        return list(tuple(component) for component in nx.weakly_connected_components(self.graph))

    def _compute_asymmetric_symbols(self) -> list[str]:
        """
        Get a list of symbols that have edges with non-empty transformations.
        """
        empty_transformations = ((),)
        asymmetric_symbols = []
        for node in self.symbol_keys:
            for _, _, data in self.graph.edges(node, data=True):
                if data["transformations"] != empty_transformations:
                    asymmetric_symbols.append(node)
                    break
        return asymmetric_symbols

    def all_paths_to_symbol(
        self, symbol_key: str
    ) -> list[tuple[str, tuple[st.Transformation, ...]]]:
        """
        Get all possible paths to a symbol key, starting from some symbol and applying transformations.
        The tuple of transformations here is a pre-flattened permutation of all possibilities.
        """
        # The heavy lifting was already done with transitivity, so each path is just a single inward edge.
        paths = []
        # First we need to see IF this symbol has any in-edge relations whatsoever.
        # We're only checking leaders, so that excludes the non-leaders that have no
        # path to them, but we must still consider loner symbols that have nothing.
        if not self.graph.in_edges(symbol_key):
            return [(symbol_key, ())]

        # We want to insert self symmetries at the end of each path's transformations, if it
        # isn't a self-loop already. The self-symmetries are a single layer of transformations,
        # so we can unpack them with a single [0] index.
        if symbol_key in self.symbols_with_self_symmetry:
            self_symmetries = self.graph[symbol_key][symbol_key]["transformations"][0]
        else:
            self_symmetries = ()
            # Since there is no self-loop to iterate over for in_edges, we must add the
            # self-symmetric identity ourselves.
            paths.append((symbol_key, ()))
        # We still want to retain the possibility of no self-symmetry being applied, so we
        # add the identity transformation to the tuple of options.
        self_symmetries = ((st.Transformation.identity(), *self_symmetries),)

        for source, dest, data in self.graph.in_edges(symbol_key, data=True):
            if source != dest:
                # Slap on the self-symmetries at the end of the transformations.
                transformation_list = data["transformations"] + self_symmetries
            else:
                # Otherwise use the self-symmetries that are expanded by the identity.
                transformation_list = self_symmetries

            # We must permute over the transformation options.
            # The transformations are tuples of tuples. We choose one from each tuple.
            for transformations in product(*transformation_list):
                # Attempt to simplify the transformations,
                # and check if this is a duplicate.
                simplified_transformations = st.simplify_transformations(transformations)
                if (source, simplified_transformations) not in paths:
                    paths.append((source, simplified_transformations))
                # else:
                #     logger.warning(
                #         f"Duplicate path found: {source} -> {symbol_key}. {transformations} -> {simplified_transformations}"
                #     )
        return paths

    def all_symbols_to_symbol(self, symbol_key: str) -> list[str]:
        """
        Get all symbols that can reach the given symbol key.
        This will never include the symbol key itself.
        """
        return list(nx.ancestors(self.graph, symbol_key))

    @cache
    def get_similarity_group(self, symbol_key: str) -> tuple[str, ...]:
        """
        Finds with similarity group the symbol belongs to.

        :param symbol_key: The symbol key.
        :return: The similarity group.
        """
        for group in self.symbols_grouped_by_similarity:
            if symbol_key in group:
                return group
        # This must never happen.
        raise KeyError(f"Symbol {symbol_key} not found in any symmetry group.")

    @cache
    def get_symmetry_group(self, symbol_key: str) -> tuple[str, ...]:
        """
        Finds with symmetry group the symbol belongs to.

        :param symbol_key: The symbol key.
        :return: The symmetry group.
        """
        for group in self.symbols_grouped_by_transitive_symmetry:
            if symbol_key in group:
                return group
        # This must never happen.
        raise KeyError(f"Symbol {symbol_key} not found in any symmetry group.")


def load_symbols() -> dict[str, st.Symbol]:
    """
    Load the symbols from the symbols.json file.

    :return: A dictionary of key to symbols.
    """

    with resource_path(handtex.data.symbol_metadata, "symbols.json") as symbols_file:
        symbol_list = st.Symbol.from_json(symbols_file)
    return {symbol.key: symbol for symbol in symbol_list}


def select_leader_symbols(
    symbol_keys: list[str], lookalikes: dict[str, tuple[str, ...]]
) -> list[str]:
    """
    Select the leader symbols from the list of symbol keys.
    Leaders either have no lookalikes or are the first symbol among a set of lookalikes.

    :param symbol_keys: List of symbol keys.
    :param lookalikes: Dictionary mapping symbol keys to sets of lookalike symbol keys.
    :return: List of leader symbols.
    """
    # Iterate over the lookalike's keys first, to ensure that
    # the first symbol in the lookalike set is always the leader.
    # Otherwise, if we iterated over all symbols, which have random order,
    # a different symbol might appear first and be selected as the leader.
    leaders = []
    for key in lookalikes:
        if not any(k in leaders for k in lookalikes[key]):
            leaders.append(key)
    for key in symbol_keys:
        if key not in leaders and key not in lookalikes:
            leaders.append(key)

    return leaders


def load_symbol_metadata_similarity_groups() -> list[tuple[str, ...]]:
    """
    Load the metadata for symbols.
    Similarity is an equivalence relation, each line encodes an equivalence class.
    The first symbol in each line is the leader of the class.

    File format: each line contains a space separated list of symbol keys.

    :return: A list of tuples of symbol keys.
    """
    with resources.path(handtex.data.symbol_metadata, "") as metadata_dir:
        metadata_dir = Path(metadata_dir)
    files = list(metadata_dir.glob("similar*"))

    equivalence_classes = []
    for file in files:
        with file.open("r") as f:
            for line in f:
                similar_keys = line.strip().split()
                equivalence_classes.append(tuple(similar_keys))

    return equivalence_classes


def load_symbol_metadata_similarity() -> dict[str, tuple[str, ...]]:
    """
    Load the metadata for symbols.
    File format: each line contains a space separated list of symbol keys.

    :return: A dictionary mapping symbol keys to sets of symbol keys.
    """
    with resources.path(handtex.data.symbol_metadata, "") as metadata_dir:
        metadata_dir = Path(metadata_dir)
    files = list(metadata_dir.glob("similar*"))

    symbol_map = {}
    for file in files:
        with file.open("r") as f:
            for line in f:
                similar_keys = line.strip().split()
                for key in similar_keys:
                    if key not in symbol_map:
                        symbol_map[key] = tuple(k for k in similar_keys if k != key)
                    else:
                        logger.error(f"Duplicate symbol key found: {key}")

    return symbol_map


def construct_to_leader_mapping(similarity_groups: list[tuple[str, ...]]) -> dict[str, str]:
    """
    Construct a mapping from symbols to their leaders.
    Any symbol not in the mapping is its own leader.

    :param similarity_groups: List of tuples of symbol keys.
    :return: A dictionary mapping symbols to their leaders.
    """
    to_leader = {}
    for group in similarity_groups:
        leader = group[0]
        for key in group[1:]:
            to_leader[key] = leader
    return to_leader


def load_symbol_metadata_self_symmetry() -> dict[str, list[st.Transformation]]:
    """
    Load the metadata for symbol self-symmetric transformations.

    File format:
    symbol_key: symmetry1 symmetry2 symmetry3 ...

    :return: A dictionary mapping symbol keys to lists of symmetries.
    """
    with resource_path(handtex.data.symbol_metadata, "symmetry_self.txt") as file_path:
        with open(file_path, "r") as file:
            lines = file.readlines()

    symmetries = {}
    for line in lines:
        parts = line.strip().split()
        key = parts[0][:-1]  # Remove the colon at the end.
        symmetries[key] = [st.Transformation(sym) for sym in parts[1:]]
    return symmetries


def normalize_self_symmetry_to_leaders(
    self_symmetries: dict[str, list[st.Transformation]], to_leader: dict[str, str]
) -> dict[str, list[st.Transformation]]:
    """
    We want to ensure that the self-symmetries only pertain to leader symbols.
    If a symbol has different self-symmetries than its leader, we log an error.
    """
    normalized_symmetries = {}
    for key, symmetries in self_symmetries.items():
        leader = to_leader.get(key, key)
        if leader not in normalized_symmetries:
            normalized_symmetries[leader] = symmetries
        else:
            if normalized_symmetries[leader] != symmetries:
                logger.error(f"Leader {leader} has different self-symmetries than symbol {key}.")
    return normalized_symmetries


def load_symbol_metadata_other_symmetry() -> dict[str, list[tuple[str, list[st.Transformation]]]]:
    """
    Load the metadata for symbol other-symmetric transformations.
    The file must not contain two-way assignments.
    These are instead generated here, ensuring that the symmetry is always two-way.

    File format:
    symbol_key_from -- transform1 transform2 ... transformN --> symbol_key_to

    # Importantly the list defined in the file describes various single transformations to
    # apply to the symbol to receive the target symbol. Chained transformations are not supported
    # in the file format, but are generated here to transitively apply all symmetries.

    :return: A dictionary mapping symbol keys to tuples of target symbol key and list of transforms.
    """
    symmetries: dict[str, list[tuple[str, list[st.Transformation]]]]

    with resource_path(handtex.data.symbol_metadata, "symmetry_other.txt") as file_path:
        with open(file_path, "r") as file:
            lines = file.readlines()

    pattern = re.compile(r"(\S+) -- (.*?) ?--> (\S+)")
    symmetries = defaultdict(list)
    for line in lines:
        match = pattern.match(line)
        if not match:
            logger.error(f"Failed to parse line: {line.strip()}")
            continue
        key_from = match.group(1)
        key_to = match.group(3)
        transformation_options = [st.Transformation(sym) for sym in match.group(2).split()]
        transformation_options_inverted = [t.invert() for t in transformation_options]
        symmetries[key_from].append((key_to, transformation_options))
        symmetries[key_to].append((key_from, transformation_options_inverted))

    return symmetries


def normalize_other_symmetry_to_leaders(
    other_symmetries: dict[str, list[tuple[str, list[st.Transformation]]]],
    to_leader: dict[str, str],
    self_symmetries: dict[str, list[st.Transformation]],
) -> dict[str, list[tuple[str, list[st.Transformation]]]]:
    """
    We want to ensure that the other-symmetries only pertain to leader symbols.
    If a symbol has different other-symmetries than its leader, we log an error,
    unless the leader has no other-symmetries at all. In that case, we copy the other-symmetries
    to the leader.

    Important: We must strip out any other-symmetries that operate within the same similarity group.

    Sometimes the other-symmetries end up describing self-symmetries once the similarity groups are taken into account.
    We want to assert that these other-symmetries are present in self-symmetries instead.

    We also want to ensure that the target symbols are leaders.

    :param other_symmetries: Dictionary mapping symbol keys to tuples of target symbol key and list of transforms.
    :param to_leader: Dictionary mapping symbols to their leaders.
    :param self_symmetries: Dictionary mapping symbol keys to lists of self-symmetries. Used to check for conflicts.
    :return: A dictionary mapping leader symbols to tuples of target symbol key and list of transforms.
    """
    normalized_symmetries = {}
    for key, symmetries in other_symmetries.items():
        leader = to_leader.get(key, key)
        for target, transforms in symmetries:
            target_leader = to_leader.get(target, target)
            # Check if this is, applying similarity, actually a self-symmetry.
            if leader == target_leader:
                if set(transforms) != set(self_symmetries[leader]):
                    logger.error(
                        f"Leader {leader} has different self-symmetries than symbol {key} for target {target}."
                    )
                continue
            if leader not in normalized_symmetries:
                normalized_symmetries[leader] = [(target_leader, transforms)]
            elif (target_leader, transforms) not in normalized_symmetries[leader]:
                # Ensure we don't already have a mapping for the same target but different transforms.
                # This would indicate a conflict.
                if any(t == target_leader for t, _ in normalized_symmetries[leader]):
                    logger.error(
                        f"Leader {leader} has different other-symmetries than similar symbol {key} "
                        f"for target {target} (lead by {target_leader}).\n"
                        f"- Leader: {other_symmetries.get(leader, [])}\n"
                        f"- Similar: {other_symmetries.get(key, [])}"
                    )
                normalized_symmetries[leader].append((target_leader, transforms))
    return normalized_symmetries


def build_graph(
    symbol_keys: list[str],
    similarity_groups: list[tuple[str, ...]],
    normalized_self_symmetries: dict[str, list[st.Transformation]],
    normalized_other_symmetries: dict[str, list[tuple[str, list[st.Transformation]]]],
) -> nx.DiGraph:
    """
    Build a complete graph of all known symbol relations.

    Nodes are the symbol keys.
    Edges are transformations between symbols. Each edge has a transformations attribute,
    which is a tuple of tuples of transformations. From each tuple in the tuple, one transformation
    must be applied to reach the target symbol.

    Important: Use the normalized self- and other-symmetries to ensure that the graph is consistent.

    :param symbol_keys: List of symbol keys.
    :param similarity_groups: List of tuples of symbol keys.
    :param normalized_self_symmetries: Dictionary mapping symbol keys to lists of self-symmetries.
    :param normalized_other_symmetries: Dictionary mapping symbol keys to tuples of target symbol key and list of transforms.
    :return: A directed graph of symbol relations.
    """
    graph = nx.DiGraph()

    # Add all nodes
    for key in symbol_keys:
        graph.add_node(key)

    # Load similarity groups as edges with an identity transformation.
    # These relations are NOT stored symmetrically, we want to keep the
    # focus on the leader symbols.
    # Due to transitivity being applied later on, the exact leader won't be clear
    # anymore, so we mark this special case with a leader attribute.
    for group in similarity_groups:
        leader = group[0]
        for key in group[1:]:
            graph.add_edge(key, leader, transformations=(), leader=True)

    # Load self-symmetries as self-loops with the given transformations as a list.
    for key, symmetries in normalized_self_symmetries.items():
        graph.add_edge(key, key, transformations=(tuple(symmetries),))

    # Load other-symmetries as edges with the given transformations as a list.
    for key, symmetries in normalized_other_symmetries.items():
        for target, transforms in symmetries:
            graph.add_edge(key, target, transformations=(tuple(transforms),))

    return graph


def load_graph() -> nx.DiGraph:
    """
    Load the graph of symbol relations.
    """
    symbol_keys = list(load_symbols().keys())
    similarity_groups = load_symbol_metadata_similarity_groups()
    self_symmetries = load_symbol_metadata_self_symmetry()
    other_symmetries = load_symbol_metadata_other_symmetry()
    to_leader = construct_to_leader_mapping(similarity_groups)
    normalized_self_symmetries = normalize_self_symmetry_to_leaders(self_symmetries, to_leader)
    normalized_other_symmetries = normalize_other_symmetry_to_leaders(
        other_symmetries, to_leader, self_symmetries
    )
    return build_graph(
        symbol_keys, similarity_groups, normalized_self_symmetries, normalized_other_symmetries
    )


def apply_transitivity(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Apply transitivity to the graph of symbol relations.
    This means that if A -> B and B -> C, then we add A -> C.

    Adding the transformation information requires concatenating the transformations.

    :param graph: The graph of symbol relations.
    :return graph: The graph with transitive relations added.
    """
    # Edges have a transformations attribute: tuple[tuple[Transformation, ...], ...]
    # List to collect new edges without modifying the graph while iterating.
    new_edges = []

    # Iterate over each node in the graph to start a BFS for transitive reachability.
    for start_node in graph.nodes:
        # Use a deque for BFS traversal.
        queue = deque([(start_node, ())])  # (current_node, path_transformations)

        # Track visited nodes with a set.
        visited = {start_node}

        while queue:
            current_node, current_transformations = queue.popleft()

            # Iterate over all successors of the current node.
            for neighbor in graph.successors(current_node):
                # Avoid self-loops and repeated paths.
                if neighbor in visited:
                    continue

                # Get the transformations for the edge between current_node and neighbor.
                edge_data = graph.get_edge_data(current_node, neighbor)
                edge_transformations = edge_data["transformations"]

                # Concatenate the current path's transformations with the edge's transformations.
                new_path_transformations = current_transformations + edge_transformations

                # Register this node as visited in this traversal to avoid cycles.
                visited.add(neighbor)

                # Add to queue for further exploration.
                queue.append((neighbor, new_path_transformations))

                # Add the transitive edge if it is not redundant.
                if not graph.has_edge(start_node, neighbor):
                    new_edges.append((start_node, neighbor, new_path_transformations))

    # Add all new transitive edges to the graph.
    for source, target, transformations in new_edges:
        graph.add_edge(source, target, transformations=transformations)

    return graph


@cache
def get_leader(symbol_key: str, graph: nx.DiGraph) -> tuple[str, bool]:
    """
    Get the leader symbol key for the given symbol key.
    We find the leader by checking for outgoing edges with the identity transformation.
    We also return a boolean indicating if the symbol is a leader.

    :param symbol_key: The symbol key.
    :param graph: The graph of symbol relations.
    :return: A tuple of the leader symbol key and True if the symbol is a leader.
    """
    for _, target, data in graph.edges(symbol_key, data=True):
        if data["transformations"] == ((st.Transformation.identity(),),):
            return target, False
    return symbol_key, True
