from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(slots=True)
class Stage:
    matching: list[tuple[int, int]]
    bytes_total: float


def decompose_matchings(matrix: np.ndarray) -> list[Stage]:
    residual = np.asarray(matrix, dtype=float).copy()
    np.fill_diagonal(residual, 0.0)
    stages: list[Stage] = []

    while np.max(residual) > 1e-9:
        graph = nx.Graph()
        rows = [f"r{i}" for i in range(residual.shape[0])]
        cols = [f"c{j}" for j in range(residual.shape[1])]
        graph.add_nodes_from(rows, bipartite=0)
        graph.add_nodes_from(cols, bipartite=1)
        for i in range(residual.shape[0]):
            for j in range(residual.shape[1]):
                if residual[i, j] > 1e-9:
                    graph.add_edge(f"r{i}", f"c{j}")

        matching_map = nx.algorithms.bipartite.maximum_matching(graph, top_nodes=rows)
        matching: list[tuple[int, int]] = []
        for row in rows:
            col = matching_map.get(row)
            if col is not None:
                i = int(row[1:])
                j = int(col[1:])
                if residual[i, j] > 1e-9:
                    matching.append((i, j))
        if not matching:
            i, j = np.unravel_index(np.argmax(residual), residual.shape)
            matching = [(int(i), int(j))]

        weight = min(float(residual[i, j]) for i, j in matching)
        stages.append(Stage(matching=matching, bytes_total=weight))
        for i, j in matching:
            residual[i, j] = max(0.0, residual[i, j] - weight)

    return stages


def reconstruct_from_stages(stages: list[Stage], shape: tuple[int, int]) -> np.ndarray:
    matrix = np.zeros(shape, dtype=float)
    for stage in stages:
        for i, j in stage.matching:
            matrix[i, j] += stage.bytes_total
    return matrix
