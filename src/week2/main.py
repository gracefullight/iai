from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .pq import PriorityQueue

# Build path relative to this file: src/week2 -> src/assets/Romania-map.png
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def get_path(filename: str) -> str:
    return str(ASSETS_DIR / filename)


def heuristic(
    u: tuple[float, float], v: tuple[float, float]
) -> float:  # u and v are tuples (two coordinates)
    return float(np.sqrt(sum((x - y) ** 2 for x, y in zip(u, v, strict=True))))


def gbfs_path(graph: nx.Graph, start: str, goal: str) -> list[str] | None:
    """Find a path from start to goal using greedy best first Search."""
    # Check the start node
    print(f"start node: {start}")

    # Initialise a priority queue object to be the frontier -- an empty priority list
    pq: PriorityQueue[tuple[str, list[str]]] = PriorityQueue()
    # Initialise the frontier by the start location, it's path
    pq.push((start, [start]), heuristic(cities[start], cities[goal]))

    # initialise the explored set to be an empty set
    visited = set()

    index = 0
    while not pq.is_empty():
        print(f"Interation {index}")
        # Check the highest priority node in the frontier before expansion if the frontie is not empty
        print(
            "A new expansion. Element with the highest priority in the frontier before expansions:"
        )
        print(pq.peek())

        (vertex, path) = pq.pop()
        index = index + 1

        # Check the node chosen to expand
        print(f"Node chosen to expand: {vertex}")

        # Check the frontier after removing a node to expand
        if not (pq.is_empty()):
            print(
                f"Element with the highest priority in the frontie after removing a node {vertex} to expand:"
            )
            print(pq.peek())
        else:
            print("frontie is empty")

        if vertex in visited:
            continue
        visited.add(vertex)

        # Check the explored set or the closed list again after adding a node, visited
        print(f"Elements in the closed list after adding a node {vertex}:")
        for element in visited:
            print(element)

        if vertex == goal:
            return path

        for neighbor in graph[vertex]:
            print("neighbor is ", neighbor)
            if neighbor in visited:
                continue
            pq.push((neighbor, [*path, neighbor]), heuristic(cities[neighbor], cities[goal]))

            # Check the frontier after adding a node
            print(
                f"Element with the highest priority in the frontie after adding a neighbour node {neighbor}:"
            )
            print(pq.peek())

    return None  # Return None if no path is found


def ucs_path(graph: nx.Graph, start: str, goal: str) -> list[str] | None:
    """Find a path from start to goal using uniform cost Search."""
    # Check the start node
    print(f"start node: {start}")

    pq: PriorityQueue[tuple[str, list[str]]] = PriorityQueue()
    pq.push((start, [start]), 0)

    visited: set[str] = set()

    index = 0
    while not pq.is_empty():
        print(f"Interation {index}")
        # Check the highest priority node in the frontier before expansion if the frontie is not empty
        print(
            "A new expansion. Element with the highest priority in the frontier before expansions:"
        )
        print(pq.peek())

        (vertex, path) = pq.pop()
        index = index + 1

        # Check the node chosen to expand
        print(f"Node chosen to expand: {vertex}")

        # Check the frontier after removing a node to expand
        if not (pq.is_empty()):
            print(
                f"Element with the highest priority in the frontie after removing a node {vertex} to expand:"
            )
            print(pq.peek())
        else:
            print("frontie is empty")

        if vertex in visited:
            continue
        visited.add(vertex)

        # Check the closed list again after adding a node, visited
        print(f"Elements in the closed list after adding a node {vertex}:")
        for element in visited:
            print(element)

        if vertex == goal:
            return path

        for neighbor in graph[vertex]:
            print("neighbor is ", neighbor)
            if neighbor in visited:
                continue
            next_path = [*path, neighbor]
            pq.push((neighbor, next_path), nx.path_weight(graph, next_path, "weight"))

            # Check the frontier after adding a node
            print(
                "Element with the highest priority in the frontie after adding a neighbour "
                "node "
                f"{neighbor}:"
            )
            print(pq.peek())

    return None  # Return None if no path is found


df_cities = pd.read_csv(get_path("cities_coordinates.csv"))
df_matrix = pd.read_csv(get_path("cities_correlation_matrix.csv"))

print(df_cities.head())

cities = {}

name = df_cities["LocationName"]
locationX = df_cities["LocationX"]
locationY = df_cities["LocationY"]

location = zip(locationX, locationY, strict=True)

for x1, x2 in zip(name, location, strict=True):
    cities.update({x1: x2})

print(cities)

neighbors = []

for i in range(df_matrix.shape[0]):
    x = df_matrix.iloc[i, :]

    relation_key = x.iloc[0]

    for j in range(1, df_matrix.shape[1]):
        if j > i and x.iloc[j] != 0:
            temp_tuple = (relation_key, df_matrix.columns[j], int(x.iloc[j]))
            neighbors.append(temp_tuple)

print(f"City neighbors data: {neighbors}")
print(f"\n{len(neighbors)} number of elements in the list of city neighbors")

G = nx.Graph()
G.add_weighted_edges_from(neighbors)
nx.draw(G, with_labels=True)
plt.show()

print("=" * 80)


source = "Arad"
print(f"The start city is: {source}")

# Define the goal state/city. goal
goal = "Bucharest"

print(f"The goal city is: {goal}")

path = gbfs_path(G, source, goal)
print("Path from source node to goal node using GBFS algorithm:", path)
print("=" * 80)


path = ucs_path(G, source, goal)
print("Path from source node to goal node using UCS algorithm:", path)
print("=" * 80)
