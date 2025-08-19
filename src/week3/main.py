import bisect
import random
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from itertools import islice
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd

from .utils import argmax_random_tie, argmin_random_tie, define_init, probability

T = TypeVar("T")
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
df = pd.read_csv(ASSETS_DIR / "problem_data.csv")


locations = {}
for x1, x2 in zip(
    df["LocationName"], zip(df["LocationX"], df["LocationY"], strict=False), strict=False
):
    locations[x1] = x2


def define_map(
    locations: dict[str, tuple[int, int]],
) -> tuple[list[str], dict[str, dict[str, float]]]:
    city_locations = locations
    all_cities = []
    distances: dict[str, dict[str, float]] = {}

    for city in city_locations.keys():
        distances[city] = {}
        all_cities.append(city)
    all_cities.sort()

    for name_1, coordinates_1 in city_locations.items():
        for name_2, coordinates_2 in city_locations.items():
            distances[name_1][name_2] = float(
                np.linalg.norm(
                    [coordinates_1[0] - coordinates_2[0], coordinates_1[1] - coordinates_2[1]]
                )
            )
            distances[name_2][name_1] = distances[name_1][name_2]
    return (all_cities, distances)


all_cities, distances = define_map(locations)

num = int(input("input K as the number of items in an inner dictionary to display: "))
for city in distances.keys():
    city_distances = distances[city]

    # Use islice to get the first `num` items and put them into a dictionary
    top_k_items = dict(islice(city_distances.items(), num))

    print(
        f"The target city is {city}, \nIts distances to the first {num} cities are {top_k_items}:"
    )


def generate_k_neighbouring_paths(path: list[str], k: int = 5) -> list[list[str]]:
    """Generate k neighbors of a given path."""
    print(f"The given path is {path}")  # 입력받은 경로 출력

    current_path = path[:]  # 원본 경로 복사
    neighbouring_path = path[:]  # 이웃 경로 복사
    neighbouring_paths = []  # 결과를 담을 리스트

    for _ in range(k):  # k번 반복
        left = random.randint(1, len(current_path) - 1)  # 구간의 왼쪽 인덱스 랜덤 선택 (시작은 1)
        right = random.randint(
            1, len(current_path) - 1
        )  # 구간의 오른쪽 인덱스 랜덤 선택 (시작은 1)

        if left > right:  # 인덱스 순서 정렬
            left, right = right, left
        # 선택된 구간을 뒤집어서 이웃 경로 생성
        neighbouring_path[left : right + 1] = reversed(neighbouring_path[left : right + 1])
        neighbouring_paths.append(neighbouring_path)  # 결과 리스트에 추가
        neighbouring_path = path[:]  # 다음 반복을 위해 원본 경로로 초기화

    print(f"The generated k neighbours are {neighbouring_paths}")  # 생성된 이웃 경로들 출력

    return neighbouring_paths


def cost_fun(path: list[str]) -> float:
    total_distance = 0.0

    for i in range(len(path) - 1):  # 경로의 각 도시 쌍에 대해 반복
        city1 = path[i]  # 현재 도시
        city2 = path[i + 1]  # 다음 도시
        total_distance += distances[city1][city2]  # 두 도시 사이의 거리 더하기

    start_city = path[0]  # 시작 도시
    final_city = path[-1]  # 마지막 도시
    total_distance += distances[start_city][final_city]  # 시작~마지막 도시 거리(순환 경로) 더하기

    return total_distance


start_city = "Arad"
print(f"The start city is {start_city}.")


# PART2 HC
def hill_climbing(
    city_list: list[str],
    neighbour_function: Callable[[list[str]], list[list[str]]],
    f: Callable[[list[str]], float],
    patience: int = 500,
    max_iters: int = 1000,
) -> tuple[list[str], float]:
    """Perform hill climbing optimization on a list of cities.

    city_list: initial solution
    neighbor_function: a function that produces the neighbors of a given sample per iteration
    f: objective function or a cost function to MINIMIZE
    patience: consecutive iterations with no improvement allowed
    max_iters: max iterations before giving up
    """
    print(f"The input city_list is \n{city_list}.\n The cost is {f(city_list)}")
    current = city_list
    no_improve = 0
    iters = 0

    while iters < max_iters:
        print(f"\nThe number of iteration is {iters}")
        neighbours = neighbour_function(current)
        print(f"The neighbors of the current path is \n{neighbours}")
        iters += 1

        if not neighbours:
            break

        neighbour = argmin_random_tie(neighbours, key=lambda node: cost_fun(node))
        print(f"The best path in the neighbours in \n{neighbour}")

        if no_improve >= patience:
            break

        print(
            "The costs of neighbour path and current path are "
            f"{cost_fun(neighbour)} and {cost_fun(current)}"
        )
        if cost_fun(neighbour) < cost_fun(current):
            print("neighbor path is better than current path. SWAP!")
            current = neighbour
            print(f"new current path is \n{current}")
            no_improve = 0
        else:
            no_improve += 1
            print("neighbour path is not better than current path. DON'T Swap!")
            print(f"new current path is \n{current}")

    return (current, cost_fun(current))


init = define_init(all_cities, start_city)
print(
    f"The initial solution is \n{init} and \nthe cost of this initial solution is {cost_fun(init)}"
)

max_iterations = 10  # For illustration only
n_times = 5
print(
    f"The max_iterations is {max_iterations} and the times for no-change before termination is {n_times}"
)

solution, cost = hill_climbing(
    init, generate_k_neighbouring_paths, cost_fun, n_times, max_iterations
)
print(f"The best solution found is \n{solution} and the corresponding cost is \n{cost}")

# PART3 GA


def test_duplicated(x: list[str]) -> bool:
    duplicated = False
    duplicate = dict(Counter(x))
    """dict() converts the argument into a dictionary"""
    for value in duplicate.values():
        if value > 1:
            duplicated = True
    return duplicated


def roulette_wheel_selection(
    population: list[list[str]], fitness_fun: Callable[[list[str]], float], num_selections: int = 1
) -> list[list[str]]:
    """Perform roulette wheel selection.

    Args:
        population (list): List of individuals.
        fitness_fun: fitness function used to calculate fitness values of individuals
        num_selections (int): Number of individuals to select.

    Returns:
        list: Selected individuals.

    """
    # Compute fitness values of the given population
    fitnesses: list[float] = []
    for i in population:
        temp = fitness_fun(i)
        fitnesses.append(temp)

    # Compute the total fitness
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        msg = "Total fitness is zero. Can't perform roulette wheel selection."
        sys.exit(msg)

    # Normalize the fitness values to probabilities
    probabilities = [f / total_fitness for f in fitnesses]

    # Compute cumulative probabilities
    cumulative_probs: list[float] = []
    cumulative_sum = 0.0
    for p in probabilities:
        cumulative_sum += p
        cumulative_probs.append(cumulative_sum)

    # Select individuals
    selected: list[list[str]] = []
    for _ in range(num_selections):
        r = random.random()  # Use random for secure random number generation
        for idx, individual in enumerate(population):
            if r <= cumulative_probs[idx]:
                selected.append(individual)
                break

    return selected


def generate_for_non_duplication(x: list[str], y: list[str], start_point: str) -> list[str]:
    new_gen1 = x[:]
    new_gen2 = y[:]
    new_gen1.remove(start_point)
    new_gen2.remove(start_point)
    n = len(new_gen1)
    new_gen = []
    while True:
        c = random.randint(0, n)
        new_gen = new_gen1[:c] + new_gen2[c:]
        if test_duplicated(new_gen):
            pass
        else:
            new_gen.insert(0, start_point)
            break
    return new_gen


def mutate_for_non_duplication(x: list[str], pmut: float, start_point: str) -> list[str]:
    if random.uniform(0, 1) >= pmut:
        print("The random number is not less than the mutation rate, do not mutation")
        return x
    print("The random number is less than the mutation rate, perform mutation")
    mutate_population = x[:]
    mutate_population.remove(start_point)
    random.shuffle(mutate_population)
    mutate_population.insert(0, start_point)
    return mutate_population


def recombine_for_non_duplication(x: list[str], y: list[str], start_point: str) -> list[str]:
    new_gen1 = x[:]
    new_gen2 = y[:]
    new_gen1.remove(start_point)
    new_gen2.remove(start_point)
    n = len(new_gen1)
    duplicated = False
    c = random.randrange(0, n)
    new_gen = new_gen1[:c] + new_gen2[c:]
    new_gen.insert(0, start_point)
    duplicated = test_duplicated(new_gen)
    if duplicated:
        new_gen = generate_for_non_duplication(x, y, start_point)
    return new_gen


def weighted_sampler(seq: Sequence[T], weights: Sequence[float]) -> Callable[[], T]:
    """Return a weighted random sampler function that picks a random element from seq according to the given weights."""
    totals: list[float] = []  # a list
    for w in weights:
        totals.append(w + totals[-1] if totals else w)  # builds a cumulative weight list
    return lambda: seq[
        bisect.bisect(totals, random.uniform(0, totals[-1]))
    ]  # Uses bisect.bisect() to find which interval it falls into


def select(
    population: list[list[str]], fitness_fn: Callable[[list[str]], float]
) -> list[list[str]]:
    fitnesses: list[float] = list(map(fitness_fn, population))
    print(
        f"The current population is \n{population} and the corresponding fitness values are "
        f"{fitnesses}"
    )
    sorted_population = sorted(population, key=fitness_fn)
    sorted_fitnesses: list[float] = list(map(fitness_fn, sorted_population))
    print(
        f"The sorted population by fitness is \n{sorted_population} and the corresponding "
        f"fitness values are {sorted_fitnesses}"
    )
    sampler = weighted_sampler(sorted_population, fitnesses)
    return [sampler() for i in range(2)]


def recombine(x: list[str], y: list[str]) -> list[str]:
    n = len(x)
    c = random.randrange(0, n)
    return x[:c] + y[c:]


def mutate(x: list[str], gene_pool: list[str], pmut: float) -> list[str]:
    if random.uniform(0, 1) >= pmut:
        return x

    n: int = len(x)
    g: int = len(gene_pool)
    c: int = random.randrange(0, n)
    r: int = random.randrange(0, g)

    new_gene: str = gene_pool[r]
    return [*x[:c], new_gene, *x[c + 1 :]]


def fitness_threshold(
    fitness_fn: Callable[[list[str]], float],
    f_thres: float,
    population: list[list[str]],
) -> list[str] | None:
    if not f_thres:
        return None

    fittest_individual = max(population, key=fitness_fn)
    if fitness_fn(fittest_individual) >= f_thres:
        print("A path exceeds the threshold fitness, search process terminates")
        return fittest_individual

    return None


def init_population_for_non_duplication(
    pop_number: int, gene_pool: list[str], state_length: int, start: str
) -> list[list[str]]:
    g = len(gene_pool)
    initial_population = gene_pool.copy()
    population = []
    for i in range(pop_number):
        random.shuffle(initial_population)
        initial_population.remove(start)
        initial_population.insert(0, start)
        population.append(initial_population)
        print(f"The {i}th individual in the population is\n{initial_population}")
        initial_population = gene_pool.copy()
    return population


def genetic_algorithm(
    population: list[list[str]],
    fitness_fn: Callable[[list[str]], float],
    f_thres: float,
    ngen: int,
    pmut: float,
    start: str,
    non_duplication: bool = True,
    gene_pool: list[str] | None = None,
) -> tuple[list[str], int]:
    print(f"The initial population is \n{population}")
    print(f"The number of generations is {ngen} and the mutation rate is {pmut}")

    # Use cost_fun as the fitness function for consistency
    fitness_fun = fitness_fn

    if non_duplication:
        for generation in range(ngen):
            print(f"\nThe {generation!s}th generation:")
            new_population = []
            for i in range(len(population)):
                print(f"\nGenerate {i}th individual in the new population")
                individual1, individual2 = roulette_wheel_selection(population, fitness_fun, 2)

                print(
                    f"The two selected individuals using selection operator are \n{individual1} and {individual2} "
                    f"and their fitness values are {fitness_fun(individual1)} and {fitness_fun(individual2)}."
                )
                individual_recombined = recombine_for_non_duplication(
                    individual1, individual2, start
                )
                print(
                    f"The individual created from crossover or recombination is \n{individual_recombined} "
                    f"and its fitness is {fitness_fun(individual_recombined)}"
                )
                individual_final = mutate_for_non_duplication(individual_recombined, pmut, start)
                print(
                    f"The {i}th individual in the new population is \n{individual_final} "
                    f"and the fitness is {fitness_fun(individual_final)}"
                )
                new_population.append(individual_final)

            population = new_population
            print(f"The new generation is \n{population}")

            # stores the fittest individual genome in the current population
            current_best = max(population, key=fitness_fun)
            if generation % 100 == 0:
                print(f"Generation: {generation!s}\t\tFitness: {fitness_fun(current_best)}\r")

            # compare the fitness of the current best individual to f_thres
            fittest_individual = fitness_threshold(fitness_fun, f_thres, population)

            # if fitness is greater than or equal to f_thres, we terminate the algorithm
            if fittest_individual:
                return fittest_individual, generation
    else:
        if gene_pool is None:
            raise ValueError("gene_pool must be provided when non_duplication is False.")
        for generation in range(ngen):
            population = [
                mutate(recombine(*select(population, fitness_fun)), gene_pool, pmut)
                for i in range(len(population))
            ]
            # stores the individual genome with the highest fitness in the current population
            current_best = max(population, key=fitness_fun)
            print(
                f"Current best: {current_best}\t\tGeneration: {generation!s}\t\tFitness: {fitness_fun(current_best)}\r",
                end="",
            )

            # compare the fitness of the current best individual to f_thres
            fittest_individual = fitness_threshold(fitness_fun, f_thres, population)
            print(fittest_individual)

            # if fitness of the best individual is greater than or equal to f_thres, we terminate the algorithm
            if fittest_individual:
                return fittest_individual, generation
    print(f"\nThe final solution is: {max(population, key=fitness_fun)}")
    return max(population, key=fitness_fun), generation


def fitness_fun(path: list[str]) -> float:
    return 5000 - cost_fun(path)


gene_pool = all_cities
max_population = 5
population = init_population_for_non_duplication(
    max_population, gene_pool, len(gene_pool), start_city
)
init_representation = argmax_random_tie(population, key=lambda x: fitness_fun(x))
print(
    f"The initial solution is:\n{init_representation}, and \nThe cost is: \n{cost_fun(init_representation)}"
)

mutation_rate = 0.05
max_gen = 50  # for illustration only
f_thres = 4000
solution, generation = genetic_algorithm(
    population, fitness_fun, f_thres, max_gen, mutation_rate, start_city
)

print(f"\nThe final solution is: \n{solution} and \nThe generation is {generation}.")
print(
    f"\nThe fitness of the final solution is: {fitness_fun(solution)} and Its cost is: {cost_fun(solution)}"
)


# PART4 SA
def simulated_annealing(
    init: list[str],
    energy: Callable[[list[str]], float],
    neighbor: Callable[[list[str]], list[list[str]]],
    schedule: Callable[[], Callable[[int], float]],
) -> tuple[list[str], float]:
    """Simulated annealing optimization.

    init: initial solution (path of city names)
    energy: cost function returning lower-is-better score
    neighbor: function generating neighbor solutions from current
    schedule: returns a temperature function T(t) mapping step->temperature
    """
    print(f"The initial solution is \n{init}")
    current = init
    # for t in range(sys.maxsize):
    for t in range(10):  # for illustration purpose
        T = schedule()(t)
        print(f"\nThe {t}th step and temperature is {T}")
        if T == 0:
            break
        neighbors = neighbor(current)

        if not neighbors:
            break
        # next_choice = neighbors
        next_choice = random.choice(neighbors)
        print(f"The neighbor solution is \n{next_choice}")
        delta_e = energy(next_choice) - energy(current)
        print(
            f"The energy of neighbor is {energy(next_choice)} and energy of current is {energy(current)} and the difference is {delta_e}"
        )

        if delta_e < 0:
            print("neighbour is better than current, swap")
            current = next_choice
        else:
            temp = probability(np.exp(-delta_e / T))
            if temp is True:
                print(
                    "neighbour is not better than current, but random number is less than the predefined probability, also swap"
                )
                current = next_choice
            else:
                print("No swap!")
    return (current, energy(current))


init = define_init(all_cities, start_city)
print(init)
print(cost_fun(init))


def exp_schedule(k: int = 20, lam: float = 0.005, limit: int = 10000) -> Callable[[int], float]:
    return lambda t: (k * np.exp(-lam * t) if t < limit else 0)


def energy(
    state: list[str],
) -> float:
    # this is the cost function which calculates the total distances
    # between city pairs in a solution
    cost: float = 0.0
    for i in range(len(state) - 1):
        cost += distances[state[i]][state[i + 1]]
    cost += distances[state[0]][state[-1]]
    return cost


def neighbouring_path(current: list[str]) -> list[str]:
    neighbouring_path = current
    path_length = len(current)
    # This line has been removed to eliminate dead code.
    left = random.randint(1, path_length - 1)
    right = random.randint(1, path_length - 1)
    if left > right:
        left, right = right, left
    print(f"current path is \n{current} and the cost is {energy(current)}")
    neighbouring_path[left : right + 1] = reversed(neighbouring_path[left : right + 1])
    print(f"neighbouring path is \n{neighbouring_path} and the cost is {energy(neighbouring_path)}")
    return neighbouring_path


solution, cost = simulated_annealing(init, energy, generate_k_neighbouring_paths, exp_schedule)
print(f"The best solution is: \n{solution} and \nThe cost is: {cost}")
