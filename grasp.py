"""
Grasp algorithm module.
"""
from xml.dom import Node
import numpy as np
from rich.traceback import install
from typing import List, Tuple, Optional, Union, cast

FndArray = np.ndarray[np.dtype[np.float_], np.dtype[np.uint]]
install(show_locals=True)


def generate_neighbour_add(solution: FndArray, points: FndArray) -> List[FndArray]:
    """
    Generates all neighbour solution by adding a new point from the available points.
    """
    neighbour_solutions = []
    for point in points:
        if np.any(np.all(solution == point, axis=1)):
            continue
        neighbour_solution = np.vstack((solution, point))
        neighbour_solutions.append(neighbour_solution)
    return neighbour_solutions


def generate_neighbour_del(solution: FndArray) -> List[FndArray]:
    """
    Generates all neighbour solution by removing a point from the solution.
    """
    neighbour_solutions = []
    for i in range(len(solution)):
        neighbour_solution = np.delete(solution, i, axis=0)
        neighbour_solutions.append(neighbour_solution)
    return neighbour_solutions

def calculate_euclidean_distance(point1: FndArray, point2: FndArray) -> float:
    """Calculates the Euclidean distance between two n-dimensional points."""
    return float(np.sqrt(np.sum((point1 - point2) ** 2)))

def calculate_gravity_center(points: FndArray) -> FndArray:
    """Calculates the gravity center of a list of n-dimensional points."""
    return np.mean(points, axis=0) # type: ignore

def best_kvalues(distances: List[Tuple[int, float]], k: int = 3) -> List[Tuple[int, float]]:
    """Returns the k elements with the highest values in a list of tuples (index, value)."""
    sortedDistances = sorted(distances, key=lambda x: x[1], reverse=True)
    return sortedDistances[:k]


def generate_neighbour_swap(solution: FndArray, points: FndArray) -> List[FndArray]:
    """
    Generates all neighbour solution by swapping two points, one from the solution
    and the other from the available points.
    """
    neighbour_solutions = []
    for point in points:
        if np.any(np.all(solution == point, axis=1)):
            continue
        for i in range(len(solution)):
            neighbour_solution = np.copy(solution)
            neighbour_solution[i] = point
            neighbour_solutions.append(neighbour_solution)
    return neighbour_solutions

def ditance_to_closest_point(
    point: FndArray, points: List[FndArray]
) -> Tuple[FndArray, np.float_]:
    """
    Returns the distance to the closest point.
    """
    distancias = np.linalg.norm(points - point, axis=1)
    closes_point_index = np.argmin(distancias)
    distance_closes_point = np.linalg.norm(point - points[closes_point_index])
    return (point, distance_closes_point)

class GRASP:
    """
    GRASP algorithm implementation.
    """
    def __init__(self, data_points: FndArray, max_iter: int, max_construct_len: int, lrc_size: int):
        self._data_points = data_points
        self._max_iter = max_iter
        self._lrc_size = lrc_size
        self._max_construct_len = max_construct_len
        self._greedy_solution: Optional[FndArray] = None
        self._local_solution: Optional[FndArray] = None
        self._best_solution: Optional[FndArray] = None

        points_len = len(data_points)
        self._num_clusters = max(int(points_len * 0.1), 2)

    def __repr__(self) -> str:
        return f"GRASP({self._num_clusters}, {self._data_points}, {self._max_iter})"

    def __construct(self) -> None:
        """Constructs an initial solution for the GRASP algorithm."""
        self._greedy_solution = np.empty(
            0, dtype=np.dtype([("floats", np.float_), ("ints", np.uint)])
        )
        initialElements = self._data_points.copy()
        solution: List[FndArray] = []
        gravityCenter = calculate_gravity_center(initialElements)
        while len(solution) < self._max_construct_len:
            distances = [(i, calculate_euclidean_distance(gravityCenter, initialElements[i])) for i in range(len(initialElements))]
            bestKDistances = best_kvalues(distances)
            random = np.random.randint(len(bestKDistances))
            chosenElement = np.copy(initialElements[bestKDistances[random][0]])
            initialElements = np.delete(initialElements, bestKDistances[random][0], axis=0)
            solution.append(chosenElement)
            solution_array = np.concatenate(solution, axis=0)
            gravityCenter = calculate_gravity_center(solution_array)
        self._greedy_solution = np.array(solution)
            
    def __postprocess(self) -> None:
        if self._greedy_solution is not None:
            current_solution = self._greedy_solution.copy()
        else:
            raise ValueError("Greedy solution is not defined")

        current_solution = self.__swap_points(current_solution)
        self._local_solution = current_solution

    def __swap_points(self, current_solution: FndArray) -> FndArray:
        s_data_points = self._data_points.copy()
        while True:
            neighbour_solutions = generate_neighbour_swap(
                current_solution, s_data_points
            )
            neighbour_solutions.sort(key=self.__calc_cost)
            best_neighbour_solution: FndArray = neighbour_solutions[0]
            while self.__equal_cost(current_solution, best_neighbour_solution):
                neighbour_solutions.pop(0)
            best_neighbour_solution = neighbour_solutions[0]
            if self.__stop_condition(current_solution, best_neighbour_solution):
                break
            current_solution = best_neighbour_solution
        return current_solution

    def __equal_cost(
        self, solution1: FndArray, solution2: FndArray
    ) -> bool:
        return self.__calc_cost(solution1) == self.__calc_cost(solution2)

    def __stop_condition(
        self, current_solution: FndArray, best_neighbour_solution: FndArray
    ) -> bool:
        return self.__calc_cost(current_solution) < self.__calc_cost(
            best_neighbour_solution
        )

    def __update_best_solution(self) -> None:
        if self._best_solution is None or self.__calc_cost(
            cast(FndArray, self._local_solution)
        ) < self.__calc_cost(self._best_solution):
            self._best_solution = self._local_solution

    def __calc_cost(self, cluster_locs: FndArray) -> Union[int, float]:
        """
        Calcula la diversidad de una solución al Maximum Diversity Problem.
        
        Parámetros:
        - elem: una matriz de tamaño (n, k) que representa los elementos a seleccionar.
        - solucion: una lista de índices de elementos seleccionados. Cada índice debe estar en el rango
        [0, n-1].
        
        Retorna:
        - La diversidad de la solución, medida como la distancia promedio de los elementos seleccionados
        al centro de gravedad del conjunto.
        """
        z: float = 0
        for i in range(len(cluster_locs)):
            for j in range(i + 1, len(cluster_locs)):
                z += calculate_euclidean_distance(cluster_locs[i], cluster_locs[j])
        return z

    def solve(self) -> Tuple[FndArray, float]:
        """
        Solve the problem.
        """
        for _ in range(self._max_iter):
            self.__construct()
            self.__postprocess()
            self.__update_best_solution()
            
        diversity = self.__calc_cost(cast(FndArray, self._best_solution))

        return (cast(FndArray, self._best_solution), diversity)
