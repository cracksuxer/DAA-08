from typing import List, Literal, Tuple, Union
from grasp import GRASP, FndArray
from constructiva import Constructiva
from rich.console import Console
from rich.traceback import install
from dataclasses import dataclass, field
from Node import Node
import numpy as np
import heapq

console = Console()
install(show_locals=True, theme="monokai")

# @profile
def objective_function(solution: FndArray) -> float:
    pairwise_distances = np.triu(np.linalg.norm(solution[:, None] - solution, axis=-1), k=1)
    return float(np.sum(pairwise_distances))

def calculate_euclidean_distance(point1: FndArray, point2: FndArray) -> float:
    return float(np.linalg.norm(point1 - point2))

def add_element_to_solution(elements: FndArray, element: Union[None, FndArray]) -> FndArray:
    if element is not None:
        elements = np.append(elements, element)
    return elements
        
@dataclass
class BranchAndBound():
    other_algorithm: Union[GRASP, Constructiva]
    _option: Union[Literal[0], Literal[1]] = 0
    problem_points: FndArray = field(default_factory=lambda: np.array([], dtype=[('x', 'f8'), ('y', 'u4')]))
    active_nodes: List[Node] = field(default_factory=list)
    n_nodes: int = 0

    def set_algorithm(self, initial_solution_algorithm: Union[GRASP, None]) -> None:
        if initial_solution_algorithm is not None:
            self.other_algorithm = initial_solution_algorithm

    @property
    def option(self) -> Union[Literal[0], Literal[1]]:
        return self._option

    @option.setter
    def option(self, option: Union[Literal[0], Literal[1]]) -> None:
        self._option = option

    def get_amount_of_nodes(self) -> int:
        return self.n_nodes
    
    def sort_by_option(self, active_nodes: List[Node]) -> List[Node]:
        if self._option == 0:  # Smallest Z
            active_nodes.sort(key=lambda node: objective_function(node.solution))
        elif self._option == 1:  # Deeper node
            active_nodes.sort(key=lambda node: node.depth)
        return active_nodes
    
    def generate_leaf(self, problem: FndArray, parent: FndArray, depth: int, solution_size_m: int) -> List[Node]:
        children: List[Node] = []
        selected_element: FndArray = np.array([])

        # Add to the possible solution the corresponding elements of the parent because of the current depth
        for i in range(depth):
            selected_element = add_element_to_solution(selected_element, parent[i])

        # Delete every element of the generated solution that are inside the initial set of elements
        initial_x: FndArray = problem.copy()
        for i in range(len(selected_element)):
            for j in range(len(initial_x)):
                if np.array_equal(selected_element[i], initial_x[j]):
                    initial_x = np.delete(initial_x, j)
                    break

        # Set a coordinate axis
        selected_element = np.resize(selected_element, depth + 1)
        for i in range(len(initial_x)):
            selected_element = np.insert(selected_element, depth, initial_x[i])
            initial_without_selected = initial_x.copy()
            initial_without_selected = initial_without_selected[i:]
            new_leaf = Node(self.generate_best_solution(initial_without_selected, selected_element, solution_size_m), depth + 1)
            children.append(new_leaf)

        return children

    def generate_best_solution(self, initial_x: FndArray, selected_element: FndArray, solution_size_m: int) -> FndArray:
        best_candidate: Tuple[Union[FndArray, None], float] = (None, 0.0) # Each position corresponds to the element and its z value
        candidate: float = 0.0
        
        while len(selected_element) < solution_size_m:
            for item in initial_x:
                aux_solution = selected_element.copy()
                aux_solution = add_element_to_solution(aux_solution, item)
                candidate = objective_function(aux_solution)
                if candidate > best_candidate[1]:
                    best_candidate = item, candidate
            selected_element = add_element_to_solution(selected_element, best_candidate[0])
            best_candidate = (None, 0.0)

        objective_function(selected_element)
        return selected_element
    
    def solve(self, solution_size: int) -> Tuple[FndArray, int, float]:
        if self.other_algorithm is None:
            raise Exception("No algorithm was set")
        lower_bound, _ = self.other_algorithm.solve()
        
        node: Node = Node(lower_bound, 0)
        self.active_nodes.append(node)
        self.n_nodes = 1
        
        while len(self.active_nodes) > 0:
            self.sort_by_option(self.active_nodes)
            actual_node = self.active_nodes.pop()
            
            while actual_node.depth < solution_size:
                nodes: List[Node] = self.generate_leaf(self.problem_points, lower_bound, actual_node.depth, solution_size)
                if actual_node in self.active_nodes:
                    self.active_nodes.remove(actual_node)
                self.sort_by_option(nodes)
                for node in nodes:
                    self.active_nodes.append(node)
                self.n_nodes += len(nodes)
                
                actual_node = self.active_nodes.pop()
                
            actual_node_z = objective_function(actual_node.solution)
            if actual_node_z > objective_function(lower_bound):
                lower_bound = actual_node.solution
                
            for i in range(len(self.active_nodes)):
                if objective_function(self.active_nodes[i].solution) > objective_function(lower_bound):
                    self.active_nodes.remove(self.active_nodes[i])
                    i -= 1
                
        return lower_bound, self.n_nodes, objective_function(lower_bound)
            
                