import itertools
import time
from typing import List, Literal, Union

import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich.traceback import install
from rich_tools import table_to_df  # type: ignore

from BandB import BranchAndBound
from constructiva import Constructiva
from grasp import GRASP, FndArray

console = Console()
install(show_locals=True)

class InvalidNNumberException(Exception):
    '''
    Exception raised when the number of n is invalid
    '''
    def __init__(self, actual: int, expected: int):
        self.actual = actual
        self.expected = expected
        super().__init__(f"Invalid n number: {actual}, expected: {expected}")

def input_parser(file: str) -> List[List[float]]:
    '''
    Parses the input file and returns a list of points
    '''
    parsed = []
    with open(file, "r", encoding="utf-8") as file_handle:
        lines = file_handle.readlines()
        _max = int(lines[0]) + 2
        nrange = int(lines[1])
        for i in range(2, _max):
            line = lines[i].replace(",", ".").replace("\t", " ").rstrip().split(" ")
            if len(line) > nrange:
                raise InvalidNNumberException(len(line), nrange)
            parsed.append([float(x) for x in line])

    return parsed


def main() -> None:
    
    constructiva_table = create_table(
        "Constructiva Voraz", "z", "S", "CPU"
    )
    grasp_table = create_table("GRASP", "Iter", "|LRC|", "z")
    grasp_table.add_column("S", style="dim", no_wrap=True)
    grasp_table.add_column("CPU", style="dim", no_wrap=True)

    branchbound_table = create_table("Branch and Bound", "Z", "S", "CPU")
    branchbound_table.add_column("generated nodes", style="dim", no_wrap=True)

    problem_list_name: List[str] = [
        'max_div_15_2.txt',
        'max_div_15_3.txt',
        'max_div_20_2.txt',
        'max_div_20_3.txt',
        'max_div_30_2.txt',
        'max_div_30_3.txt'
    ]

    problem_list = [input_parser(f"./data/{problem_name}") for problem_name in problem_list_name]

    # for problem, m in itertools.product(problem_list, range(2, 5)):
    #     index: int = problem_list.index(problem)
    #     data: FndArray = np.array(problem)
    #     start_1: float = time.perf_counter()
    #     S, Z = Constructiva(data, m).solve()
    #     end_1 = time.perf_counter()
    #     total_time_constr: float = end_1 - start_1

    #     constructiva_table.add_row(problem_list_name[index], str(len(data)), str(len(data[0])), str(m), str(Z), str(S), str(total_time_constr))

    # constructiva_df = table_to_df(constructiva_table)
    # constructiva_df.to_csv("./resultados/constructiva.csv", index=False)

    # for problem, m, _iter, lrc in itertools.product(problem_list, range(2, 6), range(10, 30, 10), range(2, 4)):
    #     index_grasp: int = problem_list.index(problem)
    #     data_grasp: FndArray = np.array(problem)
    #     start = time.perf_counter()
    #     solution, diversity = GRASP(data_grasp, _iter, m, lrc).solve()
    #     end = time.perf_counter()
    #     total_time = end - start
    #     grasp_table.add_row(problem_list_name[index_grasp], str(len(data_grasp)), str(len(data_grasp[0])), str(m), str(_iter), str(lrc), str(diversity), '{' + ', '.join(str(s) for s in solution) + '}', str(total_time))

    # grasp_df = table_to_df(grasp_table)
    # grasp_df.to_csv("./resultados/grasp.csv", index=False)

    csv_title_ending: List[str] = [
        "_constructiva_csp"
        "_grasp_csp", 
        "_consutructiva_bp", 
        "_grasp_bp"
    ]

    for i in range(1, 5):
        for problem, m in itertools.product(problem_list, range(2, 4)):
            index: int = problem_list.index(problem)
            data: FndArray = np.array(problem)
            
            algorithm: Union[Constructiva, GRASP]
            option: Union[Literal[0], Literal[1]]
            if i in [1, 3]:
                algorithm = Constructiva(data, m)
                option = 0
            if i in [2, 4]:
                algorithm = GRASP(data, 10, m, 2)
                option = 1
                
            start_1: float = time.perf_counter()
            solution, n_nodes, Z = BranchAndBound(algorithm, option, data).solve(m)
            end_1 = time.perf_counter()
            total_time_constr: float = end_1 - start_1

            branchbound_table.add_row(problem_list_name[index], str(len(data)), str(len(data[0])), str(m), str(Z), str(solution), str(total_time_constr), str(n_nodes))
            
            console.print(f"Processing problem {index+1} of {len(problem_list)} | Processing m={m} |", end=" ")
            console.print("[progress]{task.percentage:>3.0f}%[/progress]", end=" ")
            console.print("[bold green]✔[/bold green]")
            
        bb_df = table_to_df(branchbound_table)
        bb_df.to_csv(f"./resultados/BranchAndBound{csv_title_ending[i]}.csv", index=False)
        
    # data_grasp: FndArray = np.array(problem_list[0])
    # grasp = GRASP(data_grasp, 10, 3, 4)
    # solution, n_nodes, Z = BranchAndBound(grasp, 0, data_grasp).solve(4)
    # print(solution, n_nodes, Z)

            

def create_table(title: str, arg1: str, arg2: str, arg3: str) -> Table:
    result = Table(title=title)
    result.add_column("Problema", style="dim", no_wrap=True)
    result.add_column("n", style="dim", no_wrap=True)
    result.add_column("K", style="dim", no_wrap=True)
    result.add_column("m", style="dim", no_wrap=True)
    result.add_column(arg1, style="dim", no_wrap=True)
    result.add_column(arg2, style="dim", no_wrap=True)
    result.add_column(arg3, style="dim", no_wrap=True)

    return result

if __name__ == '__main__':
    main() 