import itertools
import time
from typing import List

import numpy as np
import pandas as pd  # type: ignore
from rich import print
from rich.console import Console
from rich.table import Table
from rich.traceback import install
from rich_tools import table_to_df  # type: ignore

from constructiva import Constructiva
from grasp import GRASP

console = Console()
install(show_locals=True)

FndArray = np.ndarray[np.dtype[np.float_], np.dtype[np.uint]]

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

    problem_list_name: List[str] = [
        'max_div_15_2.txt',
        'max_div_15_3.txt',
        'max_div_20_2.txt',
        'max_div_20_3.txt',
        'max_div_30_2.txt',
        'max_div_30_3.txt'
    ]

    problem_list = [input_parser(f"./data/{problem_name}") for problem_name in problem_list_name]

    for problem, m in itertools.product(problem_list, range(2, 5)):
        index: int = problem_list.index(problem)
        data: FndArray = np.array(problem)
        start_1: float = time.perf_counter()
        S, Z = Constructiva(data, m).constructivo_voraz()
        end_1 = time.perf_counter()
        total_time_constr: float = end_1 - start_1

        constructiva_table.add_row(problem_list_name[index], str(len(data)), str(len(data[0])), str(m), str(Z), str(S), str(total_time_constr))

    constructiva_df = table_to_df(constructiva_table)
    constructiva_df.to_csv("./resultados/constructiva.csv", index=False)

    for problem, m, _iter, lrc in itertools.product(problem_list, range(2, 6), range(10, 30, 10), range(2, 4)):
        index_grasp: int = problem_list.index(problem)
        data_grasp: FndArray = np.array(problem)
        start = time.perf_counter()
        solution, diversity = GRASP(data_grasp, _iter, m, lrc).solve()
        end = time.perf_counter()
        total_time = end - start
        grasp_table.add_row(problem_list_name[index_grasp], str(len(data_grasp)), str(len(data_grasp[0])), str(m), str(_iter), str(lrc), str(diversity), '{' + ', '.join(str(s) for s in solution) + '}', str(total_time))

    grasp_df = table_to_df(grasp_table)
    grasp_df.to_csv("./resultados/grasp.csv", index=False)
            

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