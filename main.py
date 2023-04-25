import itertools
import numpy as np
import pandas as pd # type: ignore
import time
from rich import print
from rich.traceback import install
from rich.console import Console
from rich.table import Table
from typing import List
from rich_tools import table_to_df # type: ignore
from constructiva import Constructiva
import random

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

def valor_solucion(solucion, datos):
    """
    Calcula el valor de una solución al problema Maximum Diversity.
    El valor de una solución se define como la distancia media entre cada par de elementos de la solución.
    
    Args:
    - solucion (array): un array de booleanos que indica si cada elemento del conjunto de datos está o no en la solución
    - datos (array): un array bidimensional con los datos del problema
    
    Returns:
    - valor (float): el valor de la solución
    """
    n = len(datos)
    indices = np.arange(n)
    if type(solucion) == list:
        solucion = np.array(solucion)
    indices_solucion = indices[solucion]
    distancias = np.zeros((len(indices_solucion), len(indices_solucion)))

    for i, si in enumerate(indices_solucion):
        for j, sj in enumerate(indices_solucion):
            distancias[i,j] = np.sqrt(np.sum(np.power(datos[si] - datos[sj], 2)))

    return np.sum(distancias) / (
        len(indices_solucion) * (len(indices_solucion) - 1) / 2
    )

def busqueda_local(datos, solucion_inicial, max_iter):
    solucion_actual = solucion_inicial.copy()
    valor_actual = valor_solucion(solucion_actual, datos)
    
    for _ in range(max_iter):
        indice_solucion = random.randint(0, len(solucion_actual)-1)
        indice_datos = random.randint(0, len(datos)-1)
        
        nuevo_solucion = solucion_actual.copy()
        nuevo_solucion[indice_solucion] = datos[indice_datos]
        nuevo_valor = valor_solucion(nuevo_solucion, datos)
        
        if nuevo_valor > valor_actual:
            solucion_actual = nuevo_solucion.copy()
            valor_actual = nuevo_valor
            
    return solucion_actual, valor_actual


def calcular_diversidad(conjunto, solucion):
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
    seleccionados = conjunto[solucion]
    centro = np.mean(seleccionados, axis=0)
    distancias = np.linalg.norm(seleccionados - centro, axis=1)
    return np.mean(distancias)

def main() -> None:
    
    constructiva_table = Table(title="Constructiva Voraz")
    constructiva_table.add_column("Problema", style="dim", no_wrap=True)
    constructiva_table.add_column("n", style="dim", no_wrap=True)
    constructiva_table.add_column("K", style="dim", no_wrap=True)
    constructiva_table.add_column("m", style="dim", no_wrap=True)
    constructiva_table.add_column("z", style="dim", no_wrap=True)
    constructiva_table.add_column("S", style="dim", no_wrap=True)
    constructiva_table.add_column("CPU", style="dim", no_wrap=True)

    problem_list_name = [
        'max_div_15_2.txt',
        'max_div_15_3.txt',
        'max_div_20_2.txt',
        'max_div_20_3.txt',
        'max_div_30_2.txt',
        'max_div_30_3.txt'
    ]

    problem_list = [input_parser(f"./data/{problem_name}") for problem_name in problem_list_name]
    solution_list = []


    for problem, m in itertools.product(problem_list, range(2, 5)):
        index: int = problem_list.index(problem)
        data: FndArray = np.array(problem)
        start: float = time.perf_counter()
        S, Z = Constructiva(data, m).constructivo_voraz()
        end = time.perf_counter()
        total_time: float = end - start
        console.print(f"Problem {index + 1} with m:{m}:", style="bold red")
        solution_list.append(list(S))

        constructiva_table.add_row(problem_list_name[index], str(len(data)), str(len(data[0])), str(m), str(Z), str(S), str(total_time))

    constructiva_df = table_to_df(constructiva_table)
    constructiva_df.to_csv("./resultados/constructiva.csv", index=False)

    print(busqueda_local(np.array(problem_list[0]), solution_list[0], 1000))

if __name__ == '__main__':
  main()