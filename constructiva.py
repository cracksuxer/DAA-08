import numpy as np

class Constructiva():
    def __init__(self, conjunto, m) -> None:
        self.conjunto = conjunto
        self.m = m
    
    def centro_gravedad(self, conjunto):
        return np.mean(conjunto, axis=0)

    def distancia(self, x, y):
        return np.linalg.norm(x - y)

    def constructivo_voraz(self):
        n, _ = self.conjunto.shape
        S = set()
        elem = set(range(n))
        sc = self.centro_gravedad(self.conjunto[list(elem)])
        while len(S) < self.m:
            d_max = -1
            for s in elem:
                d = self.distancia(self.conjunto[s], sc)
                if d > d_max:
                    d_max = d
                    s_star = s
            S.add(s_star)
            elem.remove(s_star)
            sc = self.centro_gravedad(self.conjunto[list(S)])
            
        z = self.calcular_diversidad(list(S))
            
        return S, z

    def calcular_diversidad(self, solucion):
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
        seleccionados = self.conjunto[solucion]
        centro = np.mean(seleccionados, axis=0)
        distancias = np.linalg.norm(seleccionados - centro, axis=1)
        return np.mean(distancias)