from dataclasses import dataclass

from grasp import FndArray
import numpy as np

@dataclass
class Node:
    _solution: FndArray
    _depth: int

    @property
    def depth(self) -> int:
        return self._depth

    @depth.setter
    def depth(self, depth: int) -> None:
        self._depth = depth

    @property
    def solution(self) -> FndArray:
        return self._solution

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Node):
            return (
                np.array_equal(self.solution, other.solution)
                and self.depth == other.depth
            )
        return False
