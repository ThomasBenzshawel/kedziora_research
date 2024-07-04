import numpy as np
import hashlib


class Edge:
    def __init__(self, v1: tuple, v2: tuple, directed=False):
        self._v1 = v1
        self._v2 = v2
        self._directed = directed

    @property
    def v1(self) -> float:
        return self._v1
    
    @property
    def v2(self) -> float:
        return self._v2
    
    @property
    def directed(self) -> bool:
        return self._directed
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Edge):
            if self._directed:
                return self._v1 == other.v1 and self._v2 == other.v2
            
            return self._v1 == other.v1 and self._v2 == other.v2 or self._v1 == other.v2 and self._v2 == other.v1
        
        return False
    
    def __hash__(self) -> int:
        v1 = np.array(self.v1)
        v2 = np.array(self.v2)
        origin = np.array([0,0,0])

        v1_dist = np.linalg.norm(v1-origin)
        v2_dist = np.linalg.norm(v2-origin)

        combined_array = None

        if v1_dist <= v2_dist:
            combined_array = np.concatenate((v1, v2))
        else:
            combined_array = np.concatenate((v2, v1))

        return hash(tuple(combined_array))
        
    def __str__(self) -> str:
        return str(f'[{self.v1}, {self.v2}]')