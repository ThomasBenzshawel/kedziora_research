import trimesh
import collections

from edges import Edge


def is_manifold_edge_check(mesh: trimesh.Trimesh):
    edge_counts = collections.Counter()
    for face in mesh.faces:
        for i in range(3):
            j = (i + 1) % 3  # Cycle through face edges
            v1 = tuple(mesh.vertices[face[i]])
            v2 = tuple(mesh.vertices[face[j]])
            
            edge = Edge(v1, v2)
            edge_counts[edge] += 1
    
    for edge, count in edge_counts.items():
        if count > 2:
            return False
        
    return True