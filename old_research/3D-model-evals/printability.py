import trimesh
import numpy as np

from manifold import manifold_edge_check


def wall_thickness(stl_mesh: trimesh.base.Trimesh) -> int:
    point_inside = stl_mesh.centroid  

    num_rays = 200
    directions = np.random.rand(num_rays, 3) - 0.5 

    ray_origins = np.tile(point_inside, (num_rays, 1))  
    ray_locations, index_ray, index_tri = stl_mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=directions
    )

    distances = np.linalg.norm(ray_locations - point_inside, axis=1)
    thicknesses = np.sort(distances)

    min_thickness = thicknesses.min()

    return min_thickness


def is_printable(stl_mesh: trimesh.base.Trimesh, testing=False) -> dict:
    smallest_valume = 0.001
    thinist_wall = 0.2

    manifold = None
    positive_volume = None
    no_intersections = None
    thick_walls = None

    score = 15

    try:
        if type(stl_mesh) == trimesh.scene.scene.Scene:
            raise InvalidSTLException('STL file has invalid format')

        manifold = manifold_edge_check(stl_mesh.faces, stl_mesh.vertices)

        if not manifold:
            score -= 3
            trimesh.smoothing.filter_taubin(stl_mesh, lamb=0.5, nu=0.53, iterations=10)
            manifold = manifold_edge_check(stl_mesh.faces, stl_mesh.vertices)
            
        if not manifold:
            score -= 3
        
        positive_volume = stl_mesh.volume > smallest_valume
        if not positive_volume:
            score -= 3

        no_intersections = stl_mesh.is_winding_consistent
        if not no_intersections:
            score -= 4

        thick_walls = wall_thickness(stl_mesh) >= thinist_wall
        if not thick_walls:
            score -= 2
            
    except InvalidSTLException as e:
        print(f'STL not formatted correctly')
        return 0
    except Exception as e:
        print(f'Error processing STL')
        return 0

    if testing:
        return {'manifold': manifold, 'positive_volume': positive_volume, 
            'no_intersections': no_intersections, 'thick_walls': thick_walls,
            'score': score}

    return score


class InvalidSTLException(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(message)