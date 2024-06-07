import trimesh
import os
import glob
import time

from manifold_edges import is_manifold_edge_check
from manifold import manifold_edge_check


def main():
    category_path = '/data/csc4801/KedzioraLab/thingiverse/bookshelf/stls/'
    old_times = 0
    new_times = 0

    if os.path.isdir(category_path):
        for obj in os.listdir(category_path):
            obj_path = os.path.join(category_path, obj)
            stl_file_pattern = os.path.join(obj_path, '*.stl')

            print(f'Current Project: {obj_path.split("/")[-1]}')
            for stl_file in glob.glob(stl_file_pattern):
                mesh = trimesh.load(stl_file)

                print(f'Old trial start:')
                old_start = time.time()
                is_manifold_edge_check(mesh)
                old_time = time.time() - old_start
                old_times += old_time
                print(f'Results: {old_time}s\n')

                print(f'C++ trial start:')
                new_start = time.time()
                manifold_edge_check(mesh.faces, mesh.vertices)
                new_time = time.time() - new_start
                new_times += new_time
                print(f'Results: {new_time}s\n')

    print(f'Old check time: {old_times}s')
    print(f'New check time: {new_times}s')


if __name__ == '__main__':
    main()