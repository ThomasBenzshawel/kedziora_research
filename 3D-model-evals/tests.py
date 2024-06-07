import os
import glob
import trimesh

from printability import is_printable
from manifold import manifold_edge_check


def check_valid_thingiverse(base_directory: str):
    invalid_stls = []
    manifold_count = 0
    print_count = 0
    try:
        for category in os.listdir(base_directory):
            print(category)
            category_path = os.path.join(base_directory, category, 'stls')

            if os.path.isdir(category_path):
                for obj in os.listdir(category_path):
                    stl_file_pattern = os.path.join(category_path, obj, '*.stl')

                    for stl_file in glob.glob(stl_file_pattern):

                        new_entry = is_printable(stl_file)

                        if type(new_entry) == str:
                            invalid_stls.append(new_entry)
                        elif new_entry != None:
                            if new_entry['manifold']:
                                print_count += 1

                        mesh = trimesh.load(stl_file)
                        manifold = manifold_edge_check(mesh.faces, mesh.vertices)

                        if not manifold:
                            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.53, iterations=10)
                            manifold = manifold_edge_check(mesh.faces, mesh.vertices)

                        if manifold:
                            manifold_count += 1

                        print('Manifold: {}\nPrintability: {}'.format(manifold, new_entry['manifold']))

        print(f'Manifold Total Passed: {manifold_count}\nPrintability Total Passed: {print_count}')
                        
    except (FileNotFoundError, PermissionError) as e: 
        print(f"Error accessing files: {e}") 
    except KeyboardInterrupt as e:
        print('---Exited successfully---')
    except Exception as e:
        print('Could not process STL')


if __name__ == '__main__':
    check_valid_thingiverse('/data/csc4801/KedzioraLab/thingiverse/')