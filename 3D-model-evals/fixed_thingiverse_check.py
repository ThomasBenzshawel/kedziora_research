import os
import glob
import pandas as pd
import trimesh

from printability import is_printable


def check_valid_thingiverse(base_directory: str):
    df = pd.DataFrame()
    try:
        for category in os.listdir(base_directory):
            print(category)
            category_path = os.path.join(base_directory, category)

            if os.path.isdir(category_path):
                for obj in os.listdir(category_path):
                    stl_file_pattern = os.path.join(category_path, obj, '*.stl')

                    for stl_file in glob.glob(stl_file_pattern):
                        try:
                            mesh = trimesh.load(stl_file)

                            new_entry = is_printable(mesh, testing=True)

                            if new_entry != 0:
                                df_dictionary = pd.DataFrame([new_entry])
                                df = pd.concat([df, df_dictionary], ignore_index=True)
                        except (TypeError, UnicodeDecodeError, trimesh.exchange.stl.HeaderError) as e:
                            print('Unexpected Trimesh Error')


        df.to_csv('fixed_thingiverse_stls.csv', index = False, encoding='utf-8')
                        
    except (FileNotFoundError, PermissionError) as e: 
        print(f"Error accessing files: {e}") 
    except KeyboardInterrupt as e:
        print('---Exited successfully---')


if __name__ == '__main__':
    check_valid_thingiverse('/data/csc4801/KedzioraLab/fixed_thingiverse/')