import trimesh
import os
import glob


def main():
    base_directory = '/data/csc4801/KedzioraLab/ModelNet40/'
    
    try:
        for category in os.listdir(base_directory):
            category_path = os.path.join(base_directory, category)

            if os.path.isdir(category_path):
                for train_test in os.listdir(category_path):
                    current_dir = os.path.join(category_path, train_test)
                    off_dir = os.path.join(current_dir, 'off')
                    stl_dir = os.path.join(current_dir, 'stl')

                    if not os.path.isdir(off_dir):
                        os.mkdir(off_dir)
                    if not os.path.isdir(stl_dir):
                        os.mkdir(stl_dir)

                    off_files_pattern = os.path.join(category_path, train_test, '*.off')

                    for off_file in glob.glob(off_files_pattern):
                        print(f'Processing {off_file}')
                        filename = off_file.split('.')[0].split('/')[-1]
                        
                        new_off_file_path = os.path.join(off_dir, f'{filename}.off')
                        stl_file_path = os.path.join(stl_dir, f'{filename}.stl')

                        mesh = trimesh.load(off_file)
                        mesh.export(stl_file_path)

                        os.rename(off_file, new_off_file_path)
    except (FileNotFoundError, PermissionError) as e: 
        print(f"Error converting files: {e}") 





if __name__ == '__main__':
    main()