import os
import glob
import trimesh
import shutil


def create_dir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)


def delete_bad_projects(base_dir: str, new_dir):
    try:
        create_dir(new_dir)
        for category in os.listdir(base_dir):
            print(category)
            base_category_path = os.path.join(base_dir, category, 'stls')
            del_category_path = os.path.join(new_dir, category)

            create_dir(del_category_path)

            if os.path.isdir(base_category_path):
                for obj in os.listdir(base_category_path):
                    base_obj_path = os.path.join(base_category_path, obj)
                    del_obj_path = os.path.join(del_category_path, obj)

                    copy_dir = True
                    base_stl_paths = glob.glob(os.path.join(base_obj_path, '*.stl'))
                    
                    if len(base_stl_paths) > 1:
                        copy_dir = False
                    else:
                        for stl_file in base_stl_paths:
                            try:
                                mesh = trimesh.load(stl_file)
                                if type(mesh) == trimesh.scene.scene.Scene:
                                    copy_dir = False
                            except (TypeError, UnicodeDecodeError, trimesh.exchange.stl.HeaderError) as e:
                                copy_dir = False

                    if copy_dir:
                        shutil.copytree(base_obj_path, del_obj_path)
                        
    except (FileNotFoundError, PermissionError) as e: 
        print(f"Error accessing files: {e}") 
    except KeyboardInterrupt as e:
        print('---Exited successfully---')


if __name__ == '__main__':
    delete_bad_projects('/data/csc4801/KedzioraLab/thingiverse/', '/data/csc4801/KedzioraLab/fixed_thingiverse/')