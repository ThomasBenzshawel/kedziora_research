import shutil
import os
import glob

def delete_stl_projects():
    try:
        invalid_projects = set()
        json_folders = set()

        print('Starting deleting process...')

        with open('invalid_format_thingiverse.txt', 'r') as f:
            for line in f:
                stl_dir = line.split('/')
                stl_dir.pop()
                
                json_dir = stl_dir[0:6]
                json_dir.append('json')

                stl_dir = '/'.join(stl_dir)
                json_dir = '/'.join(json_dir)
                invalid_projects.add(stl_dir)
                json_folders.add(json_dir)
            f.close()

        print('Successfully loaded directories!')

        for project in invalid_projects:
            if os.path.isdir(project):
                shutil.rmtree(project)

        print('Successfully deleted stl projects!')

        for json in json_folders:
            files = glob.glob(os.path.join(json, '*'))
            for f in files:
                os.remove(f)

        print('Successfully deleted json files!\nDeletion fisished successfully!')

                        
    except (FileNotFoundError, PermissionError) as e: 
        print(f'Error accessing files: {e}') 
    except KeyboardInterrupt as e:
        print('---Exited successfully---')


if __name__ == '__main__':
    delete_stl_projects()