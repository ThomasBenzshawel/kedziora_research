# sample to return a mesh

import trimesh as tri

def export_mesh(mesh, path, filetype):
    with open(path, "wb") as f:
        mesh.export(file_obj=f, file_type=filetype)
    
def import_mesh(path):
    return tri.load(path)


def generate(images):
    print("Print from generator")
    return import_mesh("test.stl")