"""
Convert from obj to STL
"""
import open3d as o3d
import argparse

parser = argparse.ArgumentParser(description='Make the initial mesh')
parser.add_argument('--obj', type=str, action="store", dest='obj')
parser.add_argument('--stl', type=str, action="store", dest='stl')
args = parser.parse_args()

mesh = o3d.io.read_triangle_mesh(args.obj)
mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(mesh)
o3d.io.write_triangle_mesh(args.stl, mesh)
