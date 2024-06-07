"""
The shrink wrap needs an initial mesh to start from.
From prior work, I've found it beneficial to use the alpha shapes algorithm to
get a large estimation, then shrink wrap from there.
This python script makes the initial mesh.
"""
import open3d as o3d
import os
import argparse
import scipy.io
import numpy as np

parser = argparse.ArgumentParser(description='Make the initial mesh')
parser.add_argument('--pc', type=str, action="store", dest='pc')
parser.add_argument('--output-dir', type=str, action="store", dest='dir')
parser.add_argument('--perspective', type=int, action="store", dest='per')
args = parser.parse_args()

# Get the point cloud
if args.pc.endswith(".ply"):
    pcd = o3d.io.read_point_cloud(args.pc)
else:
    mat_import = scipy.io.loadmat(args.pc)
    mat = mat_import["pointcloud"][:,0]
    pts = np.array(mat)[args.per]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd = pcd.select_by_index(ind)

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=50, max_nn=500))
pcd.orient_normals_consistent_tangent_plane(100)
o3d.io.write_point_cloud(f"{args.dir}/pc.ply", pcd, write_ascii=True)

## Ball Pivot
# radii = [0.02, 0.04, 0.08]
# rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     pcd, o3d.utility.DoubleVector(radii))

## Poisson
# rec_mesh, densities =
# o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12,
#         linear_fit=False)
# rec_mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(rec_mesh)
# rec_mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([rec_mesh])

rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.5)
rec_mesh.compute_vertex_normals()

# Write to stl
# print("Write STL Mesh:",o3d.io.write_triangle_mesh(args.dir+"/init.stl", rec_mesh))
print("Write OBJ Mesh:",o3d.io.write_triangle_mesh(args.dir+"/init.obj", rec_mesh))

