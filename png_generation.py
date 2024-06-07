# %%
import trimesh as tri
import numpy as np
import os
import glob
import scipy.io
import multiprocessing as mp
import warnings
from PIL import Image
import random

densifyN = 100000


# %%
def parseObj(fname):
    vertex, edge, face = [], [], []
    # parse vertices
    with open(fname) as file:
        for line in file:
            token = line.strip().split(" ")
            if token[0] == "v":
                vertex.append([float(token[1]), float(token[2]), float(token[3])])
    vertex = np.array(vertex, dtype=np.float64)
    # parse faces
    with open(fname) as file:
        for line in file:
            token = line.strip().split()
            if len(token) > 0 and token[0] == "f":
                idx1 = int(token[1].split("/")[0]) - 1
                idx2 = int(token[2].split("/")[0]) - 1
                idx3 = int(token[3].split("/")[0]) - 1
                # check if good triangle
                M = vertex[[idx1, idx2, idx3]]
                if np.linalg.matrix_rank(M) == 3:
                    face.append([idx1, idx2, idx3])
    face = np.array(face, dtype=np.float64)
    # parse edges
    for f in face:
        edge.append([min(f[0], f[1]), max(f[0], f[1])])
        edge.append([min(f[0], f[2]), max(f[0], f[2])])
        edge.append([min(f[1], f[2]), max(f[1], f[2])])
    edge = [list(s) for s in set([tuple(e) for e in edge])]
    edge = np.array(edge, dtype=np.float64)
    return vertex, edge, face


def removeWeirdDuplicate(F):
    F.sort(axis=1)
    F = [f for f in F]
    F.sort(key=lambda x: [x[0], x[1], x[2]])
    N = len(F)
    for i in range(N - 1, -1, -1):
        if F[i][0] == F[i - 1][0] and F[i][1] == F[i - 1][1] and F[i][2] == F[i - 1][2]:
            F.pop(i)
    return F


def edgeLength(V, E, i):
    return np.linalg.norm(V[int(E[i][0])] - V[int(E[i][1])])


def pushEtoFandFtoE(EtoF, FtoE, E, f, v1, v2):
    if v1 > v2: v1, v2 = v2, v1
    e = np.where(np.all(E == [v1, v2], axis=1))[0][0]
    EtoF[e].append(f)
    FtoE[f].append(e)


def pushAndSort(Elist, V, E, ei):
    l = edgeLength(V, E, ei)
    if edgeLength(V, E, ei) > edgeLength(V, E, Elist[0]):
        Elist.insert(0, ei)
    else:
        left, right = 0, len(Elist)
        while left + 1 < right:
            mid = (left + right) // 2
            if edgeLength(V, E, ei) > edgeLength(V, E, Elist[mid]):
                right = mid
            else:
                left = mid
        Elist.insert(left + 1, ei)


def densify(V, E, F, EtoF, FtoE, Elist):
    vi_new = len(V)
    ei_new = len(E)
    # longest edge
    eL = Elist.pop(0)
    # create new vertex
    vi1, vi2 = int(E[eL][0]), int(E[eL][1])
    v_new = (V[vi1] + V[vi2]) / 2
    V.append(v_new)
    # create new edges
    e_new1 = np.array([vi1, vi_new], dtype=np.float64)
    e_new2 = np.array([vi2, vi_new], dtype=np.float64)
    E.append(e_new1)
    E.append(e_new2)
    EtoF.append([])
    EtoF.append([])
    # push Elist and sort
    pushAndSort(Elist, V, E, ei_new)
    pushAndSort(Elist, V, E, ei_new + 1)
    # create new triangles
    for f in EtoF[eL]:
        fi_new = len(F)
        vio = [i for i in F[f] if i not in E[eL]][0]
        f_new1 = np.array([(vi_new if i == vi2 else i) for i in F[f]], dtype=np.float64)
        f_new2 = np.array([(vi_new if i == vi1 else i) for i in F[f]], dtype=np.float64)
        F.append(f_new1)
        F.append(f_new2)
        e_new = np.array([vio, vi_new], dtype=np.float64)
        E.append(e_new)
        EtoF.append([])
        e_out1 = [e for e in FtoE[f] if min(E[e][0], E[e][1]) == min(vi1, vio) and
                  max(E[e][0], E[e][1]) == max(vi1, vio)][0]
        e_out2 = [e for e in FtoE[f] if min(E[e][0], E[e][1]) == min(vi2, vio) and
                  max(E[e][0], E[e][1]) == max(vi2, vio)][0]
        # update EtoF and FtoE
        EtoF[e_out1] = [(fi_new if fi == f else fi) for fi in EtoF[e_out1]]
        EtoF[e_out2] = [(fi_new + 1 if fi == f else fi) for fi in EtoF[e_out2]]
        EtoF[ei_new].append(fi_new)
        EtoF[ei_new + 1].append(fi_new + 1)
        EtoF[-1] = [fi_new, fi_new + 1]
        FtoE.append([(e_out1 if i == e_out1 else ei_new if i == eL else len(EtoF) - 1) for i in FtoE[f]])
        FtoE.append([(e_out2 if i == e_out2 else ei_new + 1 if i == eL else len(EtoF) - 1) for i in FtoE[f]])
        FtoE[f] = []
        pushAndSort(Elist, V, E, len(EtoF) - 1)
    # # # delete old edge
    E[eL] = np.ones_like(E[eL]) * np.nan
    EtoF[eL] = []
    # delete old triangles
    for f in EtoF[eL]:
        F[f] = np.ones_like(F[f]) * np.nan


# %%
# conda create -n image_conversion python=3.9
# conda activate image_conversion
# pip install trimesh rtree pyglet==1.5.28 embree pyembree PyOpenGL pyrender pillow imageio matplotlib embreex pyvirtualdisplay
# conda install -c anaconda libglu
# conda install np_conda_kernels
# conda install ipykernel
# python -m ipykernel install --user --name=image_conversion

# %%
def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = np.cos(yaw / 2.0)
    c2 = np.cos(pitch / 2.0)
    c3 = np.cos(roll / 2.0)
    s1 = np.sin(yaw / 2.0)
    s2 = np.sin(pitch / 2.0)
    s3 = np.sin(roll / 2.0)
    qa = c1 * c2 * c3 + s1 * s2 * s3
    qb = c1 * c2 * s3 - s1 * s2 * c3
    qc = c1 * s2 * c3 + s1 * c2 * s3
    qd = s1 * c2 * c3 - c1 * s2 * s3
    return [qa, qb, qc, qd]


def quaternionProduct(q1, q2):
    qa = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    qb = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    qc = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    qd = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    return [qa, qb, qc, qd]


def rotMatrixToQuaternion(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    r = np.sqrt(1 + t)
    qa = 0.5 * r
    qb = np.sign(R[2, 1] - R[1, 2]) * np.abs(0.5 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]))
    qc = np.sign(R[0, 2] - R[2, 0]) * np.abs(0.5 * np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2]))
    qd = np.sign(R[1, 0] - R[0, 1]) * np.abs(0.5 * np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2]))
    return [qa, qb, qc, qd]


def quaternionToRotMatrix(q):
    R = np.array(
        [[1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[0] * q[2] + q[1] * q[3]), 0],
         [2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] - q[0] * q[1]), 0],
         [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2), 0],
         [0, 0, 0, 1]], dtype=np.float64)
    return R


def randomRotation():
    return [random.uniform(-1, 1) for _ in range(4)]
    # pos = np.inf
    # while np.linalg.norm(pos) > 1:
    #     pos = np.random.rand(3) * 2 - 1
    # pos /= np.linalg.norm(pos)
    # phi = np.arcsin(pos[2])
    # theta = np.arccos(pos[0] / np.cos(phi))
    # if pos[1] < 0: theta = 2 * np.pi - theta
    # elev = np.rad2deg(phi)
    # azim = np.rad2deg(theta)
    # rho = 1
    # theta = np.random.rand() * 360
    # return [rho, azim, elev, theta]


def camPosToQuaternion(camPos):
    [cx, cy, cz] = camPos
    q1 = [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2]
    camDist = np.linalg.norm([cx, cy, cz])
    cx, cy, cz = cx / camDist, cy / camDist, cz / camDist
    t = np.linalg.norm([cx, cy])
    tx, ty = cx / t, cy / t
    yaw = np.arccos(ty)
    yaw = 2 * np.pi - np.arccos(ty) if tx > 0 else yaw
    pitch = 0
    roll = np.arccos(np.clip(tx * cx + ty * cy, -1, 1))
    roll = -roll if cz < 0 else roll
    q2 = quaternionFromYawPitchRoll(yaw, pitch, roll)
    q3 = quaternionProduct(q2, q1)
    return q3


# %%
def densify_shape(shape_file):
    V, E, F = parseObj(shape_file)
    F = removeWeirdDuplicate(F)
    Vorig, Eorig, Forig = V.copy(), E.copy(), F.copy()

    # sort by length (maintain a priority queue)
    Elist = list(range(len(E)))
    Elist.sort(key=lambda i: edgeLength(V, E, i), reverse=True)

    # create edge-to-triangle and triangle-to-edge lists
    EtoF = [[] for j in range(len(E))]
    FtoE = [[] for j in range(len(F))]

    for f in range(len(F)):
        v = F[f]
        pushEtoFandFtoE(EtoF, FtoE, E, f, v[0], v[1])
        pushEtoFandFtoE(EtoF, FtoE, E, f, v[0], v[2])
        pushEtoFandFtoE(EtoF, FtoE, E, f, v[1], v[2])
    V, E, F = list(V), list(E), list(F)

    # repeat densification
    for z in range(densifyN):
        densify(V, E, F, EtoF, FtoE, Elist)

    densifyV = np.array(V[-densifyN:], dtype=np.float64)

    return {"V": Vorig, "E": Eorig, "F": Forig, "Vd": densifyV}


def is_valid_point(mesh, rotation_matrix):
    try:
        mesh_copy = mesh.copy()
        mesh_copy.apply_transform(rotation_matrix)

        # Check if the mesh is empty after applying the rotation
        if len(mesh_copy.vertices) == 0:
            return False

        # Create a scene with the rotated mesh
        scene = mesh_copy.scene()
        scene.camera.resolution = [256, 256]
        scene.camera.fov = 60 * (scene.camera.resolution / scene.camera.resolution.max())

        # Get camera rays
        origins, vectors, pixels = scene.camera_rays()

        # Check if origins, vectors, or pixels are empty arrays
        if len(origins) == 0 or len(vectors) == 0 or len(pixels) == 0:
            return False

        # Perform ray-mesh intersection
        locations, index_ray, index_tri = mesh_copy.ray.intersects_location(
            origins, vectors, multiple_hits=False)

        # Check if any intersections are found
        if len(locations) == 0:
            return False

        return True

    except Exception as e:
        print(f"Error in is_valid_point: {e}")
        return False


# %%
def generate_color_image(mesh, rotation_matrix):
    a = np.zeros((256, 256), dtype=np.float64)
    try:
        mesh.apply_transform(rotation_matrix)
        scene = mesh.scene()
    except Exception as e:
        print(f"Error mesh generation: {e}")
        return a

    try:
        scene.camera.resolution = [256, 256]
        scene.camera.fov = 60 * (scene.camera.resolution / scene.camera.resolution.max())
    except Exception as e:
        print(f"Error camera setup: {e}")
        return a

    try:
        origins, vectors, pixels = scene.camera_rays()
    except Exception as e:
        print(f"Error generating camera rays: {e}")
        return a

    try:
        points, index_ray, index_tri = mesh.ray.intersects_location(
            origins, vectors, multiple_hits=False)
    except Exception as e:
        print(f"Error generating intersections: {e}")
        return a

    try:
        points_np = np.asarray(points, dtype=np.float64)
        origins_np = np.asarray(origins[0], dtype=np.float64)
        vectors_np = np.asarray(vectors[index_ray], dtype=np.float64)

        vectors_np = vectors_np / np.linalg.norm(vectors_np, axis=1, keepdims=True)

        # print("Points Length: ", len(points_np))
        # print("Origins Length: ", len(origins_np))
        # print("Vectors Length: ", len(vectors_np))
        # print("vec, len", vectors_np, len(vectors_np))

        with np.errstate(over='ignore'):
            depth = tri.util.diagonal_dot(points_np - origins_np, vectors_np)
        # depth = tri.util.diagonal_dot(points - origins[0], vectors[index_ray])
        pixel_ray = pixels[index_ray]
    except Exception as e:
        print(f"Error generating base depth ONE: {e}")
        return a

    try:
        # print(f"Depth min: {depth.min()}")
        # print(f"Depth max: {depth.max()}")
        # print(f"Depth ptp: {depth.ptp()}")

        # print("Depth: ", depth)

        depth_float = (depth - depth.min()) / depth.ptp()
    except Exception as e:
        print(f"Error generating depth float ONE: {e}")
        return a

    try:
        depth_scaled = 0.8 * (depth_float - depth_float.min()) / depth_float.ptp() + 0.2
    except Exception as e:
        print(f"Error generating depth scaled: {e}")
        return a

    try:
        depth_int_scaled = (depth_scaled * 255).round().astype(np.uint8)
        a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int_scaled
    except Exception as e:
        print(f"Error applying depth scaled: {e}")
        return a

    # try:
    #     downsampling_factor = scene.camera.resolution[0] // 64
    #
    #     blocks = a.reshape(64, downsampling_factor, 64, downsampling_factor)
    #
    #     a = blocks.mean(axis=(1, 3))
    # except Exception as e:
    #     print(f"Error reshaping: {e}")
    #     return a

    try:
        tmp = a

        tmp1 = []
        for a_row in tmp:
            r = []
            for b in a_row:
                r.append([b, b, b])
            tmp1.append(r)
    except Exception as e:
        print(f"Error reformatting array: {e}")
        return a

    return np.array(tmp1, dtype=int)


# %%
def generate_stuff(stl_file, img_dir):
    print(f'Processing FILE: {stl_file}')
    filename = stl_file.split('.')[0].split('/')[-1]

    # class_inputRGB
    img_file_path = os.path.join(img_dir, filename)

    try:
        # print(stl_file, "loading_mesh")
        mesh = tri.load_mesh(stl_file)
    # print(stl_file, "load_mesh done")
    except Exception as e:
        print(f"Error loading - {stl_file}: {e}")

    err_index = 0
    try:
        for i in range(5):
            err_index = i
            point = randomRotation()

            while not is_valid_point(mesh, quaternionToRotMatrix(point)):
                point = randomRotation()

            image = generate_color_image(mesh.copy(), quaternionToRotMatrix(point))
            im = Image.fromarray(np.uint8(image))
            im.save(img_file_path + "_" + str(i) + ".png")

    except Exception as e:
        print(f"Error generating image({err_index}) - {stl_file}: {e}")


# %%
def main():
    print("Starting conversion")

    base_directory = '/data/csc4801/KedzioraLab/ModelNet12/'
    save_directory = '/data/csc4801/KedzioraLab/TrainingData/Images/'

    for category_name in os.listdir(base_directory):
        category_path = os.path.join(base_directory, category_name)
        training_path = os.path.join(save_directory, category_name)
        if not os.path.exists(training_path):
            os.makedirs(training_path)

        if os.path.isdir(category_path):
            for train_test in os.listdir(category_path):
                img_dir = os.path.join(training_path, category_name)
                current_dir = os.path.join(category_path, train_test)
                stl_dir = os.path.join(current_dir, 'stl')

                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)

                stl_files_pattern = os.path.join(stl_dir, '*.stl')

                args = []
                for stl_file in glob.glob(stl_files_pattern):
                    args.append(
                        [stl_file, img_dir])

                print(f'Processing {category_path}')
                print(f'Number of files: {len(args)}')
                # print(f'Number of processes: {mp.cpu_count()}') # ALL CPUS ON THE NODE
                print(f'Number of processes: 8')  # 8 PROCESSES
                with mp.Pool(processes=8) as p:  # only 8 processes because thats how many we requested
                    p.starmap(generate_stuff, args)
                    p.close()
                    p.join()

                print(f'Finished {category_path}')


# %%
print("Starting conversion 1")
# warnings.filterwarnings("ignore")
main()

# %%


# %%
