import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Modification / additions
from stl import mesh

def get_point_clouds(stls):
    """ Get a numpy array of the pointcloud for each of the given stl files """
    pts = []
    pths = []
    for stl in stls:
        try:
            cur = mesh.Mesh.from_file(stl)
            # cur.vectors is [face,vertex,x/y/z]
            cur = cur.vectors.reshape(-1,cur.vectors.shape[-1])
            np.random.shuffle(cur)
            if cur.shape[0] < 2048:
                continue
            cur = cur[:2048,:]
            pts.append(cur)
            pths.append(stl)
        except Exception as e:
            print("FAILED:", stl)
            # print(e)
    return pts, pths
    

def compile_partition(partition,wipe):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    if (not wipe) and os.path.exists(os.path.join(DATA_DIR,'modelnet40_ply_hdf5_2048','ply_data_%s.h5'%partition)):
        return
    with h5py.File(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s.h5'%partition), 'w') as hf:
        with open(os.path.join(DATA_DIR,'shape_names.txt')) as file:
            names = file.readlines()

        pts = []
        pths = []
        labels = []

        for i, name in enumerate(names):
            name = name.strip()
            stls = glob.glob(os.path.join(DATA_DIR,partition,name,'**','*.stl'), recursive=True)
            res = get_point_clouds(stls)
            pts += res[0]
            pths += res[1]
            labels += [[i]] * len(res[0])

        hf.create_dataset('paths',data=pths)
        hf.create_dataset('data',data=pts)
        hf.create_dataset('label',data=labels)

def compile_train_test(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    with h5py.File(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', f"ply_data_{partition}_train.h5"), 'w') as hf_train:
        with h5py.File(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', f"ply_data_{partition}_test.h5"), 'w') as hf_test:
            with open(os.path.join(DATA_DIR,'shape_names.txt')) as file:
                names = file.readlines()

            train_pts = []
            train_pths = []
            train_labels = []
            test_pts = []
            test_pths = []
            test_labels = []

            split=0.8

            for i, name in enumerate(names):
                name = name.strip()
                stls = glob.glob(os.path.join(DATA_DIR,partition,name,'**','*.stl'), recursive=True)
                res = get_point_clouds(stls)
                cnt = len(res[0])
                idx = int(cnt*split)

                train_pts += res[0][:idx]
                train_pths += res[1][:idx]
                train_labels += [[i]] * idx

                test_pts += res[0][idx:]
                test_pths += res[1][idx:]
                test_labels += [[i]] * (cnt-idx)

            hf_train.create_dataset('paths',data=train_pths)
            hf_train.create_dataset('data',data=train_pts)
            hf_train.create_dataset('label',data=train_labels)

            hf_test.create_dataset('paths',data=test_pths)
            hf_test.create_dataset('data',data=test_pts)
            hf_test.create_dataset('label',data=test_labels)


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def load_data(partition,wipe):
    download()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        # print(f"h5_name: {h5_name}")
        f = h5py.File(h5_name,'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def load_data_with_paths(partition,wipe):
    download()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    all_data = []
    all_label = []
    all_paths = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        # print(f"h5_name: {h5_name}")
        f = h5py.File(h5_name,'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        paths = f['paths'][:].astype('object')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_paths.append(paths)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label, all_paths


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', wipe=False):
        self.data, self.label = load_data(partition,wipe)
        self.num_points = num_points
        self.partition = partition        
        print(self.data.shape)
        print(self.label.shape)

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if 'train' in self.partition:
            # pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40_Paths(ModelNet40):
    def __init__(self, num_points, partition='thingiverse', wipe=False):
        compile_partition(partition,wipe)
        self.data, self.label, self.paths = load_data_with_paths(partition,wipe)
        self.paths = self.paths[0]
        self.num_points = num_points
        self.partition = partition        
        print(self.data.shape)
        print(self.label.shape)
        print(len(self.paths))

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        up = np.max(pointcloud)
        dwn = np.min(pointcloud)
        pointcloud = (pointcloud-dwn) / (up-dwn).astype(float) * 2. - 1.
        label = self.label[item]
        path = self.paths[item].split(b'/')[-1]
        return pointcloud, label, path


if __name__ == '__main__':
    print("Train size")
    train = ModelNet40(1024)
    print("Test size")
    test = ModelNet40(1024, 'test')
    print("Thingiverse size")
    thing = ModelNet40_Paths(1024, 'thingiverse')
    print("Generated size")
    generated = ModelNet40_Paths(1024, 'generated',wipe=True)
    print("ModelNet12 size")
    mn12 = ModelNet40_Paths(1024, 'ModelNet12',wipe=True)
    print("ModelNet12 train/test size")
    compile_train_test('ModelNet12')
    mn12_train = ModelNet40_Paths(1024, 'ModelNet12_train')
    mn12_test = ModelNet40_Paths(1024, 'ModelNet12_test')
    # for data, label in train:
    #     print(data.shape)
    #     print(label.shape)

    from torch.utils.data import DataLoader
    print("TRAIN")
    train_loader = DataLoader(ModelNet40(partition='train', num_points=1024), num_workers=4,
                              batch_size=32, shuffle=True, drop_last=True)
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | label shape: {label.shape}")
    print("THINGIVERSE")
    train_loader = DataLoader(ModelNet40_Paths(partition='thingiverse', num_points=1024), num_workers=4,
                              batch_size=32, shuffle=True, drop_last=True)
    for batch_idx, (data, label, path) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | label shape: {label.shape} | path len: {len(path)}")

    train_set = ModelNet40(partition='train', num_points=1024)
    test_set = ModelNet40(partition='test', num_points=1024)
    print(f"train_set size {train_set.__len__()}")
    print(f"test_set size {test_set.__len__()}")
    print(f"thingiverse size {thing.__len__()}")
    print(f"generated size {generated.__len__()}")
    print(f"mn12 size {mn12.__len__()}")
