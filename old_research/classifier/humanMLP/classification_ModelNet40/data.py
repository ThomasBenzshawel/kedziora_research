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
        except Exception as e:
            print("FAILED:", stl)
    return pts
    

def compile_partition(partition,wipe):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    if (not wipe) and os.path.exists(os.path.join(DATA_DIR,'modelnet40_ply_hdf5_2048','ply_data_%s.h5'%partition)):
        return
    label = 1 if partition == 'generated' else 0
    with h5py.File(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s.h5'%partition), 'w') as hf:
        with open(os.path.join(DATA_DIR,'modelnet40_ply_hdf5_2048','shape_names.txt')) as file:
            names = file.readlines()

        pts = []
        labels = []

        for name in names:
            name = name.strip()
            stls = glob.glob(os.path.join(DATA_DIR,partition,name,'**','*.stl'), recursive=True)
            pts += get_point_clouds(stls)
            labels += [[label]] * len(pts)

        hf.create_dataset('data',data=pts)
        hf.create_dataset('label',data=labels)


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
    if partition in ["ModelNet40","thingiverse", "generated"]:
        compile_partition(partition,wipe)

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
        if self.partition == 'train':
            # pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    print("ModelNet40 size")
    train = ModelNet40(1024, 'ModelNet40')
    for data, label in train:
        print(data.shape)
        print(label.shape)
    # print("Test size")
    # test = ModelNet40(1024, 'test')
    # print("Thingiverse size")
    # thing = ModelNet40(1024, 'thingiverse')
    print("Generated size")
    generated = ModelNet40(1024, 'generated',wipe=True)
    for data, label in generated:
        print(data.shape)
        print(label.shape)

    from torch.utils.data import DataLoader
    # print("TRAIN")
    # train_loader = DataLoader(ModelNet40(partition='train', num_points=1024), num_workers=4,
    #                           batch_size=32, shuffle=True, drop_last=True)
    # for batch_idx, (data, label) in enumerate(train_loader):
    #     print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | ;lable shape: {label.shape}")
    # print("THINGIVERSE")
    # train_loader = DataLoader(ModelNet40(partition='thingiverse', num_points=1024), num_workers=4,
    #                           batch_size=32, shuffle=True, drop_last=True)
    # for batch_idx, (data, label) in enumerate(train_loader):
    #     print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | ;lable shape: {label.shape}")

    # train_set = ModelNet40(partition='train', num_points=1024)
    # test_set = ModelNet40(partition='test', num_points=1024)
    # print(f"train_set size {train_set.__len__()}")
    # print(f"test_set size {test_set.__len__()}")
    # print(f"thingiverse size {thing.__len__()}")
