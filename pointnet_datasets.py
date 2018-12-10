import torch.utils.data as data
#import h5py
import numpy as np
import os
from glob import glob
from pyntcloud import PyntCloud
import numpy as np
from sklearn.neighbors import KDTree
from utils import hand
from utils.config import config
from utils.database import *
import torch


class ModelNetDataset(data.Dataset):
    def __init__(self, train=True):
        if train:
            data_file = 'data/modelnet40_ply_hdf5_2048/train_files.txt'
        else:
            data_file = 'data/modelnet40_ply_hdf5_2048/test_files.txt'
        file_list = [line.rstrip() for line in open(data_file, 'r')]
        
        all_data = np.zeros([0, 2048, 3], np.float32)
        all_label = np.zeros([0, 1], np.int64)
        for filename in file_list:
            f = h5py.File(filename)
            data = f['data'][:]
            label = f['label'][:]

            all_data = np.concatenate([all_data, data], 0)
            all_label = np.concatenate([all_label, label], 0)

        self.pointcloud = all_data
        self.label = all_label

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        return self.pointcloud[index], self.label[index]

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class TensorBodyDataset():
    def __init__(self, data_dir, normalize=True, train=True):
        self.normalize = normalize
        self.pointcloud_files = []
        self.label_files = []
        file_list = os.path.join(data_dir, 'data_list.txt')
        with open(file_list, 'r') as file:
            for line in file:
                if line:
                    pointcloud_file, label_file = line.rstrip().split(' ')
                    self.pointcloud_files.append(os.path.join(data_dir, pointcloud_file))
                    self.label_files.append(os.path.join(data_dir, label_file))
        if train:
            self.idxs = np.arange(len(self.pointcloud_files))[:30000]
        else:
            self.idxs = np.arange(len(self.pointcloud_files))[30000:]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        pointcloud = np.load(self.pointcloud_files[self.idxs[index]]).astype(np.float32)
        label = np.load(self.label_files[self.idxs[index]]).astype(np.int64)

        if self.normalize:
            pointcloud = pc_normalize(pointcloud)

        return pointcloud, label

class SMPLDataset():
    def __init__(self, data_dir, normalize=True, train=True):
        self.normalize = normalize
        self.pointcloud_files = glob(os.path.join(data_dir, 'pointclouds', '*/*.npy'))
        self.label_files = glob(os.path.join(data_dir, 'labels', '*/*.npy'))
        N = len(self.pointcloud_files)  
        indices = np.random.choice(N, N, replace=False)
        part = int(N * 0.8)
        if train:
            self.idxs = indices[:part]
        else:
            self.idxs = indices[part:]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        pointcloud = np.load(self.pointcloud_files[self.idxs[index]]).astype(np.float32)
        label = np.load(self.label_files[self.idxs[index]]).astype(np.int64)

        if self.normalize:
            pointcloud = pc_normalize(pointcloud)

        return pointcloud, label
    
def sample_pt_cld(scale, grasp_rescale, abs_model_path):
    m = PyntCloud.from_file(abs_model_path)
    pt_cld = m.get_sample("mesh_random", n=10000, rgb=False, normals=False).values
    pt_cld *= grasp_rescale
    pt_cld *= scale
    return pt_cld

def get_joint_locations(grasp_joints, grasp_position):
    start = np.array(grasp_position)[np.newaxis, 1:4].T
    quat = np.array(grasp_position)[4:8]

    h = hand.makeHumanHand(start, quat)
    h.updateDofs(np.array(grasp_joints)[1:])
    hand_pts = h.getJointPositions()
    hand_pts = np.concatenate(hand_pts, axis=1).T[1:]
    return hand_pts

def get_contact_probs(preds, pt_cld, hand_pts, contact_dist=10):
    #preds 1*10000*20
    #contact_dist number of mm for joint to be considered contact point
    #returns contact joints (where joint 0 is from finger not hand root) and probs of those contact points
    
    tree = KDTree(pt_cld)              
    dist, ind = tree.query(hand_pts[1:], k=1)       

    joints = np.arange(len(hand_pts[1:]))[:, np.newaxis]

    ind = ind[dist < contact_dist]
    joints = joints[dist < contact_dist]
    contact_pts = pt_cld[ind]

    probs = preds[0, ind, joints].data.cpu().numpy()
    return joints, probs
    
def get_joint_probs(preds, pt_cld, hand_pts):
    return get_contact_probs(preds, pt_cld, hand_pts, contact_dist=np.inf)
    
class ColumbiaGraspDataset():
    def __init__(self, normalize=True):
        self.normalize = normalize
        self.params = config(section='data')
        self.mr = ModelReader()
        self.data = [self.mr.prepare_sample(grasp) for grasp in self.mr.getAll()]
        #self.models = [self.mr.getModelInfo(grasp["scaled_model_id"]) for grasp in self.data]
        self.models = []
        for grasp in self.data:
            try:
                self.models.append(self.mr.getModelInfo(grasp["scaled_model_id"]))
            except Exception as e:
                self.models.append((None, None, None))
                #Some of the scaled_model_id aren't listed in the database
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        grasp = self.data[index]
        
        #scale, grasp_rescale, model_path = self.mr.getModelInfo(grasp["scaled_model_id"])
        scale, grasp_rescale, model_path = self.models[index]
        
        if scale is None:
            return None, None
        
#         m = PyntCloud.from_file(self.params['model_dir'] + model_path)
#         pt_cld = m.get_sample("mesh_random", n=10000, rgb=False, normals=False).values
#         pt_cld *= grasp_rescale
#         pt_cld *= scale
     
        pt_cld = sample_pt_cld(scale, grasp_rescale, self.params['model_dir'] + model_path)

#         start = np.array(grasp['grasp_grasp_position'])[np.newaxis, 1:4].T
#         quat = np.array(grasp['grasp_grasp_position'])[4:8]

#         h = hand.makeHumanHand(start, quat)
#         h.updateDofs(np.array(grasp['grasp_grasp_joints'])[1:])
#         hand_pts = h.getJointPositions()
#         hand_pts = np.concatenate(hand_pts, axis=1).T[1:]
        
        hand_pts = get_joint_locations(grasp['grasp_grasp_joints'], grasp['grasp_grasp_position'])

        tree = KDTree(pt_cld)              
        dist, ind = tree.query(hand_pts[1:], k=1)                

        joints = np.arange(len(hand_pts[1:]))[:, np.newaxis]

        contact_dist = 10 #number of mm for joint to be considered contact point

        ind = ind[dist < contact_dist]
        joints = joints[dist < contact_dist]
        contact_pts = pt_cld[ind]

        label = np.zeros((len(pt_cld), 20))
        
        if len(contact_pts) == 0:
            return None, None

        tau = 20.8936034 #value such that a point 20mm away has value 0.4

        contact_dists, contact_inds = tree.query(contact_pts, k=len(pt_cld))
        #approximating geodesic distance with euclidean distance as described in 
        #Distance Functions and Geodesics on Points Clouds by Memoli et. al.


        for contact_dist, contact_ind, joint in zip(contact_dists, contact_inds, joints):
            label[contact_ind, joint] = np.exp(-np.power(contact_dist, 2) / tau**2)

        #pointcloud = torch.tensor(pc_normalize(pt_cld)).cuda().unsqueeze(0).permute(0, 2, 1)            
        #label = torch.tensor(label, dtype=torch.float32).cuda().unsqueeze(0)
        pt_cld = pc_normalize(pt_cld) if self.normalize else pt_cld
        return pt_cld, label

def collate_fn(data):
    
    pointclouds = []
    labels = []
    for pointcloud, label in data:
        if not pointcloud is None:
            pointclouds.append(torch.tensor(pointcloud).unsqueeze(0).permute(0, 2, 1))
            labels.append(torch.tensor(label, dtype=torch.float32).unsqueeze(0))
    pointclouds = torch.cat(pointclouds)
    labels = torch.cat(labels)
    
    return pointclouds, labels

if __name__ == '__main__':
    #dataset = ModelNetDataset()
    #dataset = TensorBodyDataset('data/seg1024')
    dataset = SMPLDataset('D:\\Data\\CMUPointclouds')
    print(len(dataset))
