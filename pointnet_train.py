import argparse
#from pointnet import PointNetCls
from pointnet2 import *
from pointnet_datasets import pc_normalize, collate_fn, ColumbiaGraspDataset
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
from pyntcloud import PyntCloud
import numpy as np
from sklearn.neighbors import KDTree

from utils.config import config
from utils import hand
from utils.database import *



def train(args):
    from torch.utils.data.sampler import SubsetRandomSampler

    import warnings
    warnings.simplefilter("ignore")

    dataset = ColumbiaGraspDataset()
    batch_size = args.batch_size
    validation_split = .1
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler, collate_fn=collate_fn, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, collate_fn=collate_fn, num_workers=1)

    params = config(section='data')

    classifier = PointNet2Seg(num_classes=20)
    optimizer = optim.Adam(classifier.parameters())
    #criterion = torch.nn.MultiLabelSoftMarginLoss()
    criterion = torch.nn.MSELoss()

    print("Start training...")
    cudnn.benchmark = True
    classifier.cuda()

    try:
        for epoch in range(args.num_epochs):
            print("--------Epoch {}--------".format(epoch))

            # train one epoch
            classifier.train()
            total_train_loss = 0
            correct_examples = 0


            for i, (pointclouds, labels) in enumerate(train_loader):

                pointclouds = pointclouds.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()
                preds = classifier(pointclouds).permute(0, 2, 1)

                loss = criterion(preds, labels)

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                #TODO last layer of classifier is linear or sigmoid???

                print("Train loss: {:.4f}".format(total_train_loss / (i+1)))

            print("Epoch Train loss: {:.4f}".format(total_train_loss / len(train_loader)))

            classifier.eval()
            total_valid_loss = 0

            for i, (pointclouds, labels) in enumerate(valid_loader):

                pointclouds = pointclouds.cuda()
                labels = labels.cuda()

                preds = classifier(pointclouds).permute(0, 2, 1)

                loss = criterion(preds, labels)

                total_valid_loss += loss.item()

            print("Valid loss: {:.4f}".format(total_valid_loss / len(valid_loader)))
    except:
        #torch.save(classifier, 'trained_model.pth')
        pass
            
    torch.save(classifier, 'trained_model.pth')
            
        

    
#TODO save model

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pointnet Trainer')
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=10)
    parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
    args = parser.parse_args()
    train(args)
    