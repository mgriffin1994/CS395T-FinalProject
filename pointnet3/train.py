import argparse
from pointnet import PointNetCls
from pointnet2 import *
from datasets import ModelNetDataset
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
from config import config
from pyntcloud import PyntCloud
from datasets import pc_normalize
import numpy as np


def train(args):
    # init training dataset
    #train_dataset = ModelNetDataset(train=True)
    #train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    #train_examples = len(train_dataset)
    #train_batches = len(train_dataloader)

    #test_dataset = ModelNetDataset(train=False)
    #test_examples = len(test_dataset)
    #test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    #test_batches = len(test_dataloader)

    params = config(section='data')

    classifier = PointNet2ClsSsg(out_size=5) #outsize corrrect TODO???
    optimizer = optim.Adam(classifier.parameters())

    print("Train examples: {}".format(train_examples))
    print("Evaluation examples: {}".format(test_examples))
    print("Start training...")
    cudnn.benchmark = True
    classifier.cuda()
    for epoch in range(num_epochs):
        print("--------Epoch {}--------".format(epoch))

        # train one epoch
        classifier.train()
        total_train_loss = 0
        correct_examples = 0

        mr = ModelReader(args.batch_size)

        batch = mr.getGraspBatch()
        while len(batch) > 0:
            
            pointclouds = []
            labels = []
            
            for grasp in batch:
                scale, model_path = mr.getModelInfo(grasp["scaled_model_id"])
                m = PyntCloud.from_file(params['model_dir'] + model_path)
                pointcloud = m.get_sample("mesh_random", n=10000, rgb=False, normals=False).values
                pointcloud = pc_normalize(pointcloud)
                pointclouds.append(torch.tensor(pointcloud))
                
                contacts = np.array(one_grasp['grasp_contacts'])
                #TODO make this into point cloud label

            pointclouds = torch.cat(pointclouds).cuda() 
            labels = labels.cuda()
            
            
            optimizer.zero_grad()
            preds = classifier(pointclouds)

            loss = F.nll_loss(preds, labels.view(-1))
            preds_choice = preds.max(1)[1]

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            correct_examples += preds_choice.eq(labels.view(-1)).sum().item()
            
            batch = mr.getGraspBatch()

            
        print("Train loss: {:.4f}, train accuracy: {:.2f}%".format(total_train_loss / train_batches, correct_examples / train_examples * 100.0))

        # eval one epoch
#         classifier.eval()
#         correct_examples = 0
#         for batch_idx, data in enumerate(test_dataloader, 0):
#             pointcloud, label = data
#             pointcloud = pointcloud.permute(0, 2, 1)
#             pointcloud, label = pointcloud.cuda(), label.cuda()

#             pred = classifier(pointcloud)
#             pred_choice = pred.max(1)[1]
#             correct = pred_choice.eq(label.view(-1)).sum()
#             correct_examples += correct.item()

#         print("Eval accuracy: {:.2f}%".format(correct_examples / test_examples * 100.0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pointnet Trainer')
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=8)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=10)
    parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
    args = parser.parse_args()
    train(args)
    