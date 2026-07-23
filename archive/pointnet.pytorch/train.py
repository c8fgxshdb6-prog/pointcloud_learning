import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决OpenMP冲突

import sys
import argparse
import h5py
import numpy as np

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 现在可以正常导入
from pointnet.model import PointNetCls, feature_transform_regularizer

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import random

# HDF5 格式的 ModelNet40 数据集类
class ModelNetH5Dataset(torch.utils.data.Dataset):
    def __init__(self, root, npoints=2500, split='train', data_augmentation=True):
        self.npoints = npoints
        self.split = split
        self.data_augmentation = data_augmentation
        
        # 根据 split 确定要加载的文件
        if split == 'train' or split == 'trainval':
            self.files = [os.path.join(root, 'ply_data_train0.h5'),
                         os.path.join(root, 'ply_data_train1.h5'),
                         os.path.join(root, 'ply_data_train2.h5'),
                         os.path.join(root, 'ply_data_train3.h5'),
                         os.path.join(root, 'ply_data_train4.h5')]
        else:  # test
            self.files = [os.path.join(root, 'ply_data_test0.h5'),
                         os.path.join(root, 'ply_data_test1.h5')]
        
        # 加载所有数据
        self.data = []
        self.labels = []
        for file_path in self.files:
            if os.path.exists(file_path):
                with h5py.File(file_path, 'r') as f:
                    self.data.append(f['data'][:])
                    self.labels.append(f['label'][:])
            else:
                print(f"警告: 文件 {file_path} 不存在")
        
        if len(self.data) == 0:
            raise FileNotFoundError(f"在 {root} 中找不到任何 HDF5 文件")
        
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0).squeeze()
        
        # 加载类别名称
        shape_names_path = os.path.join(root, 'shape_names.txt')
        if os.path.exists(shape_names_path):
            with open(shape_names_path, 'r') as f:
                self.classes = [line.strip() for line in f]
        else:
            # 如果没有 shape_names.txt，从数据中推断类别
            self.classes = [str(i) for i in range(int(np.max(self.labels)) + 1)]
        
        print(f"Loaded {len(self.data)} samples for {split}")

    def __getitem__(self, index):
        point_set = self.data[index].astype(np.float32)
        label = self.labels[index].astype(np.int64)
        
        # 随机采样到固定点数
        if point_set.shape[0] < self.npoints:
            # 如果点数不足，重复采样
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        else:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        
        point_set = point_set[choice, :]
        
        # 数据归一化
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # 中心化
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # 尺度归一化

        if self.data_augmentation:
            # 随机旋转
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # 在x-z平面旋转
            # 添加随机抖动
            point_set += np.random.normal(0, 0.02, size=point_set.shape)

        return point_set, label

    def __len__(self):
        return self.data.shape[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--num_points', type=int, default=1024, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--nepoch', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

    opt = parser.parse_args()
    print(opt)

    blue = lambda x: '\033[94m' + x + '\033[0m'

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset_type == 'modelnet40':
        # 使用我们新的 HDF5 数据集类
        dataset = ModelNetH5Dataset(
            root=opt.dataset,
            npoints=opt.num_points,
            split='trainval')

        test_dataset = ModelNetH5Dataset(
            root=opt.dataset,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False)
    else:
        exit('wrong dataset type')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers))

    print(f"训练样本: {len(dataset)}, 测试样本: {len(test_dataset)}")
    num_classes = len(dataset.classes)
    print('类别数量:', num_classes)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)
    classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    num_batch = len(dataset) / opt.batchSize

    for epoch in range(opt.nepoch):
        total_loss = 0
        for i, data in enumerate(dataloader, 0):
            points, target = data
            # 移除 target[:, 0] - 我们的标签已经是一维的
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                # 移除 target[:, 0] - 我们的标签已经是一维的
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                pred, _, _ = classifier(points)
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

        # 在每个epoch结束后调用scheduler.step()
        scheduler.step()
        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

    total_correct = 0
    total_testset = 0
    for i,data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        # 移除 target[:, 0] - 我们的标签已经是一维的
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]

    print("最终准确率: {}".format(total_correct / float(total_testset)))

if __name__ == '__main__':
    main()