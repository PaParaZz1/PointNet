#-*- coding=utf-8-*-

import os.path
import torch.utils.data
import torch
import numpy as np
import random
import math

class PartDataset(torch.utils.data.Dataset):
    def __init__(self, root, catfile='cat.txt', npoints=4096, train=True):
        self.npoints = npoints
        self.root = root
        self.category = {}
        self.classification = False

        #生成所有类别及其分类编号
        with open(os.path.join(root, 'data/BDCI/training', catfile)) as f_catfile:
            lines = f_catfile.readlines()
            self.category = dict(zip(lines, range(len(lines))))
        #建立路径
        sample_train_point = os.path.join(self.root, 'data/BDCI/sample_train/pts')
        sample_train_seg = os.path.join(self.root, 'data/BDCI/sample_train/category')

        sample_dev_point = os.path.join(self.root, 'data/BDCI/sample_dev/pts')
        sample_dev_seg = os.path.join(self.root, 'data/BDCI/sample_dev/category')

        #构建数据集
        if train:
            #数据集为train所用的sample
            point_files = sorted(os.listdir(sample_train_point))
            seg_files = sorted(os.listdir(sample_train_seg))
            self.datapath = [(os.path.join(sample_train_point, point_files[i]), os.path.join(sample_train_seg, seg_files[i]))
                             for i in range(len(point_files))]
        else:
            #数据集dev
            point_files = sorted(os.listdir(sample_dev_point))
            seg_files = sorted(os.listdir(sample_dev_seg))
            self.datapath = [(os.path.join(sample_dev_point, point_files[i]), os.path.join(sample_dev_seg, seg_files[i]))
                             for i in range(len(point_files))]

        # 分割的类别个数
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath) // 50):
                ls = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                if ls > self.num_seg_classes:
                    self.num_seg_classes = ls
        self.num_seg_classes = 8

    def __getitem__(self, index):
        fn = self.datapath[index]
        #载入sample文件内容
        point_set = np.loadtxt(fn[0], delimiter=',').astype(np.float32)
        seg_set = np.loadtxt(fn[1]).astype(np.int64)
        #转为torch的Tensor类型
        point_set = torch.from_numpy(point_set)
        seg_set = torch.from_numpy(seg_set)
        #返回数据对
        return point_set, seg_set

    def __len__(self):
        return len(self.datapath)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, catfile='cat.txt', npoints=4096, train=True, classification=False):
        self.npoints = npoints
        self.root = root
        self.category = {}
        self.classification = classification

        #生成所有类别及其分类编号
        with open(os.path.join(root, 'data/BDCI/training', catfile)) as f_catfile:
            lines = f_catfile.readlines()
            self.category = dict(zip(lines, range(len(lines))))
        #建立路径
        test_point = os.path.join(self.root, 'data/BDCI/test_set/pts')
        test_seg = os.path.join(self.root, 'data/BDCI/test_set/category')

        #构建数据集
        if train:
            #数据集为0.8
            point_files = sorted(os.listdir(test_point)[:int(0.8*len(test_point))])
            seg_files = sorted(os.listdir(test_seg)[:int(0.8*len(test_seg))])
            self.datapath = [(os.path.join(test_point, point_files[i]), os.path.join(test_seg, seg_files[i]))
                             for i in range(len(point_files))]
        else:
            #数据集为all
            point_files = sorted(os.listdir(test_point))
            seg_files = sorted(os.listdir(test_seg))
            self.datapath = [(os.path.join(test_point, point_files[i]), os.path.join(test_seg, seg_files[i]))
                             for i in range(len(point_files))]

        # 分割的类别个数
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath) // 50):
                ls = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                if ls > self.num_seg_classes:
                    self.num_seg_classes = ls

    def __getitem__(self, index):
        fn = self.datapath[index]
        #载入sample文件内容
        point_set = list(np.loadtxt(fn[0], delimiter=',').astype(np.float32).tolist())
        #归一化
        means = list(np.mean(point_set, axis=0).tolist())
        fangcha = list(np.var(point_set, axis=0).tolist())
        fangcha = list(map(math.sqrt, fangcha))
        for i in range(len(point_set)):
            point_set[i][0] = (point_set[i][0] - means[0]) / fangcha[0]
            point_set[i][1] = (point_set[i][1] - means[1]) / fangcha[1]
            point_set[i][2] = (point_set[i][2] - means[2]) / fangcha[2]
        point_set = np.array(point_set)
        point_set = torch.from_numpy(point_set)
        #返回坐标，文件名
        return point_set, os.path.basename(fn[0])

    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    print('test')
    d = PartDataset(root='/mnt/lustre/niuyazhe')
    print(len(d))
    ps, seg = d[0]
    print(ps.size(), ps.type(), seg.size(), seg.type())
    d = TestDataset(root='/mnt/lustre/niuyazhe')
    print(len(d))
    ps, name = d[0]
    print(name)
