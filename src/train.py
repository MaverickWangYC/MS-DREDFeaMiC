from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from h5_reader import read_h5
from read_MI import read_MI_batch
from read_CHD import read_CHD_batch
from read_CC import read_CC_batch
from read_tomato import read_tomato
from read_KidneyDisease import read_KidneyDisease
import math
import pandas as pd
import torch
from torch import nn
from torch.utils import data
import shap
import numpy as np
from model import (baselinemlp, RNN, LSTM, MSDREDFeaMiC, pureMamba, Transformer)
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
import torch.nn.functional as F
import random
# from d2l import torch as d2l
from torch.nn import functional as F
import pywt
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
import argparse
from sklearn.preprocessing import MinMaxScaler
# from DBLoss import BalancedCrossEntropyLoss
# device = "cpu"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


loss2 = nn.BCEWithLogitsLoss()


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_K_fold_data_CC(K, i, file_name, mode):
    # 最大K为3
    if mode == 'NP':
        file_dir = '../data/ColonCancer/' + file_name
    if mode == 'EW':
        file_dir = '../data/ColonCancer/' + file_name
    batch_1, labels_1, batch_2, labels_2= read_CC_batch(file_dir)
    batch_1 = batch_1 / np.max(batch_1, axis=1)[:, None]
    batch_2 = batch_2 / np.max(batch_2, axis=1)[:, None]

    unique_chars, integer_labels_1 = np.unique(labels_1, return_inverse=True)
    unique_chars, integer_labels_2 = np.unique(labels_2, return_inverse=True)
    if mode == 'NP':
        wp_ids = np.where(integer_labels_1 == 2)
        integer_labels_1[wp_ids] = 0
        wp_ids = np.where(integer_labels_2 == 2)
        integer_labels_2[wp_ids] = 0
        train_data = batch_1
        train_label = integer_labels_1
        test_data = batch_2
        test_label_true = integer_labels_2
    elif mode == 'EW':
        ids_1 = np.where(integer_labels_1 != 1)
        ids_2 = np.where(integer_labels_2 != 1)
        wp_ids = np.where(integer_labels_1 == 2)
        integer_labels_1[wp_ids] = 1
        wp_ids = np.where(integer_labels_2 == 2)
        integer_labels_2[wp_ids] = 1

        batch_1_1 = batch_1[ids_1]
        print(batch_1_1.shape)
        integer_labels_1_1 = integer_labels_1[ids_1]
        batch_2_1 = batch_2[ids_2]
        print(batch_2_1.shape)
        integer_labels_2_1 = integer_labels_2[ids_2]

        train_data = batch_1_1
        train_label = integer_labels_1_1
        test_data = batch_2_1
        test_label_true = integer_labels_2_1

    train_p = sum(train_label)
    train_n = len(train_label) - train_p

    train_data = torch.tensor(
        train_data, dtype=torch.float32
    ).to(device)

    train_label = torch.tensor(
        train_label
    ).to(device)

    test_data = torch.tensor(
        test_data, dtype=torch.float32
    ).to(device)

    test_label_true = torch.tensor(
        test_label_true
    ).to(device)

    all_len = train_n + train_p
    class_weights = torch.tensor(
        [1.0 - float(train_n / all_len), 1.0 - float(train_p / all_len)])

    print(train_data.shape)
    return train_data, train_label, test_data, test_label_true, class_weights.to(device)

def get_K_fold_data_MI(K, i):
    # 最大K为3
    file_dir = '../data/MI'
    batch_1, labels_1, batch_2, labels_2, batch_3, labels_3 = read_MI_batch(file_dir)
    batch_1 = batch_1 / np.max(batch_1, axis=1)[:, None]
    batch_2 = batch_2 / np.max(batch_2, axis=1)[:, None]
    batch_3 = batch_3 / np.max(batch_3, axis=1)[:, None]

    unique_chars, integer_labels_1 = np.unique(labels_1, return_inverse=True)
    unique_chars, integer_labels_2 = np.unique(labels_2, return_inverse=True)
    unique_chars, integer_labels_3 = np.unique(labels_3, return_inverse=True)


    if i == 0:
        train_data = np.append(batch_1, batch_2, axis=0)
        train_label = np.append(integer_labels_1, integer_labels_2)
        test_data = batch_3
        test_label_true = integer_labels_3


    elif i == 1:
        train_data = np.append(batch_1, batch_3, axis=0)
        train_label = np.append(integer_labels_1, integer_labels_3)
        test_data = batch_2
        test_label_true = integer_labels_2

    elif i == 2:
        train_data = np.append(batch_2, batch_3, axis=0)
        train_label = np.append(integer_labels_2, integer_labels_3)
        test_data = batch_1
        test_label_true = integer_labels_1

    train_p = sum(train_label)
    train_n = len(train_label) - train_p

    train_data = torch.tensor(
        train_data, dtype=torch.float32
    ).to(device)

    train_label = torch.tensor(
        train_label
    ).to(device)

    test_data = torch.tensor(
        test_data, dtype=torch.float32
    ).to(device)

    test_label_true = torch.tensor(
        test_label_true
    ).to(device)

    all_len = train_n + train_p
    class_weights = torch.tensor(
        [1.0 - float(train_n / all_len), 1.0 - float(train_p / all_len)])

    print(train_data.shape)
    return train_data, train_label, test_data, test_label_true, class_weights.to(device)

def get_K_fold_data_KidneyDisease(K, i):
    file_dir = '../data/KidneyDisease'
    batch_1, labels_1, batch_2, labels_2 = read_KidneyDisease(file_dir)

    batch_1 = batch_1 / np.max(batch_1, axis=1)[:, None]
    batch_2 = batch_2 / np.max(batch_2, axis=1)[:, None]

    unique_chars, integer_labels_1 = np.unique(labels_1, return_inverse=True)
    unique_chars, integer_labels_2 = np.unique(labels_2, return_inverse=True)

    train_data = batch_1
    train_label = integer_labels_1
    test_data = batch_2
    test_label_true = integer_labels_2

    train_p = sum(train_label)
    train_n = len(train_label) - train_p

    train_data = torch.tensor(
        train_data, dtype=torch.float32
    ).to(device)

    train_label = torch.tensor(
        train_label
    ).to(device)

    test_data = torch.tensor(
        test_data, dtype=torch.float32
    ).to(device)

    test_label_true = torch.tensor(
        test_label_true
    ).to(device)

    all_len = train_n + train_p
    class_weights = torch.tensor(
        [1.0 - float(train_n / all_len), 1.0 - float(train_p / all_len)])

    print(train_data.shape)
    return train_data, train_label, test_data, test_label_true, class_weights.to(device)

def get_K_fold_data_TOMATO(K, i):
    file_dir = '../data/Tomato'
    batch_1, labels_1, batch_2, labels_2 = read_tomato(file_dir)

    batch_1 = batch_1 / np.max(batch_1, axis=1)[:, None]
    batch_2 = batch_2 / np.max(batch_2, axis=1)[:, None]
    unique_chars, integer_labels_1 = np.unique(labels_1, return_inverse=True)
    unique_chars, integer_labels_2 = np.unique(labels_2, return_inverse=True)


    train_data = batch_1
    train_label = integer_labels_1
    test_data = batch_2
    test_label_true = integer_labels_2

    train_p = sum(train_label)
    train_n = len(train_label) - train_p

    train_data = torch.tensor(
        train_data, dtype=torch.float32
    ).to(device)

    train_label = torch.tensor(
        train_label
    ).to(device)

    test_data = torch.tensor(
        test_data, dtype=torch.float32
    ).to(device)

    test_label_true = torch.tensor(
        test_label_true
    ).to(device)

    all_len = train_n + train_p
    class_weights = torch.tensor(
        [1.0 - float(train_n / all_len), 1.0 - float(train_p / all_len)])

    print(train_data.shape)
    return train_data, train_label, test_data, test_label_true, class_weights.to(device)

def get_K_fold_data_CHD(K, i):
    # 最大K为3
    file_dir = '../data/CHD'
    batch_1, labels_1, batch_2, labels_2, batch_3, labels_3, batch_4, labels_4 = read_CHD_batch(file_dir)

    batch_1 = batch_1 / np.max(batch_1, axis=1)[:,None]
    batch_2 = batch_2 / np.max(batch_2, axis=1)[:,None]
    batch_3 = batch_3 / np.max(batch_3, axis=1)[:,None]
    batch_4 = batch_4 / np.max(batch_4, axis=1)[:,None]
    unique_chars, integer_labels_1 = np.unique(labels_1, return_inverse=True)
    unique_chars, integer_labels_2 = np.unique(labels_2, return_inverse=True)
    unique_chars, integer_labels_3 = np.unique(labels_3, return_inverse=True)
    unique_chars, integer_labels_4 = np.unique(labels_4, return_inverse=True)
    print(batch_1.shape)
    if i == 0:
        train_data = np.append(batch_1,batch_2, axis=0)
        train_label = np.append(integer_labels_1, integer_labels_2)
        test_data = np.append(batch_3,batch_4, axis=0)
        test_label_true = np.append(integer_labels_3, integer_labels_4)
    elif i == 1:
        train_data = np.append(batch_1,batch_3, axis=0)
        train_label = np.append(integer_labels_1, integer_labels_3)
        test_data = np.append(batch_2,batch_4, axis=0)
        test_label_true = np.append(integer_labels_2, integer_labels_4)
    elif i == 2:
        train_data = np.append(batch_1,batch_4, axis=0)
        train_label = np.append(integer_labels_1, integer_labels_4)
        test_data = np.append(batch_2,batch_3, axis=0)
        test_label_true = np.append(integer_labels_2, integer_labels_3)
    elif i==3:
        train_data = np.append(batch_2,batch_3, axis=0)
        train_label = np.append(integer_labels_2, integer_labels_3)
        test_data = np.append(batch_1,batch_4, axis=0)
        test_label_true = np.append(integer_labels_1, integer_labels_4)
    elif i==4:
        train_data = np.append(batch_2,batch_4, axis=0)
        train_label = np.append(integer_labels_2, integer_labels_4)
        test_data = np.append(batch_1,batch_3, axis=0)
        test_label_true = np.append(integer_labels_1, integer_labels_3)
    elif i==5:
        train_data = np.append(batch_3,batch_4, axis=0)
        train_label = np.append(integer_labels_3, integer_labels_4)
        test_data = np.append(batch_1,batch_2, axis=0)
        test_label_true = np.append(integer_labels_1, integer_labels_2)

    train_p = sum(train_label)
    train_n = len(train_label)-train_p

    train_data = torch.tensor(
        train_data, dtype=torch.float32
    ).to(device)

    train_label = torch.tensor(
        train_label
    ).to(device)

    test_data = torch.tensor(
        test_data, dtype=torch.float32
    ).to(device)

    test_label_true = torch.tensor(
        test_label_true
    ).to(device)

    all_len = train_n + train_p
    class_weights = torch.tensor(
        [1.0 - float(train_n / all_len), 1.0 - float(train_p / all_len)])

    print(train_data.shape)
    return train_data, train_label, test_data, test_label_true, class_weights.to(device)

def get_K_fold_data_CHD_Paper(K, i):
    # 最大K为3
    file_dir = '../data/CHD'
    batch_1, labels_1, batch_2, labels_2, batch_3, labels_3, batch_4, labels_4 = read_CHD_batch(file_dir)

    batch_1 = batch_1 / np.max(batch_1, axis=1)[:,None]
    batch_2 = batch_2 / np.max(batch_2, axis=1)[:,None]
    batch_3 = batch_3 / np.max(batch_3, axis=1)[:,None]
    batch_4 = batch_4 / np.max(batch_4, axis=1)[:,None]
    unique_chars, integer_labels_1 = np.unique(labels_1, return_inverse=True)
    unique_chars, integer_labels_2 = np.unique(labels_2, return_inverse=True)
    unique_chars, integer_labels_3 = np.unique(labels_3, return_inverse=True)
    unique_chars, integer_labels_4 = np.unique(labels_4, return_inverse=True)
    print(batch_1.shape)
    if i == 0:
        train_data = np.append(batch_1,batch_2, axis=0)
        train_label = np.append(integer_labels_1, integer_labels_2)
        test_data_1 = batch_3
        test_label_1 = integer_labels_3
        test_data_2 = batch_4
        test_label_2 = integer_labels_4
    elif i == 1:
        train_data = np.append(batch_1,batch_3, axis=0)
        train_label = np.append(integer_labels_1, integer_labels_3)
        test_data_1 = batch_2
        test_label_1 = integer_labels_2
        test_data_2 = batch_4
        test_label_2 = integer_labels_4
    elif i == 2:
        train_data = np.append(batch_1,batch_4, axis=0)
        train_label = np.append(integer_labels_1, integer_labels_4)
        test_data_1 = batch_2
        test_label_1 = integer_labels_2
        test_data_2 = batch_3
        test_label_2 = integer_labels_3
    elif i==3:
        train_data = np.append(batch_2,batch_3, axis=0)
        train_label = np.append(integer_labels_2, integer_labels_3)
        test_data_1 = batch_1
        test_label_1 = integer_labels_1
        test_data_2 = batch_4
        test_label_2 = integer_labels_4
    elif i==4:
        train_data = np.append(batch_2,batch_4, axis=0)
        train_label = np.append(integer_labels_2, integer_labels_4)
        test_data_1 = batch_1
        test_label_1 = integer_labels_1
        test_data_2 = batch_3
        test_label_2 = integer_labels_3
    elif i==5:
        train_data = np.append(batch_3,batch_4, axis=0)
        train_label = np.append(integer_labels_3, integer_labels_4)
        test_data_1 = batch_1
        test_label_1 = integer_labels_1
        test_data_2 = batch_2
        test_label_2 = integer_labels_2

    train_p = sum(train_label)
    train_n = len(train_label)-train_p

    train_data = torch.tensor(
        train_data, dtype=torch.float32
    ).to(device)

    train_label = torch.tensor(
        train_label
    ).to(device)

    test_data_1 = torch.tensor(
        test_data_1, dtype=torch.float32
    ).to(device)

    test_label_1 = torch.tensor(
        test_label_1
    ).to(device)

    test_data_2 = torch.tensor(
        test_data_2, dtype=torch.float32
    ).to(device)

    test_label_2 = torch.tensor(
        test_label_2
    ).to(device)

    all_len = train_n + train_p
    class_weights = torch.tensor(
        [1.0 - float(train_n / all_len), 1.0 - float(train_p / all_len)])

    print(train_data.shape)
    return train_data, train_label, test_data_1, test_label_1, test_data_2, test_label_2, class_weights.to(device)

def get_K_fold_data_MI_pair(K, i):
    # 最大K为3
    file_dir = '../data/MI'
    batch_1, labels_1, batch_2, labels_2, batch_3, labels_3 = read_MI_batch(file_dir)
    batch_1 = batch_1 / np.max(batch_1, axis=1)[:, None]
    batch_2 = batch_2 / np.max(batch_2, axis=1)[:, None]
    batch_3 = batch_3 / np.max(batch_3, axis=1)[:, None]

    unique_chars, integer_labels_1 = np.unique(labels_1, return_inverse=True)
    unique_chars, integer_labels_2 = np.unique(labels_2, return_inverse=True)
    unique_chars, integer_labels_3 = np.unique(labels_3, return_inverse=True)

    print(batch_1.shape)
    if i == 0:
        train_data = batch_1
        train_label = integer_labels_1
        test_data_1 = batch_2
        test_label_1 = integer_labels_2
        test_data_2 = batch_3
        test_label_2 = integer_labels_3
    elif i == 1:
        train_data = batch_2
        train_label = integer_labels_2
        test_data_1 = batch_1
        test_label_1 = integer_labels_1
        test_data_2 = batch_3
        test_label_2 = integer_labels_3

    elif i == 2:
        train_data = batch_3
        train_label = integer_labels_3
        test_data_1 = batch_1
        test_label_1 = integer_labels_1
        test_data_2 = batch_2
        test_label_2 = integer_labels_2

    train_p = sum(train_label)
    train_n = len(train_label)-train_p

    train_data = torch.tensor(
        train_data, dtype=torch.float32
    ).to(device)

    train_label = torch.tensor(
        train_label
    ).to(device)

    test_data_1 = torch.tensor(
        test_data_1, dtype=torch.float32
    ).to(device)

    test_label_1 = torch.tensor(
        test_label_1
    ).to(device)

    test_data_2 = torch.tensor(
        test_data_2, dtype=torch.float32
    ).to(device)

    test_label_2 = torch.tensor(
        test_label_2
    ).to(device)


    all_len = train_n + train_p
    class_weights = torch.tensor(
        [1.0 - float(train_n / all_len), 1.0 - float(train_p / all_len)])

    print(train_data.shape)
    return train_data, train_label, test_data_1, test_label_1, test_data_2, test_label_2, class_weights.to(device)

def get_K_fold_data_CHD_Paper_pair(K, i):
    # 最大K为3
    file_dir = '../data/CHD'
    batch_1, labels_1, batch_2, labels_2, batch_3, labels_3, batch_4, labels_4 = read_CHD_batch(file_dir)

    batch_1 = batch_1 / np.max(batch_1, axis=1)[:,None]
    batch_2 = batch_2 / np.max(batch_2, axis=1)[:,None]
    batch_3 = batch_3 / np.max(batch_3, axis=1)[:,None]
    batch_4 = batch_4 / np.max(batch_4, axis=1)[:,None]
    unique_chars, integer_labels_1 = np.unique(labels_1, return_inverse=True)
    unique_chars, integer_labels_2 = np.unique(labels_2, return_inverse=True)
    unique_chars, integer_labels_3 = np.unique(labels_3, return_inverse=True)
    unique_chars, integer_labels_4 = np.unique(labels_4, return_inverse=True)
    print(batch_1.shape)
    if i == 0:
        train_data = batch_1
        train_label = integer_labels_1
        test_data_1 = batch_2
        test_label_1 = integer_labels_2
        test_data_2 = batch_3
        test_label_2 = integer_labels_3
        test_data_3 = batch_4
        test_label_3 = integer_labels_4
    elif i == 1:
        train_data = batch_2
        train_label = integer_labels_2
        test_data_1 = batch_1
        test_label_1 = integer_labels_1
        test_data_2 = batch_3
        test_label_2 = integer_labels_3
        test_data_3 = batch_4
        test_label_3 = integer_labels_4
    elif i == 2:
        train_data = batch_3
        train_label = integer_labels_3
        test_data_1 = batch_1
        test_label_1 = integer_labels_1
        test_data_2 = batch_2
        test_label_2 = integer_labels_2
        test_data_3 = batch_4
        test_label_3 = integer_labels_4
    elif i==3:
        train_data = batch_4
        train_label = integer_labels_4
        test_data_1 = batch_1
        test_label_1 = integer_labels_1
        test_data_2 = batch_2
        test_label_2 = integer_labels_2
        test_data_3 = batch_3
        test_label_3 = integer_labels_3

    train_p = sum(train_label)
    train_n = len(train_label)-train_p

    train_data = torch.tensor(
        train_data, dtype=torch.float32
    ).to(device)

    train_label = torch.tensor(
        train_label
    ).to(device)

    test_data_1 = torch.tensor(
        test_data_1, dtype=torch.float32
    ).to(device)

    test_label_1 = torch.tensor(
        test_label_1
    ).to(device)

    test_data_2 = torch.tensor(
        test_data_2, dtype=torch.float32
    ).to(device)

    test_label_2 = torch.tensor(
        test_label_2
    ).to(device)

    test_data_3 = torch.tensor(
        test_data_3, dtype=torch.float32
    ).to(device)

    test_label_3 = torch.tensor(
        test_label_3
    ).to(device)

    all_len = train_n + train_p
    class_weights = torch.tensor(
        [1.0 - float(train_n / all_len), 1.0 - float(train_p / all_len)])

    print(train_data.shape)
    return train_data, train_label, test_data_1, test_label_1, test_data_2, test_label_2, test_data_3, test_label_3, class_weights.to(device)

def load_array(data_arrays, batch_size, is_train=True):
    from torch.utils.data.sampler import WeightedRandomSampler
    dataset = data.TensorDataset(*data_arrays)
    # weights = [0.5 if l == 0 else 0.5 for d, l in dataset]
    # sampler = WeightedRandomSampler(weights, num_samples=10, replacement=True)
    return data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=False)
    # return data.DataLoader(dataset, batch_size, sampler=sampler)

class TripletLoss(object):
    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, dist_ap, dist_an):
        y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss.mean()

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    print(dist.shape)
    return dist

def hamming_dist(x, y):
    m, n = x.size(0), y.size(0)
    # 将张量展平为二维矩阵，方便后续按元素比较
    x_flat = x.view(m, -1)
    y_flat = y.view(n, -1)

    # 对展平后的张量按元素进行异或操作，相同为0，不同为1
    xor_result = torch.logical_xor(x_flat.unsqueeze(1).expand(m, n, -1),
                                   y_flat.unsqueeze(0).expand(m, n, -1))

    # 对异或结果按元素求和，得到每个样本对之间的汉明距离
    dist = xor_result.sum(dim=2)

    print(dist.shape)
    return dist

def one_minus_cosine_dist(x, y):
    m, n = x.size(0), y.size(0)

    # 对x和y进行归一化处理，使其向量长度为1
    x_normalized = torch.nn.functional.normalize(x, p=2, dim=1)
    y_normalized = torch.nn.functional.normalize(y, p=2, dim=1)

    # 计算余弦相似度
    cosine_similarity = torch.mm(x_normalized, y_normalized.t())

    # 计算1 - 余弦距离
    dist = 1 - cosine_similarity

    print(dist.shape)
    return dist

def manhattan_dist(x, y):
    m, n = x.size(0), y.size(0)
    # 将x扩展为形状为(m, n, *x.size()[1:])的张量，即每个样本重复n次
    xx = x.unsqueeze(1).expand(m, n, *x.size()[1:])
    # 将y扩展为形状为(m, n, *y.size()[1:])的张量，即每个样本重复m次
    yy = y.unsqueeze(0).expand(m, n, *y.size()[1:])

    # 计算对应元素差值的绝对值
    diff = torch.abs(xx - yy)

    # 在最后一个维度（特征维度）上求和，得到曼哈顿距离
    dist = diff.sum(dim=-1)

    print(dist.shape)
    return dist

def chebyshev_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = x.unsqueeze(1).expand(m, n, *x.size()[1:])
    yy = y.unsqueeze(0).expand(m, n, *y.size()[1:])
    diff = torch.abs(xx - yy)
    dist = torch.max(diff, dim=-1)[0]

    print(dist.shape)
    return dist

def hard_example_mining(dist_mat, labels, return_inds=False):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    pos_dist = dist_mat.masked_fill(is_neg, 0)
    neg_dist = dist_mat.masked_fill(is_pos, 0)

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    # print(dist_mat[is_neg].shape)
    dist_ap, relative_p_inds = torch.max(
        pos_dist.contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        neg_dist.contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)  # compression dimension
    dist_an = dist_an.squeeze(1)

    # calculate the indexs of hard positive and hard negative in dist_mat matrix
    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        pos_dist = ind.masked_fill(is_neg, 0)
        neg_dist = ind.masked_fill(is_pos, 0)
        p_inds = torch.gather(
            pos_dist.contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            neg_dist.contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

def global_loss(tri_loss, global_feat, labels, normalize_feature=True):
    if normalize_feature:
        global_feat = normalize(global_feat, axis=-1)
    # shape [N, N]
    dist_mat = euclidean_dist(global_feat, global_feat)
    # dist_mat = hamming_dist(global_feat, global_feat)
    # dist_mat = one_minus_cosine_dist(global_feat, global_feat)
    # dist_mat = manhattan_dist(global_feat, global_feat)
    # dist_mat = chebyshev_dist(global_feat, global_feat)
    dist_ap, dist_an, p_inds, n_inds = hard_example_mining(
        dist_mat, labels, return_inds=True)
    loss = tri_loss(dist_ap, dist_an)
    return loss, p_inds, n_inds, dist_ap, dist_an, dist_mat

def metrix_results(pred, labels, pred_score, output_size):
    target_names = ['DNK', 'HC', 'IgAN', 'MN']
    tp = np.sum(np.logical_and(np.equal(labels, 1), np.equal(pred, 1)))
    fp = np.sum(np.logical_and(np.equal(labels, 0), np.equal(pred, 1)))
    tn = np.sum(np.logical_and(np.equal(labels, 0), np.equal(pred, 0)))
    fn = np.sum(np.logical_and(np.equal(labels, 1), np.equal(pred, 0)))

    acc_scr = accuracy_score(labels, pred)
    precision_scr = precision_score(labels, pred, average="macro")
    recall_scr = recall_score(labels, pred, average="macro")
    f1_scr = f1_score(labels, pred, average="macro")
    f_scr = 2 * tp / (2 * tp + fp + fn)
    mcc = (tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    # auc_score = roc_auc_score(labels, pred_score)
    auc_score = 0
    if output_size == 4:
        for j in range(4):
            tp = np.sum(np.logical_and(np.equal(labels, j), np.equal(pred, j)))
            fp = np.sum(np.logical_and(np.not_equal(labels, j), np.equal(pred, j)))
            tn = np.sum(np.logical_and(np.not_equal(labels, j), np.not_equal(pred, j)))
            fn = np.sum(np.logical_and(np.equal(labels, j), np.not_equal(pred, j)))
            print(f'{target_names[j]}类：sensitivity = {tp/(tp+fn)},specificity = {tn/(tn+fp)}')

    return acc_scr, precision_scr, recall_scr, f1_scr, f_scr, mcc, sensitivity, specificity, auc_score

def metrix_results3(pred, labels):
    acc_scr = accuracy_score(labels, pred)
    precision_scr = precision_score(labels, pred, average="macro")
    recall_scr = recall_score(labels, pred, average="macro")
    f1_scr = f1_score(labels, pred, average="macro")
    return acc_scr, precision_scr, recall_scr, f1_scr

def metrix_results2(pred, labels):
    target_names = ['DNK', 'HC', 'IgAN', 'MN']
    tp = np.sum(np.logical_and(np.equal(labels, 1), np.equal(pred, 1)))
    fp = np.sum(np.logical_and(np.equal(labels, 0), np.equal(pred, 1)))
    tn = np.sum(np.logical_and(np.equal(labels, 0), np.equal(pred, 0)))
    fn = np.sum(np.logical_and(np.equal(labels, 1), np.equal(pred, 0)))

    acc_scr = accuracy_score(labels, pred)
    precision_scr = precision_score(labels, pred, average="macro")
    recall_scr = recall_score(labels, pred, average="macro")
    f1_scr = f1_score(labels, pred, average="macro")
    f_scr = 2 * tp / (2 * tp + fp + fn)
    mcc = (tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    return acc_scr, precision_scr, recall_scr, f1_scr, f_scr, mcc


def train_model(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size, output_size, label_smooth, num, gamma, loss_name,
                class_weights, dataset_name, trip_lambda):
    from torch.optim.lr_scheduler import LambdaLR
    stop_f1 = 0
    stop_f1_train = 0
    stop_f = 0
    stop_acc = 0
    stop_id = 0
    stop_loss = 1e9
    best_epoch = 0
    tol = 200
    if output_size == 3:
        if dataset_name == 'GCF':
            target_names = ['H', 'TP', 'UP']
        else:
            target_names = ['E-P','N','W-P']
    if output_size == 4:
        target_names = ['DNK', 'HC', 'IgAN', 'MN']
    if output_size == 2:
        if dataset_name == 'ColonCancer':
            target_names = ['E-P','W-P']
        elif dataset_name == 'COVID-19' or 'OC' or 'MI' or 'CC_NP' or 'tomato' or 'Tomato' or 'TOMATO' or 'CHDMSFC':
            target_names = ['N','P']
        elif dataset_name == 'CC_EW':
            target_names = ['E-P', 'W-P']
        elif dataset_name == 'SIMS':
            target_names = ['CER', 'DRG']
        elif dataset_name == 'ICC_rms':
            target_names = ['Astrocytes', 'Neurons']
        elif dataset_name == 'HIP_CER':
            target_names = ['Cerebellar', 'Hippocampal']
    test_ls, ACC_test, Precision_test, Recall_test, F1score_test, Fscore_test, MCC_test, \
     Sensitivity_test, Specificity_test, AUC_test = [], [], [], [], [], [], [], [], [], []
    train_ls, ACC_train, Precision_train, Recall_train, F1score_train, Fscore_train, MCC_train,\
        Sensitivity_train, Specificity_train, AUC_train = [], [], [], [], [], [], [], [], [], []
    last_score, last_precision, last_recall, last_F1 = [], [], [], []
    # 这段代码的作用是将训练数据集中的特征和标签转换为一个迭代器对象，每次迭代返回一个批次的数据。
    # d2l.load_array是一个工具函数，用于将数据转换为迭代器对象.train_iter是一个可以迭代的对象，
    # 每次迭代返回一个形状为(batch_size, feature_dim)的特征张量和一个形状为(batch_size,)的标签张量。
    train_iter = load_array((train_features, train_labels), batch_size,is_train=True)
    # 这里使用的是Adam优化算法
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    for epoch in range(num_epochs):
        lr_epoch = learning_rate * math.pow(0.9,epoch / num_epochs)
        # lr_epoch = learning_rate*(0.995**epoch)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr_epoch, weight_decay=weight_decay)
        for param_group in optimizer.param_groups:
            print("学习率:", param_group['lr'])
        # print("learning rate is ", optimizer.param_groups[0]["lr"])
        for X, y in train_iter:
            # scheduler.step(epoch)
            optimizer.zero_grad()
            pred_for_loss, _, X_triplet, _, _ = net(X)
            if loss_name == 'CEWithSmooth':
                # loss1 = F.binary_cross_entropy(pred_for_loss, y.float())
                # loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smooth)
                loss = nn.CrossEntropyLoss()
                loss1 = loss(pred_for_loss, y)

                triploss = TripletLoss(margin=1.0)
                loss2, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(triploss, X_triplet, y, False)

                l = loss1 + trip_lambda * loss2
                
                print(f'交叉熵：{loss1},三元损失：{trip_lambda * loss2}')
                """
                l = loss1
                """
            elif loss_name == 'FocalLoss':
                loss_fn = FocalLoss(class_num = output_size, gamma=gamma)
                l = loss_fn(pred_for_loss, y)
            else:
                loss = nn.CrossEntropyLoss()
                l = loss(pred_for_loss, y) # + loss_fn(pred_for_loss,y)

            l.requires_grad_(True)
            l.backward()
            optimizer.step()
        #训练集
        pred_c, _, _, _, _= net(train_features)
        pred_c = pred_c.cpu().detach().numpy()
        pred_cs = pred_c.squeeze()

        train_labels_c = train_labels.cpu().detach().numpy()
        train_labels_cs = train_labels_c.squeeze()

        predicted_class = np.argmax(pred_cs, axis=1)

        acc_train, precision_train, recall_train, f1_train, f_train, mcc_train, sensitivity_train, specificity_train, auc_train = metrix_results(predicted_class, train_labels_cs, pred_cs[:,1], output_size)

        # 测试集
        pred_c, _, x_triplet, x_encoder, x_decoder = net(test_features)

        pred_c = pred_c.cpu().detach().numpy()
        pred_cs = pred_c.squeeze()

        test_labels_c = test_labels.cpu().detach().numpy()
        test_labels_cs = test_labels_c.squeeze()

        predicted_class = np.argmax(pred_cs, axis=1)
        if output_size == 2 or 4:
            print("测试集表现")
            acc_test, precision_test, recall_test, f1_test, f_test, mcc_test, sensitivity_test, specificity_test, auc_test = metrix_results(predicted_class, test_labels_cs, pred_cs[:,1], output_size)
        else:
            acc_test, precision_test, recall_test, f1_test = metrix_results3(predicted_class, test_labels_cs)
            f_test = 0
            mcc_test = 0
        pred_train, _, X_triplet_train, _, _ = net(train_features)
        pred_test, _, X_triplet_test, _, _ = net(test_features)

        if loss_name == 'CEWithSmooth':
            # loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smooth)
            loss = nn.CrossEntropyLoss()
            loss1_train = loss(pred_train, train_labels)
            loss1_test = loss(pred_test, test_labels)

            triploss = TripletLoss(margin=1.0)

            loss2_train, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(triploss, X_triplet_train,
                                                                                  train_labels, False)
            loss2_test, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(triploss, X_triplet_test,
                                                                                  test_labels, False)
            train_loss = loss1_train# + trip_lambda * loss2_train
            test_loss = loss1_test# + trip_lambda * loss2_test
            """
            train_loss = loss1_train
            test_loss = loss1_test
            """
            """
            # 二分类损失
            loss = nn.BCEWithLogitsLoss()
            train_loss = loss(pred_train, train_labels)
            test_loss = loss(pred_test, test_labels)
            """
        elif loss_name == 'FocalLoss':
            loss_fn = FocalLoss(class_num=output_size, gamma=gamma)
            train_loss = loss_fn(pred_train, train_labels)
            test_loss = loss_fn(pred_test, test_labels)
        else:
            loss = nn.CrossEntropyLoss()
            train_loss = loss(pred_train, train_labels)
            test_loss = loss(pred_test, test_labels)

        tr_l = train_loss.cpu().detach().numpy()
        t_l = test_loss.cpu().detach().numpy()

        train_ls.append(tr_l)
        ACC_train.append(acc_train)
        Precision_train.append(precision_train)
        Recall_train.append(recall_train)
        F1score_train.append(f1_train)
        Fscore_train.append(f_train)
        MCC_train.append(mcc_train)
        Sensitivity_train.append(sensitivity_train)
        Specificity_train.append(specificity_train)
        AUC_train.append(auc_train)

        test_ls.append(t_l)
        ACC_test.append(acc_test)
        Precision_test.append(precision_test)
        Recall_test.append(recall_test)
        F1score_test.append(f1_test)
        Fscore_test.append(f_test)
        MCC_test.append(mcc_test)
        Sensitivity_test.append(sensitivity_test)
        Specificity_test.append(specificity_test)
        AUC_test.append(auc_test)


        if stop_flag == 'F1':
            if f1_test >= stop_f1:
                stop_f1 = f1_test
                stop_id = len(F1score_test)-1
                true_labels = test_labels_cs
                pred_labels = predicted_class
                print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                np.savetxt(f'data{3 - num}.csv', predicted_class, delimiter=',')
            else:
                if (len(F1score_test)-1)-stop_id >= tol:
                    break
        elif stop_flag == 'Loss_train':
            if train_loss <= stop_loss:
                best_epoch = epoch
                stop_loss = train_loss
                stop_id = len(F1score_test)-1
                true_labels = test_labels_cs
                pred_labels = predicted_class
                print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                np.savetxt(f'data{3 - num}.csv', predicted_class, delimiter=',')
            else:
                if (len(F1score_test)-1)-stop_id >= tol:
                    break
        elif stop_flag == 'F1_train':
            if f1_train > stop_f1_train:
                best_epoch = epoch
                stop_f1_train = f1_train
                stop_id = len(F1score_test) - 1
                true_labels = test_labels_cs
                pred_labels = predicted_class
                print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                np.savetxt(f'data{3 - num}.csv', predicted_class, delimiter=',')
                pred_c, _, x_triplet, x_encoder, x_decoder = net(test_features)
                np.savetxt(f'encoder_result-{dataset_name}.csv', x_encoder.cpu().detach().numpy(), delimiter=',')
                np.savetxt(f'decoder_result-{dataset_name}.csv', x_decoder.cpu().detach().numpy(), delimiter=',')
                np.savetxt(f'mamba_result-{dataset_name}.csv', x_triplet.cpu().detach().numpy(), delimiter=',')
        elif stop_flag == 'ACC_train':
            if acc_train >= stop_acc:
                best_epoch = epoch
                stop_acc = acc_train
                stop_id = len(F1score_test) - 1
                true_labels = test_labels_cs
                pred_labels = predicted_class
                print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                np.savetxt(f'data{3 - num}.csv', predicted_class, delimiter=',')
            else:
                if (len(F1score_test) - 1) - stop_id >= tol:
                    break
        elif stop_flag == 'F_train':
            if f_train > stop_f:
                best_epoch = epoch
                stop_f = f_train
                stop_id = len(F1score_test) - 1
                true_labels = test_labels_cs
                pred_labels = predicted_class
                print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                np.savetxt(f'data{3 - num}.csv', predicted_class, delimiter=',')
                pred_c, _, x_triplet, x_encoder, x_decoder = net(test_features)
                np.savetxt(f'encoder_result-{dataset_name}.csv', x_encoder.cpu().detach().numpy(), delimiter=',')
                np.savetxt(f'decoder_result-{dataset_name}.csv', x_decoder.cpu().detach().numpy(), delimiter=',')
                np.savetxt(f'mamba_result-{dataset_name}.csv', x_triplet.cpu().detach().numpy(), delimiter=',')
            else:
                if (len(F1score_test) - 1) - stop_id >= tol:
                    break

        print(f'iters:{epoch}')
        if stop_flag == 'Loss_train':
            print(f'训练损失:{train_loss}, 第{best_epoch}次迭代取得最低损失：{stop_loss}, '
                  f'对应的测试集ACC：{ACC_test[F1score_train.index(max(F1score_train))]}, '
                  f'F1:{F1score_test[F1score_train.index(max(F1score_train))]}, '
                  f'F:{Fscore_test[F1score_train.index(max(F1score_train))]}')
        if stop_flag == 'F1':
            print(f'F1:{f1_test},best_F1:{max(F1score_test)}')
        if stop_flag == 'F1_train':
            print(f'训练集F1:{f1_train},第{best_epoch}次迭代取得最高F1{max(F1score_train)}, '
                  f'对应的测试集ACC:{ACC_test[best_epoch]}, '
                  f'F1：{F1score_test[best_epoch]}, '
                  f'F：{Fscore_test[best_epoch]}, '
                  f'Precision:{Precision_test[best_epoch]}, '
                  f'Sensitivity:{Sensitivity_test[best_epoch]}, '
                  f'Specificity:{Specificity_test[best_epoch]}, '
                  f'AUC:{AUC_test[best_epoch]}')
        if stop_flag == 'ACC_train':
            print(f'训练集ACC:{acc_train},第{best_epoch}次迭代取得最高ACC:{max(ACC_train)}, '
                  f'对应的测试集ACC:{ACC_test[F1score_train.index(max(F1score_train))]}, '
                  f'F1：{F1score_test[F1score_train.index(max(F1score_train))]}, '
                  f'F：{Fscore_test[F1score_train.index(max(F1score_train))]}')
        if stop_flag == 'F_train':
            print(f'训练集F:{f_train},第{best_epoch}次迭代取得最高F{max(Fscore_train)}, '
                  f'对应的测试集ACC:{ACC_test[best_epoch]}, '
                  f'F1：{F1score_test[best_epoch]}, '
                  f'F：{Fscore_test[best_epoch]}, '
                  f'Precision:{Precision_test[best_epoch]}, '
                  f'Sensitivity:{Sensitivity_test[best_epoch]}, '
                  f'Specificity:{Specificity_test[best_epoch]}, '
                  f'AUC:{AUC_test[best_epoch]}')
    return (test_ls, ACC_test, Precision_test, Recall_test, F1score_test, Fscore_test, MCC_test, Sensitivity_test, Specificity_test, AUC_test,
            train_ls, ACC_train, Precision_train, Recall_train, F1score_train, Fscore_train, MCC_train, Sensitivity_train, Specificity_train, AUC_train,
            true_labels, pred_labels)

def train_model_for_CHD_Paper(net, train_features, train_labels, test_features_1, test_labels_1,test_features_2, test_labels_2,
          num_epochs, learning_rate, weight_decay, batch_size, output_size, label_smooth, num, gamma, loss_name,
                class_weights, dataset_name):
    stop_f1 = 0
    stop_f1_train = 0
    stop_f = 0
    stop_acc = 0
    stop_id = 0
    stop_loss = 1e9
    best_epoch = 0
    tol = 200
    trip_lambda = 0
    if output_size == 3:
        target_names = ['E-P', 'N', 'W-P']
    if output_size == 2:
        if dataset_name == 'ColonCancer':
            target_names = ['E-P', 'W-P']
        elif dataset_name == 'COVID-19' or 'OC' or 'MI':
            target_names = ['N', 'P']
        elif dataset_name == 'SIMS':
            target_names = ['CER', 'DRG']
        elif dataset_name == 'ICC_rms':
            target_names = ['Astrocytes', 'Neurons']
        elif dataset_name == 'HIP_CER':
            target_names = ['Cerebellar', 'Hippocampal']
    (locals()[f'test_ls_{1}'], locals()[f'ACC_test_{1}'], locals()[f'Precision_test_{1}'], locals()[f'Recall_test_{1}'],
     locals()[f'F1score_test_{1}'], locals()[f'Fscore_test_{1}'], locals()[f'MCC_test_{1}']) = [], [], [], [], [], [], []
    (locals()[f'test_ls_{2}'], locals()[f'ACC_test_{2}'], locals()[f'Precision_test_{2}'], locals()[f'Recall_test_{2}'],
     locals()[f'F1score_test_{2}'], locals()[f'Fscore_test_{2}'],
     locals()[f'MCC_test_{2}']) = [], [], [], [], [], [], []
    train_ls, ACC_train, Precision_train, Recall_train, F1score_train, Fscore_train, MCC_train = [], [], [], [], [], [], []
    last_score, last_precision, last_recall, last_F1 = [], [], [], []
    # 这段代码的作用是将训练数据集中的特征和标签转换为一个迭代器对象，每次迭代返回一个批次的数据。
    # d2l.load_array是一个工具函数，用于将数据转换为迭代器对象.train_iter是一个可以迭代的对象，
    # 每次迭代返回一个形状为(batch_size, feature_dim)的特征张量和一个形状为(batch_size,)的标签张量。
    train_iter = load_array((train_features, train_labels), batch_size, is_train=True)
    # 这里使用的是Adam优化算法
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    for epoch in range(num_epochs):
        lr_epoch = learning_rate * math.pow(0.9, epoch / num_epochs)
        # lr_epoch = learning_rate*(0.995**epoch)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr_epoch, weight_decay=weight_decay)
        for param_group in optimizer.param_groups:
            print("学习率:", param_group['lr'])
        # print("learning rate is ", optimizer.param_groups[0]["lr"])
        for X, y in train_iter:
            # scheduler.step(epoch)
            optimizer.zero_grad()
            pred_for_loss, _, X_triplet, _, _ = net(X)
            if loss_name == 'CEWithSmooth':
                # loss1 = F.binary_cross_entropy(pred_for_loss, y.float())
                # loss = nn.CrossEntropyLoss(label_smoothing=label_smooth)
                loss = nn.CrossEntropyLoss()
                loss1 = loss(pred_for_loss, y)
                triploss = TripletLoss(margin=1.0)
                loss2, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(triploss, X_triplet, y, False)

                l = loss1 + trip_lambda * loss2
                print(f'交叉熵：{loss1},三元损失：{trip_lambda * loss2}')
            elif loss_name == 'FocalLoss':
                loss_fn = FocalLoss(class_num=output_size, gamma=gamma)
                l = loss_fn(pred_for_loss, y)
            else:
                loss = nn.CrossEntropyLoss()
                l = loss(pred_for_loss, y)  # + loss_fn(pred_for_loss,y)

            l.requires_grad_(True)
            l.backward()
            optimizer.step()
        # 训练集
        pred_train, _, X_triplet_train, _, _ = net(train_features)
        pred_c = pred_train.cpu().detach().numpy()
        pred_cs = pred_c.squeeze()

        train_labels_c = train_labels.cpu().detach().numpy()
        train_labels_cs = train_labels_c.squeeze()

        predicted_class = np.argmax(pred_cs, axis=1)

        acc_train, precision_train, recall_train, f1_train, f_train, mcc_train = metrix_results2(predicted_class, train_labels_cs)

        locals()[f'test_features_{1}'] = test_features_1
        locals()[f'test_labels_{1}'] = test_labels_1
        locals()[f'test_features_{2}'] = test_features_2
        locals()[f'test_labels_{2}'] = test_labels_2

        if loss_name == 'CEWithSmooth':
            # loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smooth)
            loss = nn.CrossEntropyLoss()
            loss1_train = loss(pred_train, train_labels)
            triploss = TripletLoss(margin=1.0)
            loss2_train, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(triploss, X_triplet_train,
                                                                                  train_labels, False)
            train_loss = loss1_train + trip_lambda * loss2_train

        tr_l = train_loss.cpu().detach().numpy()
        train_ls.append(tr_l)
        ACC_train.append(acc_train)
        Precision_train.append(precision_train)
        Recall_train.append(recall_train)
        F1score_train.append(f1_train)
        Fscore_train.append(f_train)
        MCC_train.append(mcc_train)
        # 测试集
        for i in range(1,3):
            pred_test, _, X_triplet_test, _, _ = net(locals()[f'test_features_{i}'])
            pred_c = pred_test.cpu().detach().numpy()
            pred_cs = pred_c.squeeze()

            test_labels_c = locals()[f'test_labels_{i}'].cpu().detach().numpy()
            test_labels_cs = test_labels_c.squeeze()

            predicted_class = np.argmax(pred_cs, axis=1)
            acc_test, precision_test, recall_test, f1_test, f_test, mcc_test = metrix_results2(predicted_class,
                                                                                                    test_labels_cs)
            locals()[f'test_ls_{i}'].append(0)
            locals()[f'ACC_test_{i}'].append(acc_test)
            locals()[f'Precision_test_{i}'].append(precision_test)
            locals()[f'Recall_test_{i}'].append(recall_test)
            locals()[f'F1score_test_{i}'].append(f1_test)
            locals()[f'Fscore_test_{i}'].append(f_test)
            locals()[f'MCC_test_{i}'].append(mcc_test)

            if stop_flag == 'F1':
                if f1_test >= stop_f1:
                    stop_f1 = f1_test
                    stop_id = len(locals()[f'F1score_test_{i}']) - 1
                    locals()[f'true_labels_{i}'] = test_labels_cs
                    locals()[f'pred_labels_{i}'] = predicted_class
                    print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                    np.savetxt(f'data{6 - num}_{i}.csv', predicted_class, delimiter=',')
                else:
                    if (len(locals()[f'F1score_test_{i}']) - 1) - stop_id >= tol:
                        break
            elif stop_flag == 'Loss_train':
                if train_loss <= stop_loss:
                    best_epoch = epoch
                    stop_loss = train_loss
                    stop_id = len(locals()[f'F1score_test_{i}']) - 1
                    locals()[f'true_labels_{i}'] = test_labels_cs
                    locals()[f'pred_labels_{i}'] = predicted_class
                    print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                    np.savetxt(f'data{6 - num}_{i}.csv', predicted_class, delimiter=',')
                else:
                    if (len(locals()[f'F1score_test_{i}']) - 1) - stop_id >= tol:
                        break
            elif stop_flag == 'F1_train':
                if f1_train >= stop_f1_train:
                    best_epoch = epoch
                    stop_f1_train = f1_train
                    stop_id = len(locals()[f'F1score_test_{i}']) - 1
                    locals()[f'true_labels_{i}'] = test_labels_cs
                    locals()[f'pred_labels_{i}'] = predicted_class
                    print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                    np.savetxt(f'data{6 - num}_{i}.csv', predicted_class, delimiter=',')
                else:
                    if (len(locals()[f'F1score_test_{i}']) - 1) - stop_id >= tol:
                        break
            elif stop_flag == 'ACC_train':
                if acc_train >= stop_acc:
                    best_epoch = epoch
                    stop_acc = acc_train
                    stop_id = len(locals()[f'F1score_test_{i}']) - 1
                    locals()[f'true_labels_{i}'] = test_labels_cs
                    locals()[f'pred_labels_{i}'] = predicted_class
                    print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                    np.savetxt(f'data{6 - num}_{i}.csv', predicted_class, delimiter=',')
                else:
                    if (len(locals()[f'F1score_test_{i}']) - 1) - stop_id >= tol:
                        break
            elif stop_flag == 'F_train':
                if f_train >= stop_f:
                    best_epoch = epoch
                    stop_f = f_train
                    stop_id = best_epoch
                    locals()[f'true_labels_{i}'] = test_labels_cs
                    locals()[f'pred_labels_{i}'] = predicted_class
                    print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                    np.savetxt(f'data{6 - num}_{i}.csv', predicted_class, delimiter=',')
        # 打印每次迭代最优测试结果（按照不同训练集指标选择）
        print(f'iters:{epoch}')
        if stop_flag == 'Loss_train':
            print(f'训练损失:{train_loss}, 第{best_epoch}次迭代取得最低损失：{stop_loss}')
            for i in range(1,3):
                print(f'####### 对应的测试集 ####### {num}-{i} ########')
                ACC_test = locals()[f'ACC_test_{i}']
                F1score_test = locals()[f'F1score_test_{i}']
                Fscore_test = locals()[f'Fscore_test_{i}']
                print(f'{i}-ACC：{ACC_test[tr_l.index(min(tr_l))]},'
                      f'{i}-F1:{F1score_test[tr_l.index(max(tr_l))]}, '
                      f'{i}-F:{Fscore_test[tr_l.index(max(tr_l))]}')
        if stop_flag == 'F1_train':
            print(f'训练损失:{train_loss}, 第{best_epoch}次迭代取得最高F1：{stop_f1}')
            for i in range(1, 3):
                print(f'####### 对应的测试集 ####### {num}-{i} ########')
                ACC_test = locals()[f'ACC_test_{i}']
                F1score_test = locals()[f'F1score_test_{i}']
                Fscore_test = locals()[f'Fscore_test_{i}']
                print(f'{i}-ACC：{ACC_test[F1score_train.index(max(F1score_train))]},'
                      f'{i}-F1:{F1score_test[F1score_train.index(max(F1score_train))]}, '
                      f'{i}-F:{Fscore_test[F1score_train.index(max(F1score_train))]}')
        if stop_flag == 'ACC_train':
            print(f'训练损失:{train_loss}, 第{best_epoch}次迭代取得最高ACC：{stop_acc}')
            for i in range(1, 3):
                print(f'####### 对应的测试集 ####### {num}-{i} ########')
                ACC_test = locals()[f'ACC_test_{i}']
                F1score_test = locals()[f'F1score_test_{i}']
                Fscore_test = locals()[f'Fscore_test_{i}']

                print(f'{i}-ACC：{ACC_test[ACC_train.index(stop_acc)]},'
                      f'{i}-F1:{F1score_test[ACC_train.index(stop_acc)]},'
                      f'{i}-F:{Fscore_test[ACC_train.index(stop_acc)]}')
        if stop_flag == 'F_train':
            print(f'训练损失:{train_loss}, 第{best_epoch}次迭代取得最高F：{stop_f}')
            for i in range(1, 3):
                print(f'####### 对应的测试集 ####### {num}-{i} ########')
                ACC_test = locals()[f'ACC_test_{i}']
                F1score_test = locals()[f'F1score_test_{i}']
                Fscore_test = locals()[f'Fscore_test_{i}']
                print(f'{i}-ACC：{ACC_test[Fscore_train.index(stop_f)]},'
                      f'{i}-F1:{F1score_test[Fscore_train.index(stop_f)]}, '
                      f'{i}-F:{Fscore_test[Fscore_train.index(stop_f)]}')
    return ((locals()[f'test_ls_{1}'], locals()[f'ACC_test_{1}'], locals()[f'Precision_test_{1}'], locals()[f'Recall_test_{1}'],
            locals()[f'F1score_test_{1}'], locals()[f'Fscore_test_{1}'], locals()[f'MCC_test_{1}'],
            locals()[f'test_ls_{2}'], locals()[f'ACC_test_{2}'], locals()[f'Precision_test_{2}'], locals()[f'Recall_test_{2}'],
            locals()[f'F1score_test_{2}'], locals()[f'Fscore_test_{2}'], locals()[f'MCC_test_{2}']),
            locals()[f'true_labels_{1}'], locals()[f'pred_labels_{1}'],
            locals()[f'true_labels_{2}'], locals()[f'pred_labels_{2}'],
            train_ls, ACC_train, Precision_train, Recall_train, F1score_train, Fscore_train, MCC_train)

def train_model_for_CHD_Paper_pair(net, train_features, train_labels, test_features_1, test_labels_1,test_features_2, test_labels_2,
          test_features_3, test_labels_3, num_epochs, learning_rate, weight_decay, batch_size, output_size, label_smooth, num, gamma, loss_name,
                class_weights, dataset_name):
    stop_f1 = 0
    stop_f1_train = 0
    stop_f = 0
    stop_acc = 0
    stop_id = 0
    stop_loss = 1e9
    best_epoch = 0
    tol = 200
    trip_lambda = 0
    if output_size == 3:
        target_names = ['E-P', 'N', 'W-P']
    if output_size == 2:
        if dataset_name == 'ColonCancer':
            target_names = ['E-P', 'W-P']
        elif dataset_name == 'COVID-19' or 'OC' or 'MI':
            target_names = ['N', 'P']
        elif dataset_name == 'SIMS':
            target_names = ['CER', 'DRG']
        elif dataset_name == 'ICC_rms':
            target_names = ['Astrocytes', 'Neurons']
        elif dataset_name == 'HIP_CER':
            target_names = ['Cerebellar', 'Hippocampal']
    (locals()[f'test_ls_{1}'], locals()[f'ACC_test_{1}'], locals()[f'Precision_test_{1}'], locals()[f'Recall_test_{1}'],
     locals()[f'F1score_test_{1}'], locals()[f'Fscore_test_{1}'], locals()[f'MCC_test_{1}']) = [], [], [], [], [], [], []
    (locals()[f'test_ls_{2}'], locals()[f'ACC_test_{2}'], locals()[f'Precision_test_{2}'], locals()[f'Recall_test_{2}'],
     locals()[f'F1score_test_{2}'], locals()[f'Fscore_test_{2}'],
     locals()[f'MCC_test_{2}']) = [], [], [], [], [], [], []
    (locals()[f'test_ls_{3}'], locals()[f'ACC_test_{3}'], locals()[f'Precision_test_{3}'], locals()[f'Recall_test_{3}'],
     locals()[f'F1score_test_{3}'], locals()[f'Fscore_test_{3}'],
     locals()[f'MCC_test_{3}']) = [], [], [], [], [], [], []
    train_ls, ACC_train, Precision_train, Recall_train, F1score_train, Fscore_train, MCC_train = [], [], [], [], [], [], []
    last_score, last_precision, last_recall, last_F1 = [], [], [], []
    # 这段代码的作用是将训练数据集中的特征和标签转换为一个迭代器对象，每次迭代返回一个批次的数据。
    # d2l.load_array是一个工具函数，用于将数据转换为迭代器对象.train_iter是一个可以迭代的对象，
    # 每次迭代返回一个形状为(batch_size, feature_dim)的特征张量和一个形状为(batch_size,)的标签张量。
    train_iter = load_array((train_features, train_labels), batch_size, is_train=True)
    # 这里使用的是Adam优化算法
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    for epoch in range(num_epochs):
        lr_epoch = learning_rate * math.pow(0.9, epoch / num_epochs)
        # lr_epoch = learning_rate*(0.995**epoch)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr_epoch, weight_decay=weight_decay)
        for param_group in optimizer.param_groups:
            print("学习率:", param_group['lr'])
        # print("learning rate is ", optimizer.param_groups[0]["lr"])
        for X, y in train_iter:
            # scheduler.step(epoch)
            optimizer.zero_grad()
            pred_for_loss, _, X_triplet, _, _ = net(X)
            if loss_name == 'CEWithSmooth':
                # loss1 = F.binary_cross_entropy(pred_for_loss, y.float())
                # loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smooth)
                loss = nn.CrossEntropyLoss()
                loss1 = loss(pred_for_loss, y)
                triploss = TripletLoss(margin=1.0)
                loss2, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(triploss, X_triplet, y, False)

                l = loss1 + trip_lambda * loss2
                print(f'交叉熵：{loss1},三元损失：{trip_lambda * loss2}')
            elif loss_name == 'FocalLoss':
                loss_fn = FocalLoss(class_num=output_size, gamma=gamma)
                l = loss_fn(pred_for_loss, y)
            else:
                loss = nn.CrossEntropyLoss()
                l = loss(pred_for_loss, y)  # + loss_fn(pred_for_loss,y)

            l.requires_grad_(True)
            l.backward()
            optimizer.step()
        # 训练集
        pred_train, _, X_triplet_train, _, _ = net(train_features)
        pred_c = pred_train.cpu().detach().numpy()
        pred_cs = pred_c.squeeze()

        train_labels_c = train_labels.cpu().detach().numpy()
        train_labels_cs = train_labels_c.squeeze()

        predicted_class = np.argmax(pred_cs, axis=1)

        acc_train, precision_train, recall_train, f1_train, f_train, mcc_train = metrix_results2(predicted_class,
                                                                                                train_labels_cs)

        locals()[f'test_features_{1}'] = test_features_1
        locals()[f'test_labels_{1}'] = test_labels_1
        locals()[f'test_features_{2}'] = test_features_2
        locals()[f'test_labels_{2}'] = test_labels_2
        locals()[f'test_features_{3}'] = test_features_3
        locals()[f'test_labels_{3}'] = test_labels_3

        if loss_name == 'CEWithSmooth':
            # loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smooth)
            loss = nn.CrossEntropyLoss()
            loss1_train = loss(pred_train, train_labels)
            triploss = TripletLoss(margin=1.0)
            loss2_train, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(triploss, X_triplet_train,
                                                                                  train_labels, False)
            train_loss = loss1_train + trip_lambda * loss2_train

        tr_l = train_loss.cpu().detach().numpy()
        train_ls.append(tr_l)
        ACC_train.append(acc_train)
        Precision_train.append(precision_train)
        Recall_train.append(recall_train)
        F1score_train.append(f1_train)
        Fscore_train.append(f_train)
        MCC_train.append(mcc_train)
        # 测试集
        for i in range(1,4):
            pred_test, _, X_triplet_test, _, _ = net(locals()[f'test_features_{i}'])
            pred_c = pred_test.cpu().detach().numpy()
            pred_cs = pred_c.squeeze()

            test_labels_c = locals()[f'test_labels_{i}'].cpu().detach().numpy()
            test_labels_cs = test_labels_c.squeeze()

            predicted_class = np.argmax(pred_cs, axis=1)
            acc_test, precision_test, recall_test, f1_test, f_test, mcc_test = metrix_results2(predicted_class,
                                                                                                    test_labels_cs)
            locals()[f'test_ls_{i}'].append(0)
            locals()[f'ACC_test_{i}'].append(acc_test)
            locals()[f'Precision_test_{i}'].append(precision_test)
            locals()[f'Recall_test_{i}'].append(recall_test)
            locals()[f'F1score_test_{i}'].append(f1_test)
            locals()[f'Fscore_test_{i}'].append(f_test)
            locals()[f'MCC_test_{i}'].append(mcc_test)

            if stop_flag == 'F1':
                if f1_test >= stop_f1:
                    stop_f1 = f1_test
                    stop_id = len(locals()[f'F1score_test_{i}']) - 1
                    locals()[f'true_labels_{i}'] = test_labels_cs
                    locals()[f'pred_labels_{i}'] = predicted_class
                    print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                    np.savetxt(f'data{num}_{i}.csv', predicted_class, delimiter=',')
                else:
                    if (len(locals()[f'F1score_test_{i}']) - 1) - stop_id >= tol:
                        break
            elif stop_flag == 'Loss_train':
                if train_loss <= stop_loss:
                    best_epoch = epoch
                    stop_loss = train_loss
                    stop_id = len(locals()[f'F1score_test_{i}']) - 1
                    locals()[f'true_labels_{i}'] = test_labels_cs
                    locals()[f'pred_labels_{i}'] = predicted_class
                    print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                    np.savetxt(f'data{num}_{i}.csv', predicted_class, delimiter=',')
                else:
                    if (len(locals()[f'F1score_test_{i}']) - 1) - stop_id >= tol:
                        break
            elif stop_flag == 'F1_train':
                if f1_train >= stop_f1_train:
                    best_epoch = epoch
                    stop_f1_train = f1_train
                    stop_id = len(locals()[f'F1score_test_{i}']) - 1
                    locals()[f'true_labels_{i}'] = test_labels_cs
                    locals()[f'pred_labels_{i}'] = predicted_class
                    print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                    np.savetxt(f'data{num}_{i}.csv', predicted_class, delimiter=',')
                else:
                    if (len(locals()[f'F1score_test_{i}']) - 1) - stop_id >= tol:
                        break
            elif stop_flag == 'ACC_train':
                if acc_train >= stop_acc:
                    best_epoch = epoch
                    stop_acc = acc_train
                    stop_id = len(locals()[f'F1score_test_{i}']) - 1
                    locals()[f'true_labels_{i}'] = test_labels_cs
                    locals()[f'pred_labels_{i}'] = predicted_class
                    print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                    np.savetxt(f'data{num}_{i}.csv', predicted_class, delimiter=',')
                else:
                    if (len(locals()[f'F1score_test_{i}']) - 1) - stop_id >= tol:
                        break
            elif stop_flag == 'F_train':
                if f_train >= stop_f:
                    best_epoch = epoch
                    stop_f = f_train
                    stop_id = best_epoch
                    locals()[f'true_labels_{i}'] = test_labels_cs
                    locals()[f'pred_labels_{i}'] = predicted_class
                    print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                    np.savetxt(f'data{num}_{i}.csv', predicted_class, delimiter=',')
        # 打印每次迭代最优测试结果（按照不同训练集指标选择）
        print(f'iters:{epoch}')
        if stop_flag == 'Loss_train':
            print(f'训练损失:{train_loss}, 第{best_epoch}次迭代取得最低损失：{stop_loss}')
            for i in range(1,4):
                print(f'####### 对应的测试集 ####### {num}-{i} ########')
                ACC_test = locals()[f'ACC_test_{i}']
                F1score_test = locals()[f'F1score_test_{i}']
                Fscore_test = locals()[f'Fscore_test_{i}']
                print(f'{i}-ACC：{ACC_test[tr_l.index(min(tr_l))]},'
                      f'{i}-F1:{F1score_test[tr_l.index(max(tr_l))]}, '
                      f'{i}-F:{Fscore_test[tr_l.index(max(tr_l))]}')
        if stop_flag == 'F1_train':
            print(f'训练损失:{train_loss}, 第{best_epoch}次迭代取得最高F1：{stop_f1}')
            for i in range(1, 4):
                print(f'####### 对应的测试集 ####### {num}-{i} ########')
                ACC_test = locals()[f'ACC_test_{i}']
                F1score_test = locals()[f'F1score_test_{i}']
                Fscore_test = locals()[f'Fscore_test_{i}']
                print(f'{i}-ACC：{ACC_test[F1score_train.index(max(F1score_train))]},'
                      f'{i}-F1:{F1score_test[F1score_train.index(max(F1score_train))]}, '
                      f'{i}-F:{Fscore_test[F1score_train.index(max(F1score_train))]}')
        if stop_flag == 'ACC_train':
            print(f'训练损失:{train_loss}, 第{best_epoch}次迭代取得最高ACC：{stop_acc}')
            for i in range(1, 4):
                print(f'####### 对应的测试集 ####### {num}-{i} ########')
                ACC_test = locals()[f'ACC_test_{i}']
                F1score_test = locals()[f'F1score_test_{i}']
                Fscore_test = locals()[f'Fscore_test_{i}']

                print(f'{i}-ACC：{ACC_test[ACC_train.index(stop_acc)]},'
                      f'{i}-F1:{F1score_test[ACC_train.index(stop_acc)]},'
                      f'{i}-F:{Fscore_test[ACC_train.index(stop_acc)]}')
        if stop_flag == 'F_train':
            print(f'训练损失:{train_loss}, 第{best_epoch}次迭代取得最高F：{stop_f}')
            for i in range(1, 4):
                print(f'####### 对应的测试集 ####### {num}-{i} ########')
                ACC_test = locals()[f'ACC_test_{i}']
                F1score_test = locals()[f'F1score_test_{i}']
                Fscore_test = locals()[f'Fscore_test_{i}']
                print(f'{i}-ACC：{ACC_test[Fscore_train.index(stop_f)]},'
                      f'{i}-F1:{F1score_test[Fscore_train.index(stop_f)]}, '
                      f'{i}-F:{Fscore_test[Fscore_train.index(stop_f)]}')
    return ((locals()[f'test_ls_{1}'], locals()[f'ACC_test_{1}'], locals()[f'Precision_test_{1}'], locals()[f'Recall_test_{1}'],
            locals()[f'F1score_test_{1}'], locals()[f'Fscore_test_{1}'], locals()[f'MCC_test_{1}'],
            locals()[f'test_ls_{2}'], locals()[f'ACC_test_{2}'], locals()[f'Precision_test_{2}'], locals()[f'Recall_test_{2}'],
            locals()[f'F1score_test_{2}'], locals()[f'Fscore_test_{2}'], locals()[f'MCC_test_{2}'],
            locals()[f'test_ls_{3}'], locals()[f'ACC_test_{3}'], locals()[f'Precision_test_{3}'], locals()[f'Recall_test_{3}'],
            locals()[f'F1score_test_{3}'], locals()[f'Fscore_test_{3}'], locals()[f'MCC_test_{3}']),
            locals()[f'true_labels_{1}'], locals()[f'pred_labels_{1}'],
            locals()[f'true_labels_{2}'], locals()[f'pred_labels_{2}'],
            locals()[f'true_labels_{3}'], locals()[f'pred_labels_{3}'],
            train_ls, ACC_train, Precision_train, Recall_train, F1score_train, Fscore_train, MCC_train)

def train_model_for_MI_pair(net, train_features, train_labels, test_features_1, test_labels_1,test_features_2, test_labels_2,
          num_epochs, learning_rate, weight_decay, batch_size, output_size, label_smooth, num, gamma, loss_name,
                class_weights, dataset_name):
    stop_f1 = 0
    stop_f1_train = 0
    stop_f = 0
    stop_acc = 0
    stop_id = 0
    stop_loss = 1e9
    best_epoch = 0
    tol = 200
    trip_lambda = 0.01
    if output_size == 3:
        target_names = ['E-P', 'N', 'W-P']
    if output_size == 2:
        if dataset_name == 'ColonCancer':
            target_names = ['E-P', 'W-P']
        elif dataset_name == 'COVID-19' or 'OC' or 'MI':
            target_names = ['N', 'P']
        elif dataset_name == 'SIMS':
            target_names = ['CER', 'DRG']
        elif dataset_name == 'ICC_rms':
            target_names = ['Astrocytes', 'Neurons']
        elif dataset_name == 'HIP_CER':
            target_names = ['Cerebellar', 'Hippocampal']
    (locals()[f'test_ls_{1}'], locals()[f'ACC_test_{1}'], locals()[f'Precision_test_{1}'], locals()[f'Recall_test_{1}'],
     locals()[f'F1score_test_{1}'], locals()[f'Fscore_test_{1}'], locals()[f'MCC_test_{1}']) = [], [], [], [], [], [], []
    (locals()[f'test_ls_{2}'], locals()[f'ACC_test_{2}'], locals()[f'Precision_test_{2}'], locals()[f'Recall_test_{2}'],
     locals()[f'F1score_test_{2}'], locals()[f'Fscore_test_{2}'],
     locals()[f'MCC_test_{2}']) = [], [], [], [], [], [], []
    train_ls, ACC_train, Precision_train, Recall_train, F1score_train, Fscore_train, MCC_train = [], [], [], [], [], [], []
    last_score, last_precision, last_recall, last_F1 = [], [], [], []
    # 这段代码的作用是将训练数据集中的特征和标签转换为一个迭代器对象，每次迭代返回一个批次的数据。
    # d2l.load_array是一个工具函数，用于将数据转换为迭代器对象.train_iter是一个可以迭代的对象，
    # 每次迭代返回一个形状为(batch_size, feature_dim)的特征张量和一个形状为(batch_size,)的标签张量。
    train_iter = load_array((train_features, train_labels), batch_size, is_train=True)
    # 这里使用的是Adam优化算法
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    for epoch in range(num_epochs):
        lr_epoch = learning_rate * math.pow(0.5, epoch / num_epochs)
        # lr_epoch = learning_rate*(0.995**epoch)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr_epoch, weight_decay=weight_decay)
        for param_group in optimizer.param_groups:
            print("学习率:", param_group['lr'])
        # print("learning rate is ", optimizer.param_groups[0]["lr"])
        for X, y in train_iter:
            # scheduler.step(epoch)
            optimizer.zero_grad()
            pred_for_loss, _, X_triplet = net(X)
            if loss_name == 'CEWithSmooth':
                # loss1 = F.binary_cross_entropy(pred_for_loss, y.float())
                # loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smooth)
                loss = nn.CrossEntropyLoss()
                loss1 = loss(pred_for_loss, y)
                triploss = TripletLoss(margin=1.0)
                loss2, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(triploss, X_triplet, y, False)

                l = loss1 + trip_lambda * loss2
                print(f'交叉熵：{loss1},三元损失：{trip_lambda * loss2}')
            elif loss_name == 'FocalLoss':
                loss_fn = FocalLoss(class_num=output_size, gamma=gamma)
                l = loss_fn(pred_for_loss, y)
            else:
                loss = nn.CrossEntropyLoss()
                l = loss(pred_for_loss, y)  # + loss_fn(pred_for_loss,y)

            l.requires_grad_(True)
            l.backward()
            optimizer.step()
        # 训练集
        pred_train, _, X_triplet_train = net(train_features)
        pred_c = pred_train.cpu().detach().numpy()
        pred_cs = pred_c.squeeze()

        train_labels_c = train_labels.cpu().detach().numpy()
        train_labels_cs = train_labels_c.squeeze()

        predicted_class = np.argmax(pred_cs, axis=1)

        acc_train, precision_train, recall_train, f1_train, f_train, mcc_train = metrix_results(predicted_class,
                                                                                                train_labels_cs)


        locals()[f'test_features_{1}'] = test_features_1
        locals()[f'test_labels_{1}'] = test_labels_1
        locals()[f'test_features_{2}'] = test_features_2
        locals()[f'test_labels_{2}'] = test_labels_2

        if loss_name == 'CEWithSmooth':
            # loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smooth)
            loss = nn.CrossEntropyLoss()
            loss1_train = loss(pred_train, train_labels)
            triploss = TripletLoss(margin=1.0)
            loss2_train, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(triploss, X_triplet_train,
                                                                                  train_labels, False)
            train_loss = loss1_train + trip_lambda * loss2_train

        tr_l = train_loss.cpu().detach().numpy()
        train_ls.append(tr_l)
        ACC_train.append(acc_train)
        Precision_train.append(precision_train)
        Recall_train.append(recall_train)
        F1score_train.append(f1_train)
        Fscore_train.append(f_train)
        MCC_train.append(mcc_train)
        # 测试集
        for i in range(1,3):
            pred_test, _, X_triplet_test = net(locals()[f'test_features_{i}'])
            pred_c = pred_test.cpu().detach().numpy()
            pred_cs = pred_c.squeeze()

            test_labels_c = locals()[f'test_labels_{i}'].cpu().detach().numpy()
            test_labels_cs = test_labels_c.squeeze()

            predicted_class = np.argmax(pred_cs, axis=1)
            acc_test, precision_test, recall_test, f1_test, f_test, mcc_test = metrix_results(predicted_class,test_labels_cs)
            locals()[f'test_ls_{i}'].append(0)
            locals()[f'ACC_test_{i}'].append(acc_test)
            locals()[f'Precision_test_{i}'].append(precision_test)
            locals()[f'Recall_test_{i}'].append(recall_test)
            locals()[f'F1score_test_{i}'].append(f1_test)
            locals()[f'Fscore_test_{i}'].append(f_test)
            locals()[f'MCC_test_{i}'].append(mcc_test)

            if stop_flag == 'F1':
                if f1_test >= stop_f1:
                    stop_f1 = f1_test
                    stop_id = len(locals()[f'F1score_test_{i}']) - 1
                    locals()[f'true_labels_{i}'] = test_labels_cs
                    locals()[f'pred_labels_{i}'] = predicted_class
                    print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                    np.savetxt(f'data{num}_{i}.csv', predicted_class, delimiter=',')
                else:
                    if (len(locals()[f'F1score_test_{i}']) - 1) - stop_id >= tol:
                        break
            elif stop_flag == 'Loss_train':
                if train_loss <= stop_loss:
                    best_epoch = epoch
                    stop_loss = train_loss
                    stop_id = len(locals()[f'F1score_test_{i}']) - 1
                    locals()[f'true_labels_{i}'] = test_labels_cs
                    locals()[f'pred_labels_{i}'] = predicted_class
                    print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                    np.savetxt(f'data{num}_{i}.csv', predicted_class, delimiter=',')
                else:
                    if (len(locals()[f'F1score_test_{i}']) - 1) - stop_id >= tol:
                        break
            elif stop_flag == 'F1_train':
                if f1_train > stop_f1:
                    best_epoch = epoch
                    stop_f = f1_train
                    stop_id = best_epoch
                    locals()[f'true_labels_{i}'] = test_labels_cs
                    locals()[f'pred_labels_{i}'] = predicted_class
                    print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                    np.savetxt(f'data{num}_{i}.csv', predicted_class, delimiter=',')
            elif stop_flag == 'ACC_train':
                if acc_train >= stop_acc:
                    best_epoch = epoch
                    stop_acc = acc_train
                    stop_id = len(locals()[f'F1score_test_{i}']) - 1
                    locals()[f'true_labels_{i}'] = test_labels_cs
                    locals()[f'pred_labels_{i}'] = predicted_class
                    print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                    np.savetxt(f'data{num}_{i}.csv', predicted_class, delimiter=',')
                else:
                    if (len(locals()[f'F1score_test_{i}']) - 1) - stop_id >= tol:
                        break
            elif stop_flag == 'F_train':
                if f_train >= stop_f:
                    best_epoch = epoch
                    stop_f = f_train
                    stop_id = best_epoch
                    locals()[f'true_labels_{i}'] = test_labels_cs
                    locals()[f'pred_labels_{i}'] = predicted_class
                    print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                    np.savetxt(f'data{num}_{i}.csv', predicted_class, delimiter=',')
        # 打印每次迭代最优测试结果（按照不同训练集指标选择）
        print(f'iters:{epoch}')
        if stop_flag == 'Loss_train':
            print(f'训练损失:{train_loss}, 第{best_epoch}次迭代取得最低损失：{stop_loss}')
            for i in range(1,3):
                print(f'####### 对应的测试集 ####### {num}-{i} ########')
                ACC_test = locals()[f'ACC_test_{i}']
                F1score_test = locals()[f'F1score_test_{i}']
                Fscore_test = locals()[f'Fscore_test_{i}']
                print(f'{i}-ACC：{ACC_test[tr_l.index(min(tr_l))]},'
                      f'{i}-F1:{F1score_test[tr_l.index(max(tr_l))]}, '
                      f'{i}-F:{Fscore_test[tr_l.index(max(tr_l))]}')
        if stop_flag == 'F1_train':
            print(f'训练损失:{train_loss}, 第{best_epoch}次迭代取得最高F：{stop_f1}')
            for i in range(1, 3):
                print(f'####### 对应的测试集 ####### {num}-{i} ########')
                ACC_test = locals()[f'ACC_test_{i}']
                F1score_test = locals()[f'F1score_test_{i}']
                Fscore_test = locals()[f'Fscore_test_{i}']
                print(f'{i}-ACC：{ACC_test[F1score_train.index(stop_f1)]},'
                      f'{i}-F1:{F1score_test[F1score_train.index(stop_f1)]}, '
                      f'{i}-F:{Fscore_test[F1score_train.index(stop_f1)]}')
        if stop_flag == 'ACC_train':
            print(f'训练损失:{train_loss}, 第{best_epoch}次迭代取得最高ACC：{stop_acc}')
            for i in range(1, 3):
                print(f'####### 对应的测试集 ####### {num}-{i} ########')
                ACC_test = locals()[f'ACC_test_{i}']
                F1score_test = locals()[f'F1score_test_{i}']
                Fscore_test = locals()[f'Fscore_test_{i}']

                print(f'{i}-ACC：{ACC_test[ACC_train.index(stop_acc)]},'
                      f'{i}-F1:{F1score_test[ACC_train.index(stop_acc)]},'
                      f'{i}-F:{Fscore_test[ACC_train.index(stop_acc)]}')
        if stop_flag == 'F_train':
            print(f'训练损失:{train_loss}, 第{best_epoch}次迭代取得最高F：{stop_f}')
            for i in range(1, 3):
                print(f'####### 对应的测试集 ####### {num}-{i} ########')
                ACC_test = locals()[f'ACC_test_{i}']
                F1score_test = locals()[f'F1score_test_{i}']
                Fscore_test = locals()[f'Fscore_test_{i}']
                print(f'{i}-ACC：{ACC_test[Fscore_train.index(stop_f)]},'
                      f'{i}-F1:{F1score_test[Fscore_train.index(stop_f)]}, '
                      f'{i}-F:{Fscore_test[Fscore_train.index(stop_f)]}')
    return ((locals()[f'test_ls_{1}'], locals()[f'ACC_test_{1}'], locals()[f'Precision_test_{1}'], locals()[f'Recall_test_{1}'],
            locals()[f'F1score_test_{1}'], locals()[f'Fscore_test_{1}'], locals()[f'MCC_test_{1}'],
            locals()[f'test_ls_{2}'], locals()[f'ACC_test_{2}'], locals()[f'Precision_test_{2}'], locals()[f'Recall_test_{2}'],
            locals()[f'F1score_test_{2}'], locals()[f'Fscore_test_{2}'], locals()[f'MCC_test_{2}']),
            locals()[f'true_labels_{1}'], locals()[f'pred_labels_{1}'],
            locals()[f'true_labels_{2}'], locals()[f'pred_labels_{2}'],
            train_ls, ACC_train, Precision_train, Recall_train, F1score_train, Fscore_train, MCC_train)

def get_net(in_features, state_size, median_size, output_size, model_name):
    # net = nn.Sequential(nn.Linear(in_features, 3)).to(device)
    # net = nn.Sequential(MambaSkipLinear2(seq_len=1, d_model=in_features, state_size=64, output_size=3)).to(device)
    # net = nn.Sequential(MambaSkipLinear(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size, output_size=output_size)).to(device)
    # net = nn.Sequential(Mamba(seq_len=1, d_model=in_features, median_size = median_size, state_size=state_size, output_size=3)).to(device)
    # net = TF(in_features=in_features, drop=0.).to(device)
    if model_name == "LSTM":
        net = LSTM(input_size=in_features, hidden_size=median_size, output_size=output_size).to(device)
    if model_name == "RNN":
        net = RNN(input_size=in_features, hidden_size=median_size, output_size=output_size).to(device)

    if model_name == "MSDREDFeaMiC":
        net = MSDREDFeaMiC(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size,
                                      output_size=output_size).to(device)
    if model_name == "Mamba":
        net = nn.Sequential(pureMamba(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size,
                                      output_size=output_size)).to(device)
    if model_name == "Transformer":
        net = nn.Sequential(Transformer(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size,
                                      output_size=output_size)).to(device)
    if model_name == "MLP":
        net = nn.Sequential(baselinemlp(in_features=in_features, median_size=median_size, output_size = output_size)).to(device)

    return net

if __name__ == '__main__':
    # get_K_fold_data(5,0)
    # num_epochs, learning_rate, weight_decay, batch_size = 50, 1e-4, 1e-7, 64
    # output_size, state_size, median_size = 3, 128, 512
    test_size = 0.2
    trip_lambda = 0.01
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int, default = 50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--os', type=int, default=2)
    parser.add_argument('--ss', type=int, default=256)
    parser.add_argument('--ms', type=int, default=512)
    parser.add_argument('--label_smooth',type=float,default=0.1)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--loss', type=str, default='CEWithSmooth')
    parser.add_argument('--file', type=str, default='bin_0.1')
    parser.add_argument('--model', type=str, default='MSDREDFeaMiC')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='CC_EW')
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--balance', type=int, default=0)
    parser.add_argument('--stop_flag', type=str, default='F1_train')
    args = parser.parse_args()

    num_epochs, learning_rate, weight_decay, batch_size = args.epochs, args.lr, args.wd, args.bs
    output_size, state_size, median_size = args.os, args.ss, args.ms

    gamma = args.gamma
    label_smooth = args.label_smooth
    loss_name = args.loss
    model_name = args.model
    seed = args.seed
    dataset_name = args.dataset
    K = args.K
    balance_flag = args.balance
    stop_flag = args.stop_flag
    seed_everything(seed)
    if dataset_name == 'MI':
        K = 3
    if dataset_name == 'CHD_Paper':
        K = 6
    if dataset_name == 'CHD_Paper_Pair':
        K = 4
    ACC_epoch = np.zeros([K, num_epochs // 10])
    F1_epoch = np.zeros([K, num_epochs // 10])
    F_epoch = np.zeros([K, num_epochs // 10])
    print('__________parameters start_____________')
    if output_size == 3:
        print("_________WP、EP、N三分类任务设置___________")
    elif output_size == 2:
        print("__________WP和EP二分类任务设置_____________")
    elif output_size == 4:
        print("__________DNK、HC、IgAN和MN四分类任务设置_____________")
    # print(f'input_size={in_features}')
    print(f'output_size={output_size}')
    print(f'iters={num_epochs}')
    print(f'batch_size={batch_size}')
    print(f'test_size={test_size}')
    print(f'learning_rate={learning_rate}')
    print(f'CELoss参数label_smooth={label_smooth}')
    print(f'state_size={state_size}')
    print(f'median_size={median_size}')
    print('___________parameters end______________')
    ACC = np.zeros((K))
    PRE = np.zeros((K))
    REC = np.zeros((K))
    F1S = np.zeros((K))
    FS = np.zeros((K))
    MCC = np.zeros((K))
    Sensitivity = np.zeros((K))
    Specificity = np.zeros((K))
    AUC = np.zeros((K))
    ACC_last = np.zeros((K))
    PRE_last = np.zeros((K))
    REC_last = np.zeros((K))
    F1S_last = np.zeros((K))
    FS_last = np.zeros((K))
    MCC_last = np.zeros((K))
    Sensitivity_last = np.zeros((K))
    Specificity_last = np.zeros((K))
    AUC_last = np.zeros((K))

    for j in range(1, 4):
        locals()[f'ACC_{j}'] = np.zeros(K)
        locals()[f'PRE_{j}'] = np.zeros(K)
        locals()[f'REC_{j}'] = np.zeros(K)
        locals()[f'F1S_{j}'] = np.zeros(K)
        locals()[f'FS_{j}'] = np.zeros(K)
        locals()[f'mcc_{j}'] = np.zeros(K)
        locals()[f'sensitivity_{j}'] = np.zeros(K)
        locals()[f'specificity_{j}'] = np.zeros(K)

    for i in range(K):
        if output_size == 3:
            if dataset_name == 'ColonCancer':
                file_name = r'../data/' + dataset_name + '/' + args.file
                train_data, train_labels, test_data, test_labels, class_weights = get_K_fold_data_CC_3(K,i,file_name)
        else:
            if dataset_name == 'CC_EW':
                file_name = args.file
                train_data, train_labels, test_data, test_labels, class_weights = get_K_fold_data_CC(K, i, file_name, 'EW')
            elif dataset_name == 'CC_NP':
                file_name = args.file
                train_data, train_labels, test_data, test_labels, class_weights = get_K_fold_data_CC(K, i, file_name, 'NP')
            elif dataset_name == 'MI':
                train_data, train_labels, test_data, test_labels, class_weights = get_K_fold_data_MI(K, i)
            elif dataset_name == 'MI_Pair':
                train_data, train_labels, test_data_1, test_labels_1, test_data_2, test_labels_2, class_weights = get_K_fold_data_MI_pair(
                    K, i)
            elif dataset_name == 'KidneyDisease':
                train_data, train_labels, test_data, test_labels, class_weights = get_K_fold_data_KidneyDisease(K, i)
            elif dataset_name == 'tomato':
                train_data, train_labels, test_data, test_labels, class_weights = get_K_fold_data_TOMATO(K, i)
            elif dataset_name == 'CHD':
                train_data, train_labels, test_data, test_labels, class_weights = get_K_fold_data_CHD(K, i)
            elif dataset_name == 'CHD_Paper':
                train_data, train_labels, test_data_1, test_labels_1, test_data_2, test_labels_2, class_weights = get_K_fold_data_CHD_Paper(K, i)
            elif dataset_name == 'CHD_Paper_Pair':
                train_data, train_labels, test_data_1, test_labels_1, test_data_2, test_labels_2, test_data_3, test_labels_3, class_weights = get_K_fold_data_CHD_Paper_pair(K, i)
            in_features = train_data.shape[1]
        net = get_net(in_features, state_size, median_size, output_size,model_name)
        # net = nn.DataParallel(net)

        print(train_data.shape)
        if balance_flag==1 and dataset_name!='CHD_Paper':
            train_loss, test_loss, score, precision, recall, F1, true_labels, pred_labels = train_model_balance(net, train_data,
                                                                                                    train_labels,
                                                                                                    test_data,
                                                                                                    test_labels,
                                                                                                    num_epochs,
                                                                                                    learning_rate,
                                                                                                    weight_decay,
                                                                                                    batch_size,
                                                                                                    output_size,
                                                                                                    label_smooth, i,
                                                                                                    gamma, loss_name,
                                                                                                    class_weights,
                                                                                                    dataset_name)
        elif balance_flag==0 and dataset_name != 'CHD_Paper' and dataset_name != 'CHD_Paper_Pair' and dataset_name!='MI_Pair':
            (test_loss, score, precision, recall, F1, Fsc, MC, sensi, speci, auc,
             train_loss, score_train, precision_train, recall_train, F1_train, Fsc_train, MC_train, sensi_train, speci_train, auc_train,
             true_labels, pred_labels) = train_model(net, train_data, train_labels,test_data,test_labels,
                                                     num_epochs,learning_rate,weight_decay,batch_size,output_size,
                                                     label_smooth, i,gamma, loss_name,class_weights,dataset_name, trip_lambda)

        elif dataset_name == 'CHD_Paper' or dataset_name == 'MI_Pair':
            ((locals()[f'test_loss_{1}'], locals()[f'score_{1}'], locals()[f'precision_{1}'], locals()[f'recall_{1}'],
             locals()[f'F1score_{1}'], locals()[f'Fscore_{1}'], locals()[f'MCC_{1}'],
             locals()[f'test_loss_{2}'], locals()[f'score_{2}'], locals()[f'precision_{2}'], locals()[f'recall_{2}'],
             locals()[f'F1score_{2}'], locals()[f'Fscore_{2}'], locals()[f'MCC_{2}']),
             locals()[f'true_labels_{1}'], locals()[f'pred_labels_{1}'],
             locals()[f'true_labels_{2}'], locals()[f'pred_labels_{2}'],
             train_loss, ACC_train, Precision_train, Recall_train, F1score_train, Fscore_train, MCC_train) \
                = train_model_for_CHD_Paper(net, train_data, train_labels, test_data_1, test_labels_1, test_data_2,
                                            test_labels_2, num_epochs, learning_rate, weight_decay, batch_size,
                                            output_size, label_smooth, i, gamma, loss_name, class_weights, dataset_name)
        elif dataset_name == 'CHD_Paper_Pair':
            ((locals()[f'test_loss_{1}'], locals()[f'score_{1}'], locals()[f'precision_{1}'], locals()[f'recall_{1}'],
             locals()[f'F1score_{1}'], locals()[f'Fscore_{1}'], locals()[f'MCC_{1}'],
             locals()[f'test_loss_{2}'], locals()[f'score_{2}'], locals()[f'precision_{2}'], locals()[f'recall_{2}'],
             locals()[f'F1score_{2}'], locals()[f'Fscore_{2}'], locals()[f'MCC_{2}'],
             locals()[f'test_loss_{3}'], locals()[f'score_{3}'], locals()[f'precision_{3}'], locals()[f'recall_{3}'],
             locals()[f'F1score_{3}'], locals()[f'Fscore_{3}'], locals()[f'MCC_{3}']),
             locals()[f'true_labels_{1}'], locals()[f'pred_labels_{1}'],
             locals()[f'true_labels_{2}'], locals()[f'pred_labels_{2}'],
             locals()[f'true_labels_{3}'], locals()[f'pred_labels_{3}'],
             train_loss, ACC_train, Precision_train, Recall_train, F1score_train, Fscore_train, MCC_train) \
                = train_model_for_CHD_Paper_pair(net, train_data, train_labels, test_data_1, test_labels_1, test_data_2,
                                            test_labels_2, test_data_3, test_labels_3, num_epochs, learning_rate, weight_decay, batch_size,
                                            output_size, label_smooth, i, gamma, loss_name, class_weights, dataset_name)
        if output_size == 3:
            target_names = ['E-P', 'N', 'W-P']
        elif output_size == 2:
            if dataset_name == 'ColonCancer':
                target_names = ['E-P', 'W-P']
            elif dataset_name == 'COVID-19' or 'OC' or 'MI' or 'CHD' or 'CHD_Paper' or 'CHD_Paper_Pair' or 'tomato' or 'CHDMSFC':
                target_names = ['N', 'P']
            elif dataset_name == 'CC_EW':
                target_names = ['E-P', 'W-P']
            elif dataset_name == 'CC_NP':
                target_names = ['P', 'N']
        elif output_size == 4:
            target_names = ['DNK', 'HC', 'IgAN', 'MN']
        if dataset_name != 'CHD_Paper' and dataset_name != 'CHD_Paper_Pair' and dataset_name!='MI_Pair':
            if i == 0:
                report_1 = classification_report(true_labels, pred_labels, target_names=target_names)
                print(report_1)
            elif i == 1:
                report_2 = classification_report(true_labels, pred_labels, target_names=target_names)
                print(report_2)
            elif i == 2:
                report_3 = classification_report(true_labels, pred_labels, target_names=target_names)
                print(report_3)
            elif i == 3:
                report_4 = classification_report(true_labels, pred_labels, target_names=target_names)
                print(report_4)
            elif i == 4:
                report_5 = classification_report(true_labels, pred_labels, target_names=target_names)
                print(report_5)
            if stop_flag == 'F1':
                idx = F1.index(max(F1))
            elif stop_flag == 'Loss_train':
                idx = train_loss.index(min(train_loss))
            elif stop_flag == 'F1_train':
                idx = F1_train.index(max(F1_train))
            elif stop_flag == 'ACC_train':
                idx = score_train.index(max(score_train))
            elif stop_flag == 'F_train':
                idx = Fsc_train.index(max(Fsc_train))
            df = pd.DataFrame({"训练损失": train_loss,"测试损失": test_loss, "准确率": score, "F1得分": F1, "F得分": Fsc})
            df.to_excel(f'{dataset_name}-batch_{3-i}结果.xlsx', sheet_name="sheet1", index=False)
            print(f'第{i}折的最佳F1得分：')
            ACC[i] = score[idx]
            PRE[i] = precision[idx]
            REC[i] = recall[idx]
            F1S[i] = F1[idx]
            FS[i] = Fsc[idx]
            MCC[i] = MC[idx]
            Sensitivity[i] = sensi[idx]
            Specificity[i] = speci[idx]
            AUC[i] = auc[idx]
            ACC_last[i] = score[-1]
            PRE_last[i] = precision[-1]
            REC_last[i] = recall[-1]
            F1S_last[i] = F1[-1]
            FS_last[i] = Fsc[-1]
            MCC_last[i] = MC[-1]
            Sensitivity_last[i] = sensi[-1]
            Specificity_last[i] = speci[-1]
            AUC_last[i] = auc[-1]
            print(f'                准确率（Accuracy）={score[idx]}')
            print(f'                精确率（Precision）={precision[idx]}')
            print(f'                召回率（Recall）={recall[idx]}')
            print(f'                F1得分（F1-score）={F1[idx]}')
            print(f'                F得分（F-score）={Fsc[idx]}')
            print(f'                MCC得分（MCC）={MC[idx]}')
            print(f'                敏感性得分（Sensitivity）={sensi[idx]}')
            print(f'                特异性得分（Specificity）={speci[idx]}')
            print(f'                AUC得分（AUC）={auc[idx]}')

            xx = np.linspace(0, num_epochs, num_epochs)
            train_line = train_loss
            test_line = test_loss
            accuracy_train_line = score_train
            accuracy_test_line = score
            plt.figure(figsize=(8, 6))
            plt.plot(xx, train_line, label = '训练损失')
            plt.plot(xx, test_line, label = '测试损失')
            plt.plot(xx, accuracy_train_line, label = '训练准确率')
            plt.plot(xx, accuracy_test_line, label = '测试准确率')
            plt.rcParams['font.sans-serif'] = ['simHei']  # 中文显示
            plt.legend(loc='upper right')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.show()
            #df = pd.DataFrame({"训练损失": train_loss,"测试损失": test_loss, "准确率": score, "F1得分": F1})
            #df.to_excel(f'{dataset_name}-{model_name}-{output_size}分类-第{i}折曲线.xlsx', sheet_name="sheet1", index=False)

        elif dataset_name == 'CHD_Paper' or dataset_name == 'MI_Pair':
            if stop_flag == 'Loss_train':
                idx = train_loss.index(min(train_loss))
            elif stop_flag == 'F1_train':
                idx = F1score_train.index(max(F1score_train))
            elif stop_flag == 'ACC_train':
                idx = ACC_train.index(max(ACC_train))
            elif stop_flag == 'F_train':
                idx = Fscore_train.index(max(Fscore_train))
            print(f'第{i}折的最佳F1得分：')
            for j in range(1,3):
                locals()[f'ACC_{j}'][i] = locals()[f'score_{j}'][idx]
                locals()[f'PRE_{j}'][i] = locals()[f'precision_{j}'][idx]
                locals()[f'REC_{j}'][i] = locals()[f'recall_{j}'][idx]
                locals()[f'F1S_{j}'][i] = locals()[f'F1score_{j}'][idx]
                locals()[f'FS_{j}'][i] = locals()[f'Fscore_{j}'][idx]
                locals()[f'mcc_{j}'][i] = locals()[f'MCC_{j}'][idx]

                v1 = locals()[f'score_{j}'][idx]
                v2 = locals()[f'precision_{j}'][idx]
                v3 = locals()[f'recall_{j}'][idx]
                v4 = locals()[f'F1score_{j}'][idx]
                v5 = locals()[f'Fscore_{j}'][idx]
                v6 = locals()[f'MCC_{j}'][idx]
                print(f'第{j}个batch的最佳结果为：')
                print(f'                准确率（Accuracy）={v1}')
                print(f'                精确率（Precision）={v2}')
                print(f'                召回率（Recall）={v3}')
                print(f'                F1得分（F1-score）={v4}')
                print(f'                F得分（F-score）={v5}')
                print(f'                MCC得分（F1-score）={v6}')
        elif dataset_name == 'CHD_Paper_Pair':
            if stop_flag == 'Loss_train':
                idx = train_loss.index(min(train_loss))
            elif stop_flag == 'F1_train':
                idx = F1score_train.index(max(F1score_train))
            elif stop_flag == 'ACC_train':
                idx = ACC_train.index(max(ACC_train))
            elif stop_flag == 'F_train':
                idx = Fscore_train.index(max(Fscore_train))
            print(f'第{i}折的最佳F1得分：')
            for j in range(1,4):
                locals()[f'ACC_{j}'][i] = locals()[f'score_{j}'][idx]
                locals()[f'PRE_{j}'][i] = locals()[f'precision_{j}'][idx]
                locals()[f'REC_{j}'][i] = locals()[f'recall_{j}'][idx]
                locals()[f'F1S_{j}'][i] = locals()[f'F1score_{j}'][idx]
                locals()[f'FS_{j}'][i] = locals()[f'Fscore_{j}'][idx]
                locals()[f'mcc_{j}'][i] = locals()[f'MCC_{j}'][idx]

                v1 = locals()[f'score_{j}'][idx]
                v2 = locals()[f'precision_{j}'][idx]
                v3 = locals()[f'recall_{j}'][idx]
                v4 = locals()[f'F1score_{j}'][idx]
                v5 = locals()[f'Fscore_{j}'][idx]
                v6 = locals()[f'MCC_{j}'][idx]
                print(f'第{j}个batch的最佳结果为：')
                print(f'                准确率（Accuracy）={v1}')
                print(f'                精确率（Precision）={v2}')
                print(f'                召回率（Recall）={v3}')
                print(f'                F1得分（F1-score）={v4}')
                print(f'                F得分（F-score）={v5}')
                print(f'                MCC得分（F1-score）={v6}')
    if dataset_name!='CHD_Paper' and dataset_name!='CHD_Paper_Pair' and dataset_name!='MI_Pair':
        results = [np.mean(PRE), np.mean(REC), np.mean(ACC), np.mean(F1S), np.mean(FS), np.mean(abs(MCC))]
        print(f'{K}折每次结果：')
        print(f'准确率={ACC}')
        print(f'精确率={PRE}')
        print(f'召回率={REC}')
        print(f'F1得分={F1S}')
        print(f'F得分={FS}')
        print(f'MCC得分={MCC}')
        print(f'{K}折平均结果：[精确率, 召回率, 准确率, F1得分, F得分, MCC得分]=[{results[0]},{results[1]},{results[2]},{results[3]},{results[4]},{results[5]}]')

        print(f'{K}折第{num_epochs}次结果：')
        print(f'准确率={ACC_last}')
        print(f'精确率={PRE_last}')
        print(f'召回率={REC_last}')
        print(f'F1得分={F1S_last}')
        print(f'F得分={FS_last}')
        print(f'MCC得分={MCC_last}')
        results = [np.mean(PRE_last), np.mean(REC_last), np.mean(ACC_last), np.mean(F1S_last), np.mean(FS_last), np.mean(abs(MCC_last))]
        print(f'{K}折平均结果：[精确率, 召回率, 准确率, F1得分, F得分, MCC得分]=[{results[0]},{results[1]},{results[2]},{results[3]},{results[4]},{results[5]}]')

        print(np.average(ACC_epoch, axis=0))
        print(np.average(F1_epoch, axis=0))
        print(np.average(F_epoch, axis=0))
    elif dataset_name == 'CHD_Paper' or dataset_name == 'MI_Pair':
        print(f'{K}折每次结果：')
        for j in range(1,3):
            results = [np.mean(locals()[f'PRE_{j}']), np.mean(locals()[f'REC_{j}']), np.mean(locals()[f'ACC_{j}']), np.mean(locals()[f'F1S_{j}']), np.mean(locals()[f'FS_{j}']), np.mean(abs(locals()[f'mcc_{j}']))]
            PRE = locals()[f'PRE_{j}']
            REC = locals()[f'REC_{j}']
            ACC = locals()[f'ACC_{j}']
            F1S = locals()[f'F1S_{j}']
            FS = locals()[f'FS_{j}']
            MCC = locals()[f'mcc_{j}']
            print(f'第{j}个batch的测试结果：')
            print(f'PRE={PRE}')
            print(f'REC={REC}')
            print(f'ACC={ACC}')
            print(f'F1S={F1S}')
            print(f'FS={FS}')
            print(f'MCC={MCC}')
        print(f"平均精准率：{(np.mean(locals()[f'PRE_{1}'])+np.mean(locals()[f'PRE_{2}']))/2.0}")
        print(f"平均召回率：{(np.mean(locals()[f'REC_{1}']) + np.mean(locals()[f'REC_{2}'])) / 2.0}")
        print(f"平均准确率：{(np.mean(locals()[f'ACC_{1}']) + np.mean(locals()[f'ACC_{2}'])) / 2.0}")
        print(f"平均F1：{(np.mean(locals()[f'F1S_{1}']) + np.mean(locals()[f'F1S_{2}'])) / 2.0}")
        print(f"平均F：{(np.mean(locals()[f'FS_{1}']) + np.mean(locals()[f'FS_{2}'])) / 2.0}")
    elif dataset_name == 'CHD_Paper_Pair':
        print(f'{K}折每次结果：')
        for j in range(1,4):
            results = [np.mean(locals()[f'PRE_{j}']), np.mean(locals()[f'REC_{j}']), np.mean(locals()[f'ACC_{j}']), np.mean(locals()[f'F1S_{j}']), np.mean(locals()[f'FS_{j}']), np.mean(abs(locals()[f'mcc_{j}']))]
            PRE = locals()[f'PRE_{j}']
            REC = locals()[f'REC_{j}']
            ACC = locals()[f'ACC_{j}']
            F1S = locals()[f'F1S_{j}']
            FS = locals()[f'FS_{j}']
            MCC = locals()[f'mcc_{j}']
            print(f'第{j}个batch的测试结果：')
            print(f'PRE={PRE}')
            print(f'REC={REC}')
            print(f'ACC={ACC}')
            print(f'F1S={F1S}')
            print(f'FS={FS}')
            print(f'MCC={MCC}')
        print(f"平均精准率：{(np.mean(locals()[f'PRE_{1}'])+np.mean(locals()[f'PRE_{2}'])+np.mean(locals()[f'PRE_{3}']))/3.0}")
        print(f"平均召回率：{(np.mean(locals()[f'REC_{1}']) + np.mean(locals()[f'REC_{2}'])+np.mean(locals()[f'REC_{3}'])) / 3.0}")
        print(f"平均准确率：{(np.mean(locals()[f'ACC_{1}']) + np.mean(locals()[f'ACC_{2}'])+np.mean(locals()[f'ACC_{3}'])) / 3.0}")
        print(f"平均F1：{(np.mean(locals()[f'F1S_{1}']) + np.mean(locals()[f'F1S_{2}'])+np.mean(locals()[f'F1S_{3}'])) / 3.0}")
        print(f"平均F：{(np.mean(locals()[f'FS_{1}']) + np.mean(locals()[f'FS_{2}'])+np.mean(locals()[f'FS_{3}'])) / 3.0}")
