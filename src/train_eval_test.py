from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from read_mouse import read_mouse
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from h5_reader import read_h5
from read_OC import read_lowResolution_8_7_02
from read_MI import read_MI_batch
from read_CHD import read_CHD_batch
from read_mouse import read_mouse
import pandas as pd
import torch
from torch import nn
from torch.utils import data
import numpy as np
from model import (TF,MambaSkipLinear,baselinemlp,Jamba,RNN,LSTM,E_D_Transformer,E_D_Mamba_E_D,E_D_Mamba_NoSample,
                   Dense_MISM_Trans_MoE, E_D_IS_Mamba, E_D_Mamba, pureMamba, Transformer, DenseNet,
                   ConvMamba, CHD_Paper_Method, CHD_2024_Paper)
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
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    dist_ap, dist_an, p_inds, n_inds = hard_example_mining(
        dist_mat, labels, return_inds=True)
    loss = tri_loss(dist_ap, dist_an)
    return loss, p_inds, n_inds, dist_ap, dist_an, dist_mat

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
    # net = nn.Sequential(baselinemlp(in_features, 128)).to(device)
    # net = nn.Sequential(TransMamba(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size, output_size=output_size)).to(device)
    # net = nn.Sequential(Dense_MISM_Trans_MoE(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size, output_size=output_size)).to(device)
    # net = nn.Sequential(DenseNet(init_features=in_features,classes=output_size))
    if model_name == "E_D_IS_Mamba":
        net = nn.Sequential(E_D_IS_Mamba(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size, output_size=output_size)).to(device)
    # net = nn.Sequential(Dense_MISM_Trans_MoE(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size,output_size=output_size)).to(device)
    # net = nn.Sequential(DenseNet(init_features = in_features, classes = output_size)).to(device)
    if model_name == "E_D_Mamba_E_D":
        net = E_D_Mamba_E_D(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size,
                                      output_size=output_size).to(device)
    if model_name == "E_D_Mamba":
        net = E_D_Mamba(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size,
                                      output_size=output_size).to(device)

    if model_name == "E_D_Mamba_NoSample":
        net = E_D_Mamba_NoSample(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size,
                                      output_size=output_size).to(device)
    if model_name == "E_D_Transformer":
        net = E_D_Transformer(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size,
                                      output_size=output_size).to(device)
    if model_name == "Mamba":
        net = nn.Sequential(pureMamba(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size,
                                      output_size=output_size)).to(device)
    if model_name == "Transformer":
        net = nn.Sequential(Transformer(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size,
                                      output_size=output_size)).to(device)
    if model_name == "MLP":
        net = nn.Sequential(baselinemlp(in_features=in_features, median_size=median_size, output_size = output_size)).to(device)
    if model_name == "TransMamba":
        net = nn.Sequential(TransMamba(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size,
                                      output_size=output_size)).to(device)
    if model_name == "Jamba":
        net = nn.Sequential(Jamba(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size,
                                      output_size=output_size)).to(device)
    if model_name == "DenseNet":
        net = nn.Sequential(DenseNet(input_features = in_features,layer_num=(1,1,1,1),growth_rate=1,init_features=median_size,in_channels=median_size,classes=3)).to(device)
    if model_name == "ConvMamba":
        net = ConvMamba(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size,
                        output_size=output_size).to(device)
    if model_name == "CHD_Paper_Method":
        net = CHD_Paper_Method(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size,
                        output_size=output_size).to(device)
    if model_name == "CHD_2024_Paper":
        net = CHD_2024_Paper(seq_len=1, d_model=in_features, median_size=median_size, state_size=state_size,
                        output_size=output_size).to(device)

    return net

def train_model(net, train_features, train_labels, eval_features, eval_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size, output_size, label_smooth, gamma, loss_name,
                class_weights, dataset_name):
    from torch.optim.lr_scheduler import LambdaLR
    stop_f1 = 0
    stop_id = 0
    tol = 200
    if output_size == 3:
        target_names = ['E-P','N','W-P']
    if output_size == 2:
        if dataset_name == 'ColonCancer':
            target_names = ['E-P','W-P']
        elif dataset_name == 'COVID-19' or 'OC' or 'MI':
            target_names = ['N','P']
        elif dataset_name == 'SIMS':
            target_names = ['CER', 'DRG']
        elif dataset_name == 'ICC_rms':
            target_names = ['Astrocytes', 'Neurons']
        elif dataset_name == 'HIP_CER':
            target_names = ['Cerebellar', 'Hippocampal']
    train_ls, test_ls, score, F1score, score_test, F1score_test = [], [], [], [], [], []
    last_score, last_precision, last_recall, last_F1 = [], [], [], []
    # 这段代码的作用是将训练数据集中的特征和标签转换为一个迭代器对象，每次迭代返回一个批次的数据。
    # d2l.load_array是一个工具函数，用于将数据转换为迭代器对象.train_iter是一个可以迭代的对象，
    # 每次迭代返回一个形状为(batch_size, feature_dim)的特征张量和一个形状为(batch_size,)的标签张量。
    train_iter = load_array((train_features, train_labels), batch_size,is_train=True)
    # 这里使用的是Adam优化算法
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    trip_lambda = 0.001
    for epoch in range(num_epochs):
        lr_epoch = learning_rate * math.pow(0.9, epoch / num_epochs)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr_epoch, weight_decay=weight_decay)
        for param_group in optimizer.param_groups:
            print("学习率:", param_group['lr'])

        for X, y in train_iter:
            optimizer.zero_grad()
            pred_for_loss, _,  X_triplet= net(X)
            if loss_name == 'CEWithSmooth':
                loss = nn.CrossEntropyLoss()
                loss1 = loss(pred_for_loss, y)

                triploss = TripletLoss(margin=1.0)
                loss2, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(triploss, X_triplet, y, False)

                # l = loss1
                l = loss1 + trip_lambda * loss2
                print(f'交叉熵：{loss1},三元损失：{trip_lambda * loss2}')
            elif loss_name == 'FocalLoss':
                loss_fn = FocalLoss(class_num = output_size, gamma=gamma)
                l = loss_fn(pred_for_loss, y)
            else:
                loss = nn.CrossEntropyLoss()
                l = loss(pred_for_loss, y) # + loss_fn(pred_for_loss,y)

            l.requires_grad_(True)
            l.backward()
            optimizer.step()

        pred_c, _, _= net(eval_features)
        pred_c = pred_c.cpu().detach().numpy()
        pred_cs = pred_c.squeeze()

        eval_labels_c = eval_labels.cpu().detach().numpy()
        eval_labels_cs = eval_labels_c.squeeze()

        predicted_class = np.argmax(pred_cs, axis=1)

        acc_scr = accuracy_score(eval_labels_cs, predicted_class)
        f1_scr = f1_score(eval_labels_cs, predicted_class,average="macro")

        score.append(acc_scr)
        F1score.append(f1_scr)

        pred_c, _, _ = net(test_features)
        pred_c = pred_c.cpu().detach().numpy()
        pred_cs = pred_c.squeeze()

        test_labels_c = test_labels.cpu().detach().numpy()
        test_labels_cs = test_labels_c.squeeze()

        predicted_class = np.argmax(pred_cs, axis=1)

        acc_scr_test = accuracy_score(test_labels_cs, predicted_class)
        f1_scr_test = f1_score(test_labels_cs, predicted_class, average="macro")
        score_test.append(acc_scr_test)
        F1score_test.append(f1_scr_test)

        best_F1_eval = max(F1score)
        if f1_scr == max(F1score):
            index_best = len(F1score)-1
        else:
            index_best = F1score.index(best_F1_eval)

        print(f'iters:{epoch}')
        print(f'当前：eval_F1:{f1_scr}, test_F1:{f1_scr_test}')
        print(f'验证集最佳F1:{best_F1_eval}, 对应的测试集指标：ACC={score_test[index_best]},F1={F1score_test[index_best]}')

    return score, F1score, score_test, F1score_test

seed_everything(0)
dataset_name = 'ICC_rms'
model_name = 'E_D_Mamba'
label_smooth = 0.1
num_epochs = 50
learning_rate = 5e-4
weight_decay = 1e-3
batch_size = 128
output_size = 2
gamma = 2
loss_name = 'CEWithSmooth'
K = 5
state_size = 256
median_size = 512
ACC_test = np.zeros((K))
F1_test = np.zeros((K))
ii = 0
mat, labels = read_mouse(dataset_name)
unique_chars, integer_labels = np.unique(labels, return_inverse=True)
mat = mat / np.max(mat, axis=1)[:,None]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(mat,integer_labels):
    train_and_eval_data = mat[train_index,:]
    test_data = mat[test_index,:]
    train_and_eval_label = integer_labels[train_index]
    test_label = integer_labels[test_index]
skf = StratifiedKFold(n_splits=K, random_state=42, shuffle=True)
for train_index, eval_index in skf.split(train_and_eval_data, train_and_eval_label):
    train_data = mat[train_index, :]
    eval_data = mat[eval_index, :]
    train_label = integer_labels[train_index]
    eval_label = integer_labels[eval_index]

    train_data = torch.tensor(
        train_data, dtype=torch.float32
    ).to(device)

    train_label = torch.tensor(
        train_label
    ).to(device)

    eval_data = torch.tensor(
        eval_data, dtype=torch.float32
    ).to(device)

    eval_label = torch.tensor(
        eval_label
    ).to(device)

    test_data = torch.tensor(
        test_data, dtype=torch.float32
    ).to(device)

    test_label = torch.tensor(
        test_label
    ).to(device)
    all_len = len(labels)
    p_len = np.sum(integer_labels)
    n_len = all_len-p_len
    class_weights = torch.tensor(
        [1.0 - float(n_len / all_len), 1.0 - float(p_len / all_len)]).to(device)

    in_features = train_data.shape[1]

    net = get_net(in_features, state_size, median_size, 2, model_name)
    net = nn.DataParallel(net)

    acc, F1, acc_t, F1_t = train_model(net, train_data, train_label, eval_data, eval_label, test_data, test_label,num_epochs,learning_rate,
                                                                                                         weight_decay,
                                                                                                         batch_size,
                                                                                                         output_size,
                                                                                                         label_smooth,
                                                                                                         gamma,
                                                                                                         loss_name,
                                                                                                         class_weights,
                                                                                                         dataset_name)
    best_F1_eval = max(F1)
    index_best = F1.index(best_F1_eval)
    ACC_test[ii] = acc_t[index_best]
    F1_test[ii] = F1_t[index_best]
    ii += 1
print(f'{K}个模型的准确率：ACC_test={ACC_test}')
print(f'{K}个模型的F1：F1_test={F1_test}')
print(f'{K}个模型的平均得分：准确率为{np.mean(ACC_test)},F1得分为{np.mean(F1_test)}')