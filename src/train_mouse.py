from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from read_mouse import read_mouse
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
from model import (baselinemlp,RNN,LSTM, MSDREDFeaMiC, pureMamba, Transformer)
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
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import roc_auc_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
stop = 'ACC'
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


def metrix_results(pred, labels):
    tp = np.sum(np.logical_and(np.equal(labels, 1), np.equal(pred, 1)))
    fp = np.sum(np.logical_and(np.equal(labels, 0), np.equal(pred, 1)))
    tn = np.sum(np.logical_and(np.equal(labels, 0), np.equal(pred, 0)))
    fn = np.sum(np.logical_and(np.equal(labels, 1), np.equal(pred, 0)))

    acc_scr = accuracy_score(labels, pred)
    precision_scr = precision_score(labels, pred, average="macro")
    recall_scr = recall_score(labels, pred, average="macro")
    f1_scr = f1_score(labels, pred, average="macro")
    f_scr = 2 * tp / (2 * tp + fp + fn)

    return acc_scr, precision_scr, recall_scr, f1_scr, f_scr

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
        target_names = ['E-P', 'N', 'W-P']
    if output_size == 2:
        if dataset_name == 'ColonCancer':
            target_names = ['E-P', 'W-P']
        elif dataset_name == 'COVID-19' or 'OC' or 'MI' or 'CC_NP' or 'tomato' or 'Tomato' or 'TOMATO' or 'CHDMSFC':
            target_names = ['N', 'P']
        elif dataset_name == 'CC_EW':
            target_names = ['E-P', 'W-P']
        elif dataset_name == 'SIMS':
            target_names = ['CER', 'DRG']
        elif dataset_name == 'ICC_rms':
            target_names = ['Astrocytes', 'Neurons']
        elif dataset_name == 'HIP_CER':
            target_names = ['Cerebellar', 'Hippocampal']
    test_ls, ACC_test, Precision_test, Recall_test, F1score_test, Fscore_test = [], [], [], [], [], []
    train_ls, ACC_train, Precision_train, Recall_train, F1score_train, Fscore_train = [], [], [], [], [], []
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
                """
                l = loss1
                """
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
        pred_c, _, _, _, _ = net(train_features)
        pred_c = pred_c.cpu().detach().numpy()
        pred_cs = pred_c.squeeze()

        train_labels_c = train_labels.cpu().detach().numpy()
        train_labels_cs = train_labels_c.squeeze()

        predicted_class = np.argmax(pred_cs, axis=1)

        acc_train, precision_train, recall_train, f1_train, f_train = metrix_results(predicted_class,
                                                                                                train_labels_cs)

        # 测试集
        pred_c, _, x_triplet, x_encoder, x_decoder = net(test_features)

        pred_c = pred_c.cpu().detach().numpy()
        pred_cs = pred_c.squeeze()

        test_labels_c = test_labels.cpu().detach().numpy()
        test_labels_cs = test_labels_c.squeeze()

        predicted_class = np.argmax(pred_cs, axis=1)
        if output_size == 2:
            acc_test, precision_test, recall_test, f1_test, f_test = metrix_results(predicted_class,
                                                                                              test_labels_cs)
        else:
            acc_test, precision_test, recall_test, f1_test = metrix_results3(predicted_class, test_labels_cs)
            f_test = 0
            mcc_test = 0
        pred_train, _, X_triplet_train, _, _ = net(train_features)
        pred_test, _, X_triplet_test, _, _= net(test_features)

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
            train_loss = loss1_train + trip_lambda * loss2_train
            test_loss = loss1_test + trip_lambda * loss2_test
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


        test_ls.append(t_l)
        ACC_test.append(acc_test)
        Precision_test.append(precision_test)
        Recall_test.append(recall_test)
        F1score_test.append(f1_test)
        Fscore_test.append(f_test)


        if stop_flag == 'F1':
            if f1_test >= stop_f1:
                stop_f1 = f1_test
                stop_id = len(F1score_test) - 1
                true_labels = test_labels_cs
                pred_labels = predicted_class
                print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                np.savetxt(f'data{3 - num}.csv', predicted_class, delimiter=',')
            else:
                if (len(F1score_test) - 1) - stop_id >= tol:
                    break
        elif stop_flag == 'Loss_train':
            if train_loss <= stop_loss:
                best_epoch = epoch
                stop_loss = train_loss
                stop_id = len(F1score_test) - 1
                true_labels = test_labels_cs
                pred_labels = predicted_class
                print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                np.savetxt(f'data{3 - num}.csv', predicted_class, delimiter=',')
            else:
                if (len(F1score_test) - 1) - stop_id >= tol:
                    break
        elif stop_flag == 'F1_train':
            if f1_train >= stop_f1_train:
                best_epoch = epoch
                stop_f1_train = f1_train
                stop_id = len(F1score_test) - 1
                true_labels = test_labels_cs
                pred_labels = predicted_class
                print(print(classification_report(test_labels_cs, predicted_class, target_names=target_names)))
                np.savetxt(f'data{3 - num}.csv', predicted_class, delimiter=',')
            else:
                if (len(F1score_test) - 1) - stop_id >= tol:
                    break
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
                print(roc_auc_score(test_labels_cs,predicted_class))
                np.savetxt(f'data{3 - num}.csv', predicted_class, delimiter=',')
                pred_c, _, x_triplet, x_encoder, x_decoder = net(test_features)
                np.savetxt(f'input-{dataset_name}.csv', test_features.cpu().detach().numpy(), delimiter=',')
                np.savetxt(f'labels-{dataset_name}.csv', test_labels_cs, delimiter=',')
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
            print(f'训练集F1:{f1_train},第{best_epoch}次迭代取得最高F1:{max(F1score_train)}, '
                  f'对应的测试集ACC:{ACC_test[F1score_train.index(max(F1score_train))]}, '
                  f'F1：{F1score_test[F1score_train.index(max(F1score_train))]}, '
                  f'F：{Fscore_test[F1score_train.index(max(F1score_train))]}')
        if stop_flag == 'ACC_train':
            print(f'训练集ACC:{acc_train},第{best_epoch}次迭代取得最高ACC:{max(ACC_train)}, '
                  f'对应的测试集ACC:{ACC_test[F1score_train.index(max(F1score_train))]}, '
                  f'F1：{F1score_test[F1score_train.index(max(F1score_train))]}, '
                  f'F：{Fscore_test[F1score_train.index(max(F1score_train))]}')
        if stop_flag == 'F_train':
            print(f'训练集F:{f_train},第{best_epoch}次迭代取得最高F{max(Fscore_train)}, '
                  f'对应的测试集ACC:{ACC_test[Fscore_train.index(max(Fscore_train))]}, '
                  f'F1：{F1score_test[Fscore_train.index(max(Fscore_train))]}, '
                  f'F：{Fscore_test[Fscore_train.index(max(Fscore_train))]}')

    return (test_ls, ACC_test, Precision_test, Recall_test, F1score_test, Fscore_test,
            train_ls, ACC_train, Precision_train, Recall_train, F1score_train, Fscore_train,
            true_labels, pred_labels)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ICC_rms')
    parser.add_argument('--model', type=str, default='MLP')
    args = parser.parse_args()
    seed_everything(0)
    dataset_name = args.dataset
    model_name = args.model
    label_smooth = 0.1
    num_epochs = 50
    learning_rate = 5e-4
    weight_decay = 1e-3
    trip_lambda = 0.01
    batch_size = 256
    output_size = 2
    gamma = 2
    loss_name = 'CEWithSmooth'
    K = 5
    state_size = 256
    median_size = 512
    stop_flag = 'F_train'

    mat, labels = read_mouse(dataset_name)
    unique_chars, integer_labels = np.unique(labels, return_inverse=True)
    # mat_mean = mat.mean(axis=0)
    # mat = np.array(mat >= mat_mean, dtype='float')
    # mat = mat / np.max(mat, axis=1)[:,None]

    X_train, X_test, y_train, y_test = train_test_split(mat, integer_labels, test_size = 0.2, random_state=19)
    print(X_train.shape)
    print(X_test.shape)
    train_data = torch.tensor(
        X_train, dtype=torch.float32
    ).to(device)

    train_label = torch.tensor(
        y_train
    ).to(device)

    test_data = torch.tensor(
        X_test, dtype=torch.float32
    ).to(device)

    test_label = torch.tensor(
        y_test
    ).to(device)
    in_features = X_train.shape[1]
    net = get_net(in_features, state_size, median_size, 2, model_name)
    # net = nn.DataParallel(net)

    i = 0
    class_weights = [1,1]
    (test_loss, score, precision, recall, F1, Fsc,
     train_loss, score_train, precision_train, recall_train, F1_train, Fsc_train,
     true_labels, pred_labels) = train_model(net, train_data, train_label, test_data, test_label,
                                             num_epochs, learning_rate, weight_decay, batch_size, output_size,
                                             label_smooth, i, gamma, loss_name, class_weights, dataset_name, trip_lambda)