import csv
import glob
import os
import pandas as pd
import numpy as np
from collections import Counter

def read_KidneyDisease(file_dir):
    file = f'train-ranksum_0.05_area.csv'
    files = pd.read_csv(os.path.join(file_dir, file))
    df_label = files.iloc[:,0]
    df_intensity = files.iloc[:,1:]
    labels_train = df_label.values.tolist()
    labels_train = np.array(labels_train).T
    mat_train = df_intensity.values.tolist()
    mat_train = np.array(mat_train)
    mat_train = mat_train / np.max(mat_train, axis=1)[:, None]
    """
    file = f'train-ranksum_0.01_area.csv'
    files = pd.read_csv(os.path.join(file_dir, file))
    df_intensity = files.iloc[:,1:]
    mat_train2 = df_intensity.values.tolist()
    mat_train2 = np.array(mat_train2)
    mat_train2 = mat_train2 / np.max(mat_train2, axis=1)[:, None]

    mat_train = np.concatenate((mat_train, mat_train2), axis=1)
    """

    file = f'test-ranksum_0.05_area.csv'
    files = pd.read_csv(os.path.join(file_dir, file))
    df_label = files.iloc[:,0]
    df_intensity = files.iloc[:,1:]
    labels_test = df_label.values.tolist()
    labels_test = np.array(labels_test).T
    mat_test = df_intensity.values.tolist()
    mat_test = np.array(mat_test)
    mat_test = mat_test / np.max(mat_test, axis=1)[:, None]
    """
    file = f'test-ranksum_0.01_area.csv'
    files = pd.read_csv(os.path.join(file_dir, file))
    df_intensity = files.iloc[:,1:]
    mat_test2 = df_intensity.values.tolist()
    mat_test2 = np.array(mat_test2)
    mat_test2 = mat_test2 / np.max(mat_test2, axis=1)[:, None]

    mat_test = np.concatenate((mat_test, mat_test2), axis=1)

    """
    return mat_train, labels_train, mat_test, labels_test

if __name__ == '__main__':
    file_dir = './' # csv上一级目录
    # batch_1, labels_1, batch_2, labels_2, batch_3, labels_3 = read_MI_batch(file_dir)