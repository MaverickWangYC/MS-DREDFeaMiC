import csv
import glob
import os
import pandas as pd
import numpy as np
from collections import Counter

def read_tomato(file_dir):
    file = f'train.csv'
    files = pd.read_csv(os.path.join(file_dir, file))
    df_label = files.iloc[:,3]
    df_intensity = files.iloc[:,4:]
    labels_train = df_label.values.tolist()
    mat_train = df_intensity.values.tolist()

    file = f'test.csv'
    files = pd.read_csv(os.path.join(file_dir, file))
    df_label = files.iloc[:,3]
    df_intensity = files.iloc[:,4:]
    labels_test = df_label.values.tolist()
    mat_test = df_intensity.values.tolist()

    arr_train = np.array(mat_train)
    arr_test = np.array(mat_test)
    arr_label_train = np.array(labels_train).T
    arr_label_test = np.array(labels_test).T

    arr_train = np.where(arr_train > 1e4, arr_train, 0)
    arr_test = np.where(arr_test > 1e4, arr_test, 0)

    return arr_train, arr_label_train, arr_test, arr_label_test

def read_tomato_batch(file_dir):
    file = f'train.csv'
    files = pd.read_csv(os.path.join(file_dir, file))
    df_label = files.iloc[:,3]
    df_intensity = files.iloc[:,4:]
    labels_train = df_label.values.tolist()
    mat_train = df_intensity.values.tolist()
    mz_range = files.columns[4:]

    file = f'test.csv'
    files = pd.read_csv(os.path.join(file_dir, file))
    df_label = files.iloc[:,3]
    df_intensity = files.iloc[:,4:]
    labels_test = df_label.values.tolist()
    mat_test = df_intensity.values.tolist()
    return np.array(mat_train), np.array(labels_train).T, np.array(mat_test), np.array(labels_test).T, mz_range

if __name__ == '__main__':
    file_dir = './' # csv上一级目录
    # batch_1, labels_1, batch_2, labels_2, batch_3, labels_3 = read_MI_batch(file_dir)