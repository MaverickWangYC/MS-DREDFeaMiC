import csv
import glob
import os
import pandas as pd
import numpy as np
from collections import Counter

def read_mouse(dataset_name):
    file_dir = '../data/SCCML_data/' + dataset_name + '.csv'
    files = pd.read_csv(file_dir)
    df_labels = files.iloc[:, -1]
    df_intensity = files.iloc[:, 0:-1]
    labels = df_labels.values.tolist()
    mat = df_intensity.values.tolist()

    mat = np.array(mat)
    labels = np.array(labels).T
    print(mat.shape)
    return  mat, labels

if __name__ == '__main__':
    dataset_name = 'HIP_CER'
    mat, labels = read_mouse(dataset_name)