import csv
import glob
import os
import pandas as pd
import numpy as np
from collections import Counter

def read_CC_batch(file_dir):
    for i in range(1,3):
        locals()[f'file_batch_{i}'] = f'batch_{i}.csv'
        locals()[f'files_{i}'] = pd.read_csv(os.path.join(file_dir, locals()[f'file_batch_{i}']))
        locals()[f'df_label_{i}'] = locals()[f'files_{i}'].iloc[:,0]
        locals()[f'df_intensity_{i}'] = locals()[f'files_{i}'].iloc[:,2:]
        locals()[f'labels_batch_{i}'] = locals()[f'df_label_{i}'].values.tolist()
        locals()[f'mat_list_batch_{i}'] = locals()[f'df_intensity_{i}'].values.tolist()


    return np.array(locals()[f'mat_list_batch_{1}']), np.array(locals()[f'df_label_{1}']).T,\
        np.array(locals()[f'mat_list_batch_{2}']), np.array(locals()[f'df_label_{2}']).T


if __name__ == '__main__':
    file_dir = '../data/ColonCancer/bin_0.1' # csv上一级目录
    batch_1, labels_1, batch_2, labels_2 = read_CC_batch(file_dir)