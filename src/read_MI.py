import csv
import glob
import os
import pandas as pd
import numpy as np
from collections import Counter

def read_MI_batch(file_dir):
    for i in range(1,4):
        locals()[f'file_batch_{i}'] = f'batch_{i}_noboard.csv'
        locals()[f'files_{i}'] = pd.read_csv(os.path.join(file_dir, locals()[f'file_batch_{i}']))
        locals()[f'df_label_{i}'] = locals()[f'files_{i}'].iloc[:,0]
        locals()[f'df_intensity_{i}'] = locals()[f'files_{i}'].iloc[:,1:]
        locals()[f'labels_batch_{i}'] = locals()[f'df_label_{i}'].values.tolist()
        locals()[f'mat_list_batch_{i}'] = locals()[f'df_intensity_{i}'].values.tolist()


    return np.array(locals()[f'mat_list_batch_{1}']), np.array(locals()[f'df_label_{1}']).T,\
        np.array(locals()[f'mat_list_batch_{2}']), np.array(locals()[f'df_label_{2}']).T,\
        np.array(locals()[f'mat_list_batch_{3}']), np.array(locals()[f'df_label_{3}']).T

def read_MI_batch_after_Ka(file_dir):
    file_dir = file_dir + '/'
    """
    types = 'ranksum'
    if types == 'ka2' or 'ranksum' or 'ranksum_ka2' or 'cross' or 'union':
        prx = 'none'
    else:
        prx = 'mean'
    """
    types = 'chi2'
    prx = 'none'
    end_prx = '_chi2'
    thresh = 150
    for i in range(1,4):
        locals()[f'file_batch_{i}'] = f'batch_{i}-{types}_{thresh}.csv'
        locals()[f'files_{i}'] = pd.read_csv(os.path.join(file_dir+'train12_'+prx+end_prx, locals()[f'file_batch_{i}']))
        locals()[f'df_label_{i}'] = locals()[f'files_{i}'].iloc[:,0]
        locals()[f'df_intensity_{i}'] = locals()[f'files_{i}'].iloc[:,1:]
        locals()[f'labels_batch_{i}'] = locals()[f'df_label_{i}'].values.tolist()
        locals()[f'mat_list_batch_{i}'] = locals()[f'df_intensity_{i}'].values.tolist()
    for i in range(4,7):
        locals()[f'file_batch_{i}'] = f'batch_{i-3}-{types}_{thresh}.csv'
        locals()[f'files_{i}'] = pd.read_csv(os.path.join(file_dir+'train13_'+prx+end_prx, locals()[f'file_batch_{i}']))
        locals()[f'df_label_{i}'] = locals()[f'files_{i}'].iloc[:,0]
        locals()[f'df_intensity_{i}'] = locals()[f'files_{i}'].iloc[:,1:]
        locals()[f'labels_batch_{i}'] = locals()[f'df_label_{i}'].values.tolist()
        locals()[f'mat_list_batch_{i}'] = locals()[f'df_intensity_{i}'].values.tolist()
    for i in range(7, 10):
        locals()[f'file_batch_{i}'] = f'batch_{i - 6}-{types}_{thresh}.csv'
        locals()[f'files_{i}'] = pd.read_csv(os.path.join(file_dir + 'train23_'+prx+end_prx, locals()[f'file_batch_{i}']))
        locals()[f'df_label_{i}'] = locals()[f'files_{i}'].iloc[:, 0]
        locals()[f'df_intensity_{i}'] = locals()[f'files_{i}'].iloc[:, 1:]
        locals()[f'labels_batch_{i}'] = locals()[f'df_label_{i}'].values.tolist()
        locals()[f'mat_list_batch_{i}'] = locals()[f'df_intensity_{i}'].values.tolist()

    return np.array(locals()[f'mat_list_batch_{1}']), np.array(locals()[f'df_label_{1}']).T,\
        np.array(locals()[f'mat_list_batch_{2}']), np.array(locals()[f'df_label_{2}']).T,\
        np.array(locals()[f'mat_list_batch_{3}']), np.array(locals()[f'df_label_{3}']).T,\
        np.array(locals()[f'mat_list_batch_{4}']), np.array(locals()[f'df_label_{4}']).T, \
        np.array(locals()[f'mat_list_batch_{5}']), np.array(locals()[f'df_label_{5}']).T, \
        np.array(locals()[f'mat_list_batch_{6}']), np.array(locals()[f'df_label_{6}']).T, \
        np.array(locals()[f'mat_list_batch_{7}']), np.array(locals()[f'df_label_{7}']).T, \
        np.array(locals()[f'mat_list_batch_{8}']), np.array(locals()[f'df_label_{8}']).T, \
        np.array(locals()[f'mat_list_batch_{9}']), np.array(locals()[f'df_label_{9}']).T


if __name__ == '__main__':
    file_dir = '../data/MI' # csv上一级目录
    batch_1_1, labels_1_1, batch_2_1, labels_2_1, batch_3_1, labels_3_1,\
        batch_1_2, labels_1_2, batch_2_2, labels_2_2, batch_3_2, labels_3_2,\
        batch_1_3, labels_1_3, batch_2_3, labels_2_3, batch_3_3, labels_3_3 = read_MI_batch_after_Ka(file_dir)