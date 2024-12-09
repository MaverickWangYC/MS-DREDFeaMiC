import csv
import glob
import os
import pandas as pd
import numpy as np
from collections import Counter
def read_lowResolution_8_7_02(file_dir):
    mat_list = []
    labels = []
    nums = []
    for fn in glob.glob(os.path.join(file_dir, '*')):
        label = os.path.basename(fn)[:]
        # mat, id_part = _read_h5_file(fn)
        file_dir_label = file_dir+'/'+label+'/'
        files = os.listdir(file_dir_label)

        for e in files[0:]:
            df1 = pd.read_csv(os.path.join(file_dir_label, e)).iloc[:,1]
            labels.append(label)
            mat_list.append(df1.values.tolist())

    return np.array(mat_list), np.array(labels).T


if __name__ == '__main__':
    file_dir = '../data/OC-8-7-02/'
    mat, labels = read_lowResolution_8_7_02(file_dir)
    labels = list(labels)
    print(labels)
    label_sets = list(set(labels))
    label_sets.sort(key=labels.index)
    count = Counter(labels)

    nums = []
    for i in range(len(label_sets)):
        nums.append(count[label_sets[i]])
    print(nums)