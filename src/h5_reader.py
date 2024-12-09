import glob
import os.path

import h5py
import numpy as np


def _read_h5_file(file_path: str) -> np.ndarray:
    """ 读取 h5 文件，返回质谱数据矩阵，每行表示一副质谱图，质谱图的数量由行数决定 """
    with h5py.File(file_path, 'r') as f:
        scans = f['scans']
        scan_type = scans['type'][()]
        scan_id = scans['sample_id_array'][()]

        #print(scan_id)
        if isinstance(scan_type, (bytes, bytearray)):
            scan_type = scan_type.decode('utf-8')
        is_matrix2d = True if scan_type == '[x][[y]]' else False
        if not is_matrix2d:
            raise RuntimeError('只能读取2维举证数据')
        return scans['inte_matrix2d'][:, :].T, scan_id


def read_h5(file_dir: str):
    """ 从目录中读取 h5 文件，文件格式由 pyxjdata 输出，并经过对齐的数据

    :return 返回 Tuple(np.array(N), np.array(M, N));
            其中, 第一个参数为标签列表（Nx1）, 第2个参数为质谱数据矩阵（NxM）, M为质量数个数，N为质谱图个数
    """
    mat_list = []
    label_list = []
    ids_list = np.array(())
    num_id = []
    last_max = 0
    nums=[] #表示每种类别的ID数目
    for fn in glob.glob(os.path.join(file_dir, '*.h5')):
        print(fn)
        label = os.path.basename(fn)[:-3]
        mat, id_part = _read_h5_file(fn)
        if len(ids_list) == 0:
            #print(id_part)
            ids_list = np.append(ids_list,id_part)
        else:
            id_last = max(ids_list)
            ids_list = np.append(ids_list,id_part+id_last)
        label_list.extend([label] * len(mat))
        mat_list.append(mat)
        num_id.append(len(ids_list))

    for i in range(1,len(ids_list)):
        if ids_list[i]-ids_list[i-1] > 1:
            temp_i = ids_list[i]
            for j in range(i,len(ids_list)):
                if ids_list[j] == temp_i:
                    ids_list[j] = ids_list[i-1]+1
                else:
                    break
    for i in range(len(num_id)):
        nums.append(ids_list[num_id[i]-1]-last_max)
        last_max = ids_list[num_id[i]-1]
    #print(nums)
    #print(np.array(label_list).T)
    return np.array(label_list).T, np.concatenate(mat_list), ids_list, nums


def test():
    label_list, d, _, nums = read_h5('../data/ColonCancer/bin_0.1')
    print(d.shape)
    print(label_list)
    print(nums)

if __name__ == '__main__':
    test()
