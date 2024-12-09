from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from h5_reader import read_h5


def mlp():
    file_name = r'../data/t1_2/bin_0.1'
    labels, mat = read_h5(file_name)
    test_size = 0.2

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
    train_idx_arr, test_idx_arr = next(sss.split(mat, labels))
    print(f'训练集数据个数：{len(train_idx_arr)}, 测试集数据个数：{len(test_idx_arr)}')

    train_data = mat[train_idx_arr, :]
    train_label = labels[train_idx_arr]

    mlp_classifier = MLPClassifier(max_iter=1000)
    mlp_classifier.fit(train_data, train_label)

    test_data = mat[test_idx_arr, :]
    test_label_true = labels[test_idx_arr]
    test_label_predict = mlp_classifier.predict(test_data)

    acc_scr = accuracy_score(test_label_true, test_label_predict)
    print(f'准确度得分：{acc_scr}')


if __name__ == '__main__':
    mlp()

