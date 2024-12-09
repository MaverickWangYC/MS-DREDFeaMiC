from sklearn import manifold
from h5_reader import read_h5
from matplotlib.pylab import *


def tsne():
    file_name = r'../data/t1_2/bin_0.1'
    labels, mat = read_h5(file_name)
    n_components = 3
    init = 'random'
    perplexity = 7
    random_state = 5
    metric = 'euclidean'
    method = 'barnes_hut'
    learning_rate = 1
    n_iter = 5000
    tsne_proc = manifold.TSNE(n_components=n_components, init=init,
                              perplexity=perplexity, random_state=random_state,
                              metric=metric, learning_rate=learning_rate, method=method,
                              n_iter=n_iter)
    reduced_x = tsne_proc.fit_transform(mat)

    colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    unique_label = np.unique(labels)
    index_arr = np.arange(len(labels))
    for label_idx, label in enumerate(unique_label):
        data_idx = index_arr[labels == label]
        scatter(reduced_x[data_idx, 0], reduced_x[data_idx, 1], c=colors[label_idx], label=label)
    legend()
    show()


if __name__ == '__main__':
    tsne()
