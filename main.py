import os
# os.environ["OMP_NUM_THREADS"] = "4"
from func import train
from dataset import load_figure_dataset, load_letter_dataset
from Model import HMModel
import numpy as np
import graphic

if __name__ == '__main__':
    (observations, kmeans_data, labels, lens), (test_data, test_kmeans, test_labels, test_lens) = (
        load_figure_dataset('data_figure', n_clusters=5, dtw=True))
    models = [HMModel(n_components=5, n_iter=2000) for _ in ['a', 'e', 'i', 'o', 'u']]
    train(models, kmeans_data, labels, lens)
    graphic.plot_confusion_matrix(models, test_kmeans, test_lens, test_labels)

    model_lst = []
    iterations = list(range(100, 2100, 100))
    for iteration in iterations:
        models = [HMModel(n_components=3, n_iter=iteration) for _ in ['a', 'e', 'i', 'o', 'u']]
        train(models, kmeans_data, labels, lens)
        model_lst.append(models)

    graphic.plot_accuracy_iteration(model_lst, test_kmeans, test_lens, iterations)

    n_components = list(range(2, 13))
    for n_component in n_components:
        models = [HMModel(n_components=n_component, n_iter=2000) for _ in ['a', 'e', 'i', 'o', 'u']]
        train(models, kmeans_data, labels, lens)
        model_lst.append(models)

    graphic.plot_accuracy_n_components(model_lst, test_kmeans, test_lens, n_components)


    ks = list(range(5, 11))
    train_data_lst, test_data_lst = [], []
    train_lens_lst, test_lens_lst = [], []
    models = [HMModel(n_components=2, n_iter=200) for _ in ['a', 'e', 'i', 'o', 'u']]
    for k in ks:
        (observations, kmeans_data, labels, lens), (test_data, test_kmeans, test_labels, test_lens) = (load_figure_dataset('data_figure', n_clusters=k))
        train_data_lst.append(kmeans_data)
        test_data_lst.append(test_kmeans)
        train_lens_lst.append(lens)
        test_lens_lst.append(test_lens)
    graphic.plot_accuracy_k(models, train_data_lst, test_data_lst, train_lens_lst, test_lens_lst, ks)



