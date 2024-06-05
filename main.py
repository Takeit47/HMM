import os
# os.environ["OMP_NUM_THREADS"] = "4"
from func import train
from dataset import load_figure_dataset
from Model import HMModel
import numpy as np
import graphic

if __name__ == '__main__':
    # (observations, kmeans_data, labels, lens), (test_data, test_kmeans, test_labels, test_lens) = (
    #     load_figure_dataset('data_figure', n_clusters=16))
    #
    # model_lst = []
    # iterations = list(range(100, 2100, 100))
    # for iteration in iterations:
    #     models = [HMModel(n_components=4, n_iter=iteration) for _ in ['a', 'e', 'i', 'o', 'u']]
    #     train(models, observations, labels, lens)
    #     model_lst.append(models)
    #
    # graphic.plot_accuracy_iteration(model_lst, test_data, test_lens, iterations)
    #
    # n_components = list(range(2, 13))
    # for n_component in n_components:
    #     models = [HMModel(n_components=n_component, n_iter=2000) for _ in ['a', 'e', 'i', 'o', 'u']]
    #     train(models, observations, labels, lens)
    #     model_lst.append(models)
    #
    # graphic.plot_accuracy_n_components(model_lst, test_data, test_lens, n_components)
    # # graphic.plot_confusion_matrix(true_lst, predict_lst, test_labels)
    #
    ks = list(range(5, 26))
    train_data_lst, test_data_lst = [], []
    train_lens_lst, test_lens_lst = [], []
    models = [HMModel(n_components=2, n_iter=100) for _ in ['a', 'e', 'i', 'o', 'u']]
    for k in ks:
        (observations, kmeans_data, labels, lens), (test_data, test_kmeans, test_labels, test_lens) = (load_figure_dataset('data_figure', n_clusters=k))
        train_data_lst.append(kmeans_data)
        test_data_lst.append(test_kmeans)
        train_lens_lst.append(lens)
        test_lens_lst.append(test_lens)
    graphic.plot_accuracy_k(models, train_data_lst, test_data_lst, train_lens_lst, test_lens_lst, ks)



