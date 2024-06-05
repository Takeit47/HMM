import os
# os.environ["OMP_NUM_THREADS"] = "4"
from func import train
from dataset import load_figure_dataset
from Model import HMModel
import numpy as np
import graphic

if __name__ == '__main__':
    (observations, kmeans_data, labels, lens), (test_data, test_kmeans, test_labels, test_lens) = (
        load_figure_dataset('data_figure', n_clusters=16))

    model_lst = []
    iterations = list(range(100, 2100, 100))
    for iteration in iterations:
        models = [HMModel(n_components=8, n_iter=iteration) for _ in ['a', 'e', 'i', 'o', 'u']]
        train(models, kmeans_data, labels, lens)
        model_lst.append(models)

    graphic.plot_accuracy_iteration(model_lst, test_kmeans, test_lens, iterations)

    # graphic.plot_confusion_matrix(true_lst, predict_lst, test_labels)
