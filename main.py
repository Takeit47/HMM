import os

os.environ["OMP_NUM_THREADS"] = "1"

from func import train
from dataset import load_letter_dataset, load_letter_dataset
from Model import HMModel
import numpy as np
import graphic

if __name__ == '__main__':
    (observations, kmeans_data, labels, lens), (test_data, test_kmeans, test_labels, test_lens) = (
        load_letter_dataset('data_figure', n_clusters=20, dtw=True, cluster='kmeans'))
    models = [HMModel(n_components=20, n_iter=2000) for _ in ['a', 'e', 'i', 'o', 'u']]
    train(models, kmeans_data, labels, lens)
    graphic.plot_confusion_matrix(models, test_kmeans, test_lens, test_labels)
################################################################

    # model_lst = []
    # iterations = list(range(100, 2100, 100))
    # for iteration in iterations:
    #     models = [HMModel(n_components=7, n_iter=iteration) for _ in ['a', 'e', 'i', 'o', 'u']]
    #     train(models, kmeans_data, labels, lens)
    #     model_lst.append(models)
    #
    # graphic.plot_accuracy_iteration(model_lst, test_kmeans, test_lens, iterations)

################################################################

    # n_components = list(range(5, 20))
    # for n_component in n_components:
    #     models = [HMModel(n_components=n_component, n_iter=2000) for _ in ['a', 'e', 'i', 'o', 'u']]
    #     train(models, kmeans_data, labels, lens)
    #     model_lst.append(models)
    #
    # graphic.plot_accuracy_n_components(model_lst, test_kmeans, test_lens, n_components)

################################################################

    # ks = list(range(5, 20))
    # train_data_lst, test_data_lst = [], []
    # train_lens_lst, test_lens_lst = [], []
    # models = [HMModel(n_components=15, n_iter=2000) for _ in ['a', 'e', 'i', 'o', 'u']]
    # for k in ks:
    #     (observations, kmeans_data, labels, lens), (test_data, test_kmeans, test_labels, test_lens) = \
    #         (load_letter_dataset('data_figure', n_clusters=k, dtw=True))
    #     train_data_lst.append(kmeans_data)
    #     test_data_lst.append(test_kmeans)
    #     train_lens_lst.append(lens)
    #     test_lens_lst.append(test_lens)
    # graphic.plot_accuracy_k(models, train_data_lst, test_data_lst,
    #                         train_lens_lst, test_lens_lst, ks)

################################################################

    def show_plot_n_k_acc():
        import matplotlib.pyplot as plt
        n_values = [2, 5, 7, 10, 13, 15, 17, 20]

        # List to store accuracy rates for each n
        accuracy_rates = []

        # Iterate over n values
        for n in n_values:
            # Calculate accuracy rates for different values of k
            ks = list(range(5, 31))
            accuracy_lst = []
            for k in ks:
                (observations, kmeans_data, labels, lens), (test_data, test_kmeans, test_labels, test_lens) = (
                    load_letter_dataset('data_figure', n_clusters=k, dtw=True))
                models = [HMModel(n_components=n, n_iter=2000) for _ in ['a', 'e', 'i', 'o', 'u']]
                train(models, kmeans_data, labels, lens)

                accuracy = graphic.calculate_accuracy(models, test_kmeans, test_lens, test_labels)
                accuracy_lst.append(accuracy)

            # Append accuracy rates to the list
            accuracy_rates.append(accuracy_lst)

        # Plot k-accuracy rate curve
        for i, n in enumerate(n_values):
            plt.plot(ks, accuracy_rates[i], label=f'n={n}')

        plt.xlabel('k')
        plt.ylabel('Accuracy Rate')
        plt.title('k-Accuracy Rate Curve')
        plt.legend()
        plt.show()

    show_plot_n_k_acc()

