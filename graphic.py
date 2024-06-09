import os

os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from func import predict, train
import itertools


def calculate_accuracy(models, test_kmeans, test_lens, test_labels):
    pred_labels = []
    true_labels = []
    for idx, (samples, lens) in enumerate(zip(test_kmeans, test_lens)):
        start = 0
        for sample_len in lens:
            pred, scores = predict(models, samples[start:start + sample_len])
            start += sample_len
            pred_labels.append(pred)
            true_labels.append(idx)
    print(accuracy_score(true_labels, pred_labels))
    return accuracy_score(true_labels, pred_labels)


def plot_confusion_matrix(models, test_data, test_lens, class_names):
    """
    绘制混淆矩阵

    :param models: 模型列表
    :param test_data: 测试数据
    :param test_lens: 测试数据的长度
    :param class_names: 类别名称列表
    """
    pred_labels = []
    true_labels = []
    for idx, (samples, lens) in enumerate(zip(test_data, test_lens)):
        start = 0
        for sample_len in lens:
            pred, scores = predict(models, samples[start:start + sample_len])
            start += sample_len
            pred_labels.append(pred)
            true_labels.append(idx)
    print(accuracy_score(true_labels, pred_labels))
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true=true_labels, y_pred=pred_labels, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion matrix')
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(class_names.__len__()), class_names)
    plt.xticks(range(class_names.__len__()), class_names, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(class_names.__len__()):
        for j in range(class_names.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)
    plt.show()


def plot_accuracy_iteration(models_lst, test_data, test_lens, iterations):
    pred_labels = []
    true_labels = []
    acc_lst = []

    for (models, iteration) in zip(models_lst, iterations):
        for idx, (samples, lens) in enumerate(zip(test_data, test_lens)):
            start = 0
            for sample_len in lens:
                pred, scores = predict(models, samples[start:start + sample_len])
                # print(pred, idx)
                start += sample_len
                pred_labels.append(pred)
                true_labels.append(idx)

        acc = np.sum([true_label == pred_label for (true_label, pred_label) in zip(true_labels, pred_labels)]) / len(
            true_labels)
        acc_lst.append(acc)
        print(f"iteration {iteration} acc: {acc}")

    plt.plot(iterations, acc_lst)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy-Iteration Curve')
    plt.show()


def plot_accuracy_n_components(models_lst, test_data, test_lens, n_components):
    pred_labels = []
    true_labels = []
    acc_lst = []

    for (models, n_component) in zip(models_lst, n_components):
        for idx, (samples, lens) in enumerate(zip(test_data, test_lens)):
            start = 0
            for sample_len in lens:
                pred, scores = predict(models, samples[start:start + sample_len])
                # print(pred, idx)
                start += sample_len
                pred_labels.append(pred)
                true_labels.append(idx)
        # print(len(true_labels))
        acc = np.sum([true_label == pred_label for (true_label, pred_label) in zip(true_labels, pred_labels)]) / len(
            true_labels)
        acc_lst.append(acc)
        print(f"n_component {n_component} acc: {acc}")

    plt.plot(n_components, acc_lst)
    plt.xlabel('n_components')
    plt.ylabel('Accuracy')
    plt.title('Accuracy-n_components Curve')
    plt.show()


def plot_accuracy_k(models, train_data_lst, test_data_lst, train_lens_lst, test_lens_lst, ks):
    pred_labels = []
    true_labels = []
    acc_lst = []

    for (train_data, test_data, train_lens, test_lens, k) in zip(train_data_lst, test_data_lst, train_lens_lst,
                                                                 test_lens_lst, ks):
        train(models, train_data, ['a', 'e', 'i', 'o', 'u'], train_lens)

        for idx, (samples, lens) in enumerate(zip(test_data, test_lens)):
            start = 0
            for sample_len in lens:
                pred, scores = predict(models, samples[start:start + sample_len])
                # print(pred, idx)
                start += sample_len
                pred_labels.append(pred)
                true_labels.append(idx)
        # print(len(true_labels))
        acc = np.sum([true_label == pred_label for (true_label, pred_label) in zip(true_labels, pred_labels)]) / len(
            true_labels)
        acc_lst.append(acc)
        print(f"k {k} acc: {acc}")

    plt.plot(ks, acc_lst)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('Accuracy-K Curve')
    plt.show()


def plot_accuracy_k_n_component(k_li, n_li):
    from dataset import load_letter_dataset
    from Model import HMModel
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(len(k_li), 1, figsize=(8, 6 * len(k_li)))

    for i, k in enumerate(k_li):
        pred_labels = []
        true_labels = []
        acc_lst = []

        (observations, kmeans_data, labels, lens), (test_data, test_kmeans, test_labels, test_lens) = (
            load_letter_dataset('data_figure', n_clusters=k, dtw=True))
        # print(test_lens)
        for n in n_li:
            if n >= k:
                acc_lst.append(0)
                continue

            models = [HMModel(n_components=n, n_iter=200) for _ in ['a', 'e', 'i', 'o', 'u']]
            # print(kmeans_data, lens)
            train(models, kmeans_data, labels, lens)

            for idx, (samples, test_len) in enumerate(zip(test_kmeans, test_lens)):
                start = 0
                for sample_len in test_len:
                    # print(sample_len)
                    pred, scores = predict(models, samples[start:start + sample_len])
                    start += sample_len
                    pred_labels.append(pred)
                    true_labels.append(idx)

            acc = np.sum(
                [true_label == pred_label for (true_label, pred_label) in zip(true_labels, pred_labels)]) / len(
                true_labels)
            acc_lst.append(acc)
            print(f"k={k};n={n} acc: {acc}")
        # print(n_li, acc_lst)
        axs[i].plot(n_li, acc_lst)
        axs[i].set_xlabel('n_components')
        axs[i].set_ylabel('Accuracy')
        axs[i].set_title(f'Accuracy-n_components Curve (k={k})')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    range_k = list(range(5, 11))
    range_component = list(range(2, 13))
    plot_accuracy_k_n_component(range_k, range_component)
