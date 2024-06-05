import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from func import predict


def plot_confusion_matrix(true_labels, pred_labels, class_names):
    """
    绘制混淆矩阵

    :param true_labels: 真实标签列表
    :param pred_labels: 预测标签列表
    :param class_names: 类别名称列表
    """
    cm = confusion_matrix(true_labels, pred_labels, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
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
                start += sample_len
                pred_labels.append(pred)
                true_labels.append(idx)

        acc = np.sum(true_labels == pred_labels) / len(true_labels)
        acc_lst.append(acc)
        print(f"iteration {iteration} acc: {acc}")

    plt.plot(iterations, acc_lst)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy-Iteration Curve')
    plt.show()
    
