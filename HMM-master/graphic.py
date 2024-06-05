import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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