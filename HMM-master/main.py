import os
# os.environ["OMP_NUM_THREADS"] = "4"

from dataset import load_figure_dataset
from Model import HMModel
import numpy as np
import graphic


def train(models, observations, labels, lens):
    for (model, observation, label, len) in zip(models, observations, labels, lens):
        model.Laplacian()
        model.train(observation, len)
        print(f"model {label} train finished.")
    return models


def predict(models, sample):
    scores = [model.predict(sample) for model in models]
    return np.argmax(scores), scores


if __name__ == '__main__':
    (observations, kmeans_data, labels, lens), (test_data, test_kmeans, test_labels, test_lens) = (
        load_figure_dataset('data_figure', n_clusters=16))

    models = [HMModel(n_components=8, n_iter=1000) for _ in ['a', 'e', 'i', 'o', 'u']]
    train(models, kmeans_data, labels, lens)
    predict_lst = []
    true_lst = []

    err = 0
    for i, (samples, lens) in enumerate(zip(test_kmeans, test_lens)):
        start = 0
        for sample_len in lens:
            pred, scores = predict(models, samples[start:start + sample_len])
            start += sample_len
            if pred != i:
                err += 1
            predict_lst.append(pred)
            true_lst.append(i)
            print(pred, i)
    print(err)
    err /= np.sum([len(samples) for samples in test_lens])
    print(err)
    graphic.plot_confusion_matrix(true_lst, predict_lst, test_labels)
