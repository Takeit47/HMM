import os

os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
from hmmlearn import hmm

# 定义模型参数
n_components = 16  # 隐状态的数量
n_iter = 1000  # 训练的迭代次数


# # 创建一个高斯HMM实例
# model_A = hmm.GaussianHMM(n_components=n_components, covariance_type='full', n_iter=n_iter)
# model_E = hmm.GaussianHMM(n_components=n_components, covariance_type='full', n_iter=n_iter)


class HMModel:
    def __init__(self, n_components=n_components, n_iter=n_iter):
        self.n_components = n_components
        self.n_iter = n_iter
        self.model = hmm.CategoricalHMM(n_components=n_components, n_iter=n_iter, n_features=200)

    def train(self, observation, observation_length):
        self.model.fit(np.array(observation, dtype=int), observation_length)

    def predict(self, observation):
        # print(observation)
        score = self.model.score(np.array(observation, dtype=int))
        return score


class GaussianHMM:
    def __init__(self, n_components=n_components, n_iter=n_iter):
        self.n_components = n_components
        self.n_iter = n_iter
        self.model = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter, covariance_type='full')

    def Laplacian(self):
        pass

    def train(self, observation, observation_length):
        self.model.fit(observation, observation_length)

    def predict(self, observation):
        # print(observation)
        score = self.model.score(observation)
        return score


if __name__ == '__main__':
    import dataset_kth
    from func import *
    from dataset_kth import CATEGORY_INDEX

    models = [GaussianHMM(2, 200) for i in range(6)]
    for split_type, category_data in dataset_kth.items():
        if split_type == 'Training':
            datas = [each_class['instances'] for each_class in category_data]
            lens = [each_class['lens'] for each_class in category_data]
            train(models, category, CATEGORY_INDEX.keys(), lengths)
        if split_type == 'Test':
            pred_labels = []
            true_labels = []
            category, lengths = category_data.items()
            for idx, (samples, test_len) in enumerate(zip(category, lengths)):
                start = 0
                for sample_len in test_len:
                    # print(sample_len)
                    pred, scores = predict(models, samples[start:start + sample_len])
                    start += sample_len
                    pred_labels.append(pred)
                    true_labels.append(idx)
            acc = np.sum(true_labels == pred_labels) / len(true_labels)
            print(f"acc: {acc}")
