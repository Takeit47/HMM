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
        self.model = hmm.GaussianHMM(n_components=n_components, covariance_type='full', n_iter=n_iter)

    def Laplacian(self):
        self.model.startprob_ = np.ones(self.n_components) / self.n_components
        self.model.transmat_ = np.ones((self.n_components, self.n_components)) / self.n_components


    def train(self, observation, observation_length):
        self.model.fit(observation, observation_length)

    def predict(self, observation):
        score = self.model.score(observation)
        return score


if __name__ == '__main__':
    model_A = hmm.GaussianHMM(n_components=n_components, covariance_type='full', n_iter=n_iter)
    model_E = hmm.GaussianHMM(n_components=n_components, covariance_type='full', n_iter=n_iter)

    import dataset
    # 生成示例数据
    # 这里我们使用简单的整数观测值，但你可以根据需要使用自己的数据
    src_observations, observations, labels, lens = dataset.load_figure_dataset('data_figure/', n_clusters=32)

    observation_a = np.reshape(observations[0], (-1, 1))
    observation_e = np.reshape(observations[1], (-1, 1))
    print(list(value for value in observation_a))
    len_a = lens[0]
    len_e = lens[1]
    print(observation_a.shape, observation_e.shape, len_a, len_e)
    # 训练模型
    model_A.fit(observation_a, lengths=len_a)
    print("model_A train finished.")
    model_E.fit(observation_e, lengths=len_e)
    print("model_E train finished.")
    # # 预测观测序列的隐状态
    # logprob, state_sequence = model.decode(observation, algorithm="viterbi")
    # print("最可能的状态序列:", state_sequence)
    sample = np.reshape(observation_e[0:87], (-1, 1))
    # print(sample)
    # 计算观测序列的概率
    score_a = model_A.score(sample)
    score_e = model_E.score(sample)
    print("观测序列的概率:", np.exp(score_a), np.exp(score_e))

    # # 输出模型的参数
    # print("初始概率分布:", model.startprob_)
    # print("转移概率矩阵:", model.transmat_)
    # print("均值:", model.means_)
    # print("协方差:", model.covars_)
