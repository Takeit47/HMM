import numpy as np
from hmmlearn import hmm

# 假设我们有两个字母的训练数据，每个字母有多个样本，每个样本是一个二维坐标序列
# 这里我们用随机生成的数据来示例，实际应用中你需要用真实的笔触坐标数据

# 生成字母A的训练数据
train_data_A = [
    np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]),
    np.array([[0.2, 0.1], [0.3, 0.2], [0.4, 0.3]])
]

# 生成字母B的训练数据
train_data_B = [
    np.array([[1.1, 1.2], [1.2, 1.3], [1.3, 1.4]]),
    np.array([[1.2, 1.1], [1.3, 1.2], [1.4, 1.3]])
]

# 训练字母A的HMM模型
model_A = hmm.GaussianHMM(n_components=2, covariance_type='full', n_iter=100)
train_data_A_combined = np.vstack(train_data_A)
print(train_data_A_combined)
lengths_A = [len(seq) for seq in train_data_A]
print(lengths_A)
model_A.fit(train_data_A_combined, lengths=lengths_A)

# 训练字母B的HMM模型
model_B = hmm.GaussianHMM(n_components=2, covariance_type='full', n_iter=100)
train_data_B_combined = np.vstack(train_data_B)
lengths_B = [len(seq) for seq in train_data_B]
model_B.fit(train_data_B_combined, lengths=lengths_B)

# 对新的笔触序列进行预测
new_sequence = np.array([[0.15, 0.25], [0.25, 0.35], [0.35, 0.45]])

# 计算新序列在每个模型下的概率
logprob_A = model_A.score(new_sequence)
logprob_B = model_B.score(new_sequence)

# 转换为概率（对数概率转换为普通概率）
prob_A = np.exp(logprob_A)
prob_B = np.exp(logprob_B)

print("Probability of being letter A:", prob_A)
print("Probability of being letter B:", prob_B)

# 预测结果：选择概率最大的那个字母
predicted_letter = 'A' if prob_A > prob_B else 'B'
print("Predicted letter:", predicted_letter)
