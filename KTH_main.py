import os

os.environ["OMP_NUM_THREADS"] = '1'

from dataset_kth import load_data, load_feature_lengths, CATEGORY_INDEX
from func import *
from graphic import plot_confusion_matrix
from sklearn.cluster import KMeans
from Model import HMModel


def train_and_predict(k, n, train_data, test_data, train_lens, test_lens):
    train_data, test_data = kmeans(train_data, test_data, train_lens, test_lens, k)
    # train_clusters, train_LENS = DTW(train_data, train_lens)
    # test_clusters, test_LENS = DTW(test_data, test_lens)
    ave_acc = 0
    for _ in range(10):
        models = [HMModel(n, 2000) for i in range(6)]
        train(models, train_data, CATEGORY_INDEX.keys(), train_lens)
        # plot_confusion_matrix(models, test_data, test_lens, CATEGORY_INDEX.keys())
        pred_labels = []
        true_labels = []
        acc = 0
        for idx, (samples, test_len) in enumerate(zip(test_data, test_lens)):
            start = 0
            for sample_len in test_len:
                # print(sample_len)
                pred, scores = predict(models, samples[start:start + sample_len])
                # print(pred, idx)
                start += sample_len
                pred_labels.append(pred)
                true_labels.append(idx)
                if pred == idx:
                    acc += 1
        ave_acc += acc / len(true_labels)
        plot_confusion_matrix(models, test_data, test_lens, class_names=CATEGORY_INDEX.keys())
    print(f"n={n}, k={k}: {ave_acc / 10}")
    return ave_acc / 10


if __name__ == '__main__':

    datasets = load_data()
    datasets_lengths = load_feature_lengths()
    # # 示例输出，打印某个类别的实例形状
    # print(datasets["Training"][0]["instances"].shape)
    # print(datasets_lengths["Training"][0])
    train_data, test_data = ([datasets['Training'][i]["instances"] for i in range(6)],
                             [datasets['Test'][i]['instances'] for i in range(6)])
    train_lens, test_lens = datasets_lengths['Training'].values(), datasets_lengths['Test'].values()
    max_acc_args = []
    # for n in range(2, 6):
    #     for k in range(n, 20):
    acc = train_and_predict(15, 5, train_data, test_data, train_lens, test_lens)
    max_acc_args.append([5, 15, acc])

    print(f"max acc args:{max_acc_args.sort()}")
