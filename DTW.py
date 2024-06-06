import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def get_data_list(datas, lens):
    data_list = []

    start = 0
    for len in lens:
        data_list.append(datas[start:start + len].reshape(-1, 1))
        start += len
    return data_list


def fast_dtw(data_list):

    std_data = max(data_list, key=len)
    lens = []
    new_data_list = []
    distances = []

    for data in data_list:
        # print(std_data.shape, data.shape)
        distance, path = fastdtw(std_data, data, dist=euclidean)
        # print(path)
        new_data_path = list([_[1] for _ in path])
        # print(len(new_data_path))
        new_data = [data[i] for i in new_data_path]
        # print(path)
        distances.append(distance)
        new_data_list.append(new_data)
        lens.append(len(new_data_path))

    # print(list([len(_) for _ in new_data_list]))
    return new_data_list, lens


def reshape_2_data(data_list):
    # print(list(len(data) for data in data_list))
    new_data = np.concatenate(data_list)
    # print(new_data)
    # new_data = new_data.reshape(new_data.shape[0] * new_data.shape[1], new_data.shape[2])
    return np.array(new_data)


if __name__ == '__main__':
    datas = np.array([1, 2, 3, 4, 5, 1, 3, 5, 1, 3, 4, 5])
    lens = [5, 3, 4]
    data_list = get_data_list(datas, lens)
    data_list, lens = fast_dtw(data_list)
    new_data = reshape_2_data(data_list)