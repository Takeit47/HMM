import os
os.environ["OMP_NUM_THREADS"] = '1'

import xml.etree.ElementTree as ET
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from func import DTW


def load_letter_dataset(pth, n_clusters=3, dtw=False):
    train_results, test_results = [], []
    labels = []
    train_LENS, test_LENS = [], []
    scaler = MinMaxScaler()
    train_clusters, test_clusters = [], []

    for file in os.listdir(pth):
        # print(file)
        labels.append(file.split('.')[0])
        file_path = os.path.join(pth, file)
        tree = ET.parse(file_path)
        root = tree.getroot()
        train_coords, test_coords = [], []
        train_lens, test_lens = [], []
        coords = []

        for i, training_example in enumerate(root.findall('trainingExample')):
            len = 0
            for coord in training_example.findall('coord'):
                x = float(coord.get('x'))
                y = float(coord.get('y'))
                t = int(coord.get('t'))
                coords.append((t, x, y))
                if i % 2 == 0:
                    test_coords.append((t, x, y))
                else:
                    train_coords.append((t, x, y))
                len += 1
            if i % 2 == 0:
                test_lens.append(len)
            else:
                train_lens.append(len)
        train_coords.sort()
        test_coords.sort()
        # Extract the sorted x, y coordinates
        sorted_train = [(x, y) for _, x, y in train_coords]
        sorted_test = [(x, y) for _, x, y in test_coords]
        # Normalize the coordinates
        normalized_train = scaler.fit_transform(sorted_train)
        normalized_test = scaler.transform(sorted_test)

        kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
        # print(np.concatenate((normalized_train, normalized_test)))
        kmeans.fit(np.concatenate((normalized_train, normalized_test)))
        train_labels = kmeans.labels_[:np.sum(train_lens)]
        test_labels = kmeans.labels_[np.sum(train_lens):]

        # train_clusters.append(np.array(train_labels/(n_clusters-1), dtype=float))
        train_clusters.append(np.array(train_labels))
        train_results.append(np.array(normalized_train))
        train_LENS.append(train_lens)

        # test_clusters.append(np.array(test_labels/(n_clusters-1), dtype=float))
        test_clusters.append(np.array(test_labels))
        test_results.append(normalized_test)
        test_LENS.append(test_lens)

    if dtw:
        train_clusters, train_LENS = DTW(train_clusters, train_LENS)
        test_clusters, test_LENS = DTW(test_clusters, test_LENS)

    return ((train_results, [np.reshape(train_cluster, (-1, 1)) for train_cluster in train_clusters], labels, train_LENS),
            (test_results, [np.reshape(test_cluster, (-1, 1)) for test_cluster in test_clusters], labels, test_LENS))


def load_figure_dataset(pth, n_clusters=8, dtw=False):
    train_results, test_results = [], []
    labels = []
    train_LENS, test_LENS = [], []
    scaler = MinMaxScaler()
    train_clusters, test_clusters = [], []

    # Define the grid size
    grid_size = n_clusters

    for file in os.listdir(pth):
        labels.append(file.split('.')[0])
        file_path = os.path.join(pth, file)
        tree = ET.parse(file_path)
        root = tree.getroot()
        train_coords, test_coords = [], []
        train_lens, test_lens = [], []
        coords = []

        for i, training_example in enumerate(root.findall('trainingExample')):
            len = 0
            for coord in training_example.findall('coord'):
                x = float(coord.get('x'))
                y = float(coord.get('y'))
                t = int(coord.get('t'))
                coords.append((t, x, y))
                if i % 2 == 0:
                    test_coords.append((t, x, y))
                else:
                    train_coords.append((t, x, y))
                len += 1
            if i % 2 == 0:
                test_lens.append(len)
            else:
                train_lens.append(len)
        train_coords.sort()
        test_coords.sort()
        # Extract the sorted x, y coordinates
        sorted_train = [(x, y) for _, x, y in train_coords]
        sorted_test = [(x, y) for _, x, y in test_coords]
        # Normalize the coordinates
        normalized_train = scaler.fit_transform(sorted_train)
        normalized_test = scaler.transform(sorted_test)

        # Assign samples to grid cells
        train_labels = np.zeros((normalized_train.shape[0] ))
        test_labels = np.zeros((normalized_test.shape[0] ))

        for i, (x, y) in enumerate(normalized_train):
            grid_x = int(x * grid_size)
            grid_y = int(y * grid_size)
            train_labels[i] = int(grid_x * grid_size + grid_y)

        for i, (x, y) in enumerate(normalized_test):
            grid_x = int(x * grid_size)
            grid_y = int(y * grid_size)
            test_labels[i] = int(grid_x * grid_size + grid_y)
        # print(train_labels, test_labels)
        train_clusters.append(train_labels)
        train_results.append(np.array(normalized_train))
        train_LENS.append(train_lens)

        test_clusters.append(test_labels)
        test_results.append(normalized_test)
        test_LENS.append(test_lens)

    if dtw:
        train_clusters, train_LENS = DTW(train_clusters, train_LENS)
        test_clusters, test_LENS = DTW(test_clusters, test_LENS)

    return ((train_results, [np.reshape(train_cluster, (-1, 1)) for train_cluster in train_clusters], labels, train_LENS),
            (test_results, [np.reshape(test_cluster, (-1, 1)) for test_cluster in test_clusters], labels, test_LENS))

if __name__ == '__main__':
    # Usage
    file_path = 'data_figure/'
    (datasets, kmeans_data, labels, lens), _ = load_letter_dataset(file_path, n_clusters=16, dtw=True)
    print(kmeans_data, lens)
    print(kmeans_data[0].shape)
    for _ in kmeans_data[0]:
        print(_)

