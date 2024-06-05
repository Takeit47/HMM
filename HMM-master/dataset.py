import os
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


def load_figure_dataset(pth, n_clusters=3):
    train_results, test_results = [], []
    labels = []
    train_LENS, test_LENS = [], []
    scaler = MinMaxScaler()
    train_clusters, test_clusters = [], []

    for file in os.listdir(pth):
        print(file)
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

        kmeans = KMeans(n_clusters=n_clusters)
        # print(np.concatenate((normalized_train, normalized_test)))
        kmeans.fit(np.concatenate((normalized_train, normalized_test)))
        train_labels = kmeans.labels_[:np.sum(train_lens)]
        test_labels = kmeans.labels_[np.sum(train_lens):]

        train_clusters.append(np.array(train_labels/(n_clusters-1), dtype=float))
        train_results.append(np.array(normalized_train))
        train_LENS.append(train_lens)

        test_clusters.append(np.array(test_labels/(n_clusters-1), dtype=float))
        test_results.append(normalized_test)
        test_LENS.append(test_lens)

    return ((train_results, [np.reshape(train_cluster, (-1, 1)) for train_cluster in train_clusters], labels, train_LENS),
            (test_results, [np.reshape(test_cluster, (-1, 1)) for test_cluster in test_clusters], labels, test_LENS))


if __name__ == '__main__':
    # Usage
    file_path = 'data_figure/'
    (datasets, kmeans_data, labels, lens), _ = load_figure_dataset(file_path, n_clusters=3)
    print(datasets[0].shape, lens[0])
    print(kmeans_data[0].shape)

