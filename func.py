import os

os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np


def DTW(datas_list, lens_list):
    import DTW
    # print(datas_list)
    new_datas_list, new_lens_list =[], []
    for data, lens in zip(datas_list, lens_list):
        data_list = DTW.get_data_list(data, lens)
        data_list, lens = DTW.fast_dtw(data_list)
        # print(list(len(data) for data in data_list))
        new_data = DTW.reshape_2_data(data_list)
        new_datas_list.append(new_data)
        new_lens_list.append(lens)
    return new_datas_list, new_lens_list


def train(models, observations, labels, lens):
    for (model, observation, label, len) in zip(models, observations, labels, lens):
        model.Laplacian()
        try:
            # print(observation, len)
            model.train(observation, len)
        except ValueError as e:
            print("Error during training:", e)
            print("Observation:\n", observation)
            print("Observation lengths:\n", len)
            raise

        # print(f"model {label} train finished.")

    return models


def predict(models, sample):
    scores = [model.predict(sample) for model in models]
    return np.argmax(scores), scores
