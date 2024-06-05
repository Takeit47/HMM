import os
os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np


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

        print(f"model {label} train finished.")

    return models


def predict(models, sample):
    scores = [model.predict(sample) for model in models]
    return np.argmax(scores), scores