import numpy as np


def train(models, observations, labels, lens):
    for (model, observation, label, len) in zip(models, observations, labels, lens):
        model.Laplacian()
        model.train(observation, len)
        print(f"model {label} train finished.")
    return models


def predict(models, sample):
    scores = [model.predict(sample) for model in models]
    return np.argmax(scores), scores