import os
import pickle
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

CATEGORY_INDEX = {
    "boxing": 0,
    "handclapping": 1,
    "handwaving": 2,
    "jogging": 3,
    "running": 4,
    "walking": 5
}


class KTHDataset(Dataset):
    def __init__(self, directory, dataset="train", frame_interval=5):
        print(f"Initializing dataset: {dataset}")
        self.instances, self.labels = self.read_dataset(directory, dataset, frame_interval)
        self.instances = torch.from_numpy(self.instances)
        self.labels = torch.from_numpy(self.labels)
        print(f"Dataset {dataset} initialized with {len(self.instances)} instances")

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, idx):
        sample = {
            "instance": self.instances[idx],
            "label": self.labels[idx]
        }
        return sample

    def read_dataset(self, directory, dataset="train", frame_interval=5):
        print(f"Reading dataset: {dataset}")
        instances = []
        labels = []

        for category in CATEGORY_INDEX.keys():
            print(f"Processing category: {category}")
            folder_path = os.path.join(directory, category)
            filenames = sorted(os.listdir(folder_path))

            for filename in filenames:
                filepath = os.path.join(folder_path, filename)
                print(f"Processing file: {filename}")

                # Open the video file
                cap = cv2.VideoCapture(filepath)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frames = []

                for i in range(0, frame_count, frame_interval):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert to grayscale
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Resize frame
                    resized_frame = cv2.resize(gray_frame, (80, 60))
                    frames.append(resized_frame)

                cap.release()

                if len(frames) > 0:
                    instances.append(np.array(frames, dtype=np.float32))
                    labels.append(CATEGORY_INDEX[category])
                    print(f"Added {len(frames)} frames for file: {filename}")

        instances = np.array(instances, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)

        # Standardize the instances
        scaler = StandardScaler()
        instances_shape = instances.shape
        instances = instances.reshape(instances_shape[0], -1)
        instances = scaler.fit_transform(instances)
        instances = instances.reshape(instances_shape)

        print(f"Completed reading dataset: {dataset}")
        return instances, labels


def save_preprocessed_data(directory, frame_interval=5):
    datasets = ["train", "dev", "test"]
    for dataset in datasets:
        print(f"Saving preprocessed data for dataset: {dataset}")
        kth_dataset = KTHDataset(directory, dataset, frame_interval)
        save_path = os.path.join("data", f"{dataset}_preprocessed.p")
        with open(save_path, "wb") as f:
            pickle.dump({
                "instances": kth_dataset.instances,
                "labels": kth_dataset.labels
            }, f)
        print(f"Saved preprocessed data for dataset: {dataset} to {save_path}")


if __name__ == "__main__":
    directory = "KTH"
    save_preprocessed_data(directory)
