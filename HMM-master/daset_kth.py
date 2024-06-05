import os
import pickle
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CATEGORY_INDEX = {
    "boxing": 0,
    "handclapping": 1,
    "handwaving": 2,
    "jogging": 3,
    "running": 4,
    "walking": 5
}


class KTHDataset(Dataset):
    def __init__(self, directory, dataset="train", sequences_file="00sequences.txt"):
        print(f"Initializing dataset: {dataset}")
        self.instances, self.labels = self.read_dataset(directory, dataset, sequences_file)
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


    def read_dataset(self, directory, dataset="train", sequences_file="00sequences.txt"):
        print(f"Reading dataset: {dataset}")
        instances = []
        labels = []

        with open(os.path.join(directory, sequences_file), 'r') as file:
            lines = file.readlines()

        for line in lines:
            line_split = line.strip().split()
            video_name = line_split[0]
            frame_ranges = line_split[2:]

            category = video_name.split('_')[1]
            if category not in CATEGORY_INDEX:
                continue

            folder_path = os.path.join(directory, category)
            filepath = os.path.join(folder_path, f"{video_name}_uncomp.avi")

            print(f"Processing file: {video_name}")

            # Open the video file
            cap = cv2.VideoCapture(filepath)
            frames = []

            for frame_range in frame_ranges:
                start, end = map(int, frame_range.split('-'))
                for i in range(start - 1, end):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Resize frame to 80x60
                    resized_frame = cv2.resize(frame, (80, 60))
                    # Convert to grayscale
                    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                    frames.append(gray_frame)

            cap.release()

            if len(frames) > 0:
                # Normalize frames
                frames = np.array(frames, dtype=np.float32) / 255.0
                instances.append(frames)
                labels.append(CATEGORY_INDEX[category])
                print(f"Added {len(frames)} frames for file: {video_name}")

        # Convert instances and labels to numpy arrays
        instances = np.array(instances, dtype=object)  # Use dtype=object for variable-length sequences
        labels = np.array(labels, dtype=np.uint8)

        print(f"Completed reading dataset: {dataset}")
        return instances, labels


def save_preprocessed_data(directory, sequences_file="00sequences.txt"):
    datasets = ["train", "dev", "test"]
    for dataset in datasets:
        print(f"Saving preprocessed data for dataset: {dataset}")
        kth_dataset = KTHDataset(directory, dataset, sequences_file)
        save_path = os.path.join("data", f"{dataset}_preprocessed.p")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({
                "instances": kth_dataset.instances,
                "labels": kth_dataset.labels
            }, f)
        print(f"Saved preprocessed data for dataset: {dataset} to {save_path}")


if __name__ == "__main__":
    directory = "KTH"
    sequences_file = "00sequences.txt"
    save_preprocessed_data(directory, sequences_file)
