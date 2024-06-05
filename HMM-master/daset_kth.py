import os
import pickle
import cv2
import numpy as np
import torch
import re
from torch.utils.data import Dataset, TensorDataset

CATEGORY_INDEX = {
    "boxing": 0,
    "handclapping": 1,
    "handwaving": 2,
    "jogging": 3,
    "running": 4,
    "walking": 5
}


def parse_sequence_file(sequence_file_path):
    print("Parsing sequence file:", sequence_file_path)
    frames_idx = {}

    with open(sequence_file_path, 'r') as content_file:
        content = content_file.read()

    # Replace tab and newline characters with space, then split file's content
    # into strings.
    content = re.sub("[\t\n]", " ", content).split()

    # Current video that we are parsing.
    current_filename = ""

    for s in content:
        if s == "frames":
            # Ignore this token.
            continue
        elif s.find("-") >= 0:
            # This is the token we are looking for. e.g. 1-95.
            if s[-1] == ',':
                # Remove comma.
                s = s[:-1]

            # Split into 2 numbers => [1, 95]
            idx = s.split("-")

            # Add to dictionary.
            if current_filename not in frames_idx:
                frames_idx[current_filename] = []
            frames_idx[current_filename].append((int(idx[0]), int(idx[1])))
        else:
            # Parse next file.
            current_filename = s + "_uncomp.avi"

    return frames_idx


class KTHDataset(Dataset):
    def __init__(self, directory, dataset="train", sequences_file="00sequences.txt"):
        print(f"Initializing dataset: {dataset}")
        self.instances, self.labels = self.read_dataset(directory, dataset, sequences_file)
        self.dataset = TensorDataset(*self.instances)  # Change here
        self.labels = torch.from_numpy(self.labels)
        print(f"Dataset {dataset} initialized with {len(self.instances)} instances")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        instance = self.dataset[idx]  # Change here
        sample = {
            "instance": instance,
            "label": self.labels[idx]
        }
        return sample

    def read_dataset(self, directory, dataset="train", sequences_file="00sequences.txt"):
        print(f"Reading dataset: {dataset}")
        instances = []
        labels = []
        frames_idx = parse_sequence_file(os.path.join(directory, sequences_file))

        for filename, frame_ranges in frames_idx.items():
            category = filename.split('_')[1]
            if category not in CATEGORY_INDEX:
                continue

            folder_path = os.path.join(directory, category)
            filepath = os.path.join(folder_path, f"{filename}_uncomp.avi")

            print(f"Processing file: {filename}")

            # Open the video file
            cap = cv2.VideoCapture(filepath)
            frames = []

            for frame_range in frame_ranges:
                start, end = frame_range
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
                instances.append(torch.from_numpy(frames))  # Change here
                labels.append(CATEGORY_INDEX[category])
                print(f"Added {len(frames)} frames for file: {filename}")

        # Convert instances and labels to numpy arrays
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
