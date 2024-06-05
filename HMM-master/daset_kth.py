import os
import pickle
import cv2
import numpy as np
import torch
import re

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


def preprocess_dataset(directory, sequences_file="00sequences.txt"):
    print("Preprocessing dataset")
    instances = [[] for _ in range(len(CATEGORY_INDEX))]  # Initialize empty lists for each action
    labels = []

    frames_idx = parse_sequence_file(os.path.join(directory, sequences_file))

    for filename, frame_ranges in frames_idx.items():
        category = filename.split('_')[1]
        if category not in CATEGORY_INDEX:
            continue

        folder_path = os.path.join(directory, category)

        # Check if the action folder exists
        if not os.path.exists(folder_path):
            print(f"Action folder {category} not found.")
            continue

        filepath = os.path.join(folder_path, f"{filename}")

        # Check if the video file exists
        if not os.path.exists(filepath):
            print(f"Video file {filename}.avi not found.")
            continue

        # Open the video file
        cap = cv2.VideoCapture(filepath)

        # Check if the video file was opened successfully
        if not cap.isOpened():
            print(f"Could not open video file: {filename}.avi")
            continue

        for frame_range in frame_ranges:
            start, end = frame_range
            frames = []  # Initialize frames list for each video

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

            if len(frames) > 0:
                # Normalize frames
                frames = np.array(frames, dtype=np.float32) / 255.0
                instances[CATEGORY_INDEX[category]].append(frames)
                labels.append(CATEGORY_INDEX[category])
                print(f"Added {len(frames)} frames for file: {filename}")

        cap.release()

    # Convert instances and labels to numpy arrays
    for i in range(len(instances)):
        instances[i] = np.array(instances[i])

    labels = np.array(labels, dtype=np.uint8)

    print("Completed preprocessing dataset")
    return instances, labels


def combine_sequences(instances, labels):
    max_length = max(len(seq) for seq in instances)
    combined_sequences = []
    for i, seq in enumerate(instances):
        if len(seq) > 0:
            padded_seq = np.pad(seq, ((0, max_length - len(seq)), (0, 0), (0, 0)), mode='constant')
            combined_sequences.append(padded_seq)
        else:
            print(f"No frames found for action {i}")
    return combined_sequences



if __name__ == "__main__":
    directory = "KTH"
    sequences_file = "00sequences.txt"
    instances, labels = preprocess_dataset(directory, sequences_file)
    combined_sequences = combine_sequences(instances, labels)

    # Print the structure of combined sequences
    for i, seq in enumerate(combined_sequences):
        print(f"Action {i}: {seq.shape}")

    # Save preprocessed data
    save_path = os.path.join("data", "preprocessed_dataset.p")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({
            "instances": combined_sequences,
            "labels": list(range(len(CATEGORY_INDEX)))  # Dummy labels since instances are already grouped by action
        }, f)
    print(f"Preprocessed data saved to: {save_path}")
