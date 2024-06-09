import os
import pickle
import cv2
import numpy as np
import re

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 定义动作类别与其索引的映射关系
CATEGORY_INDEX = {
    "boxing": 0,
    "handclapping": 1,
    "handwaving": 2,
    "jogging": 3,
    "running": 4,
    "walking": 5
}

# 硬编码数据集分割
dataset_split = {
    "Training": ["person11", "person12", "person13", "person14", "person15", "person16", "person17", "person18"],
    "Validation": ["person19", "person20", "person21", "person23", "person24", "person25", "person01", "person04"],
    "Test": ["person22", "person02", "person03", "person05", "person06", "person07", "person08", "person09", "person10"]
}


# 解析序列文件，提取视频帧索引范围
def parse_sequence_file(sequence_file_path):
    print("解析序列文件：", sequence_file_path)
    frames_idx = {}
    try:
        with open(sequence_file_path, 'r') as content_file:
            content = content_file.read()
    except FileNotFoundError:
        print(f"序列文件 {sequence_file_path} 未找到。")
        return frames_idx

    lines = content.split('\n')

    # 解析帧索引范围信息
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        current_filename = parts[0] + "_uncomp.avi"
        frames_idx[current_filename] = []
        for idx_range in parts[2:]:
            if idx_range.endswith(','):
                idx_range = idx_range[:-1]
            start, end = map(int, idx_range.split('-'))
            frames_idx[current_filename].append((start, end))

    return frames_idx  # 返回包含帧索引范围的字典


# 从光流计算运动的强度和方向特征

def extract_state_features(flow, num_bins=4, grid_size=(2, 2)):
    """
    从光流场中提取运动特征，包括全局特征、方向直方图和局部特征。

    参数:
    flow (numpy.ndarray): 光流场，形状为 (height, width, 2)，包含光流的x和y分量。
    num_bins (int): 方向直方图的bin数量，默认值为4。
    grid_size (tuple): 图像划分的网格大小，默认值为 (2, 2)。

    返回:
    numpy.ndarray: 提取的特征向量。
    """

    # 获取光流场的高和宽
    h, w = flow.shape[:2]

    # 将光流从笛卡尔坐标转换为极坐标，得到运动强度 (magnitude) 和方向 (angle)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 计算全局运动强度和方向的统计特征
    mean_mag = np.mean(mag)  # 运动强度的平均值
    std_mag = np.std(mag)  # 运动强度的标准差
    mean_ang = np.mean(ang)  # 运动方向的平均值
    std_ang = np.std(ang)  # 运动方向的标准差

    # 计算运动方向的直方图，范围为 0 到 2*pi
    hist, _ = np.histogram(ang, bins=num_bins, range=(0, 2 * np.pi), density=True)

    # 将图像划分为若干子区域，并在每个子区域内计算局部特征
    grid_h, grid_w = h // grid_size[0], w // grid_size[1]
    local_features = []

    for i in range(0, h, grid_h):
        for j in range(0, w, grid_w):
            # 提取子区域的光流强度和方向
            local_mag = mag[i:i + grid_h, j:j + grid_w]
            local_ang = ang[i:i + grid_h, j:j + grid_w]

            # 计算子区域的运动强度和方向的平均值
            local_mean_mag = np.mean(local_mag)
            local_mean_ang = np.mean(local_ang)

            # 将局部特征添加到列表中
            local_features.extend([local_mean_mag, local_mean_ang])

    # 合并全局特征、方向直方图和局部特征，形成最终的特征向量
    features = [mean_mag, std_mag, mean_ang, std_ang]
    features.extend(hist)
    features.extend(local_features)

    # 返回特征向量
    return np.array(features)

# 预处理数据集，包括读取视频、计算光流和提取特征
def preprocess_dataset(directory, frames_idx, dataset_split, skip_frames=4):
    print("预处理数据集")
    datasets = {split_type: {category: [] for category in CATEGORY_INDEX.keys()} for split_type in dataset_split.keys()}
    feature_lengths = {split_type: {category: [] for category in CATEGORY_INDEX.keys()} for split_type in
                       dataset_split.keys()}

    for split_type, persons in dataset_split.items():
        for person in persons:
            for filename, frame_ranges in frames_idx.items():
                if person in filename:
                    category = filename.split('_')[1]
                    if category not in CATEGORY_INDEX:
                        continue

                    folder_path = os.path.join(directory, category)
                    if not os.path.exists(folder_path):
                        print(f"动作文件夹 {category} 未找到。")
                        continue

                    filepath = os.path.join(folder_path, filename)
                    if not os.path.exists(filepath):
                        print(f"视频文件 {filename} 未找到。")
                        continue

                    cap = cv2.VideoCapture(filepath)
                    if not cap.isOpened():
                        print(f"无法打开视频文件：{filename}")
                        continue

                    state_sequence = []
                    # 遍历视频的每一帧
                    for frame_range in frame_ranges:
                        start, end = frame_range
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)  # 设置视频的读取位置
                        prev_frame = None
                        for i in range(start, end + 1, skip_frames + 1):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            resized_frame = cv2.resize(frame, (160, 120))  # 调整帧大小
                            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
                            if prev_frame is not None:
                                # 计算光流
                                flow = cv2.calcOpticalFlowFarneback(prev_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                                state_features = extract_state_features_hist(flow)  # 提取状态特征
                                state_sequence.append(state_features)
                            prev_frame = gray_frame

                    if state_sequence:
                        # state_sequence = extract_state_features_hist(np.array(state_sequence))
                        datasets[split_type][category].append(state_sequence)
                        feature_lengths[split_type][category].append(len(state_sequence))
                        print(f"为文件 {filename} 添加了 {len(state_sequence)} 个特征向量")

                    cap.release()

    return datasets, feature_lengths


# 将预处理的数据保存到文件
def save_data(datasets, feature_lengths, directory="data"):
    os.makedirs(directory, exist_ok=True)  # 如果目录不存在，则创建它
    for split_type, category_data in datasets.items():
        split_dir = os.path.join(directory, split_type)
        os.makedirs(split_dir, exist_ok=True)
        for category, instances in category_data.items():
            if not instances:
                print(f"警告：类别 {category} 在 {split_type} 中没有数据，将跳过保存。")
                continue
            # 展平所有特征向量并拼接
            flattened_instances = [instance for video in instances for instance in video]
            merged_instances = np.array(flattened_instances)
            labels = np.full((merged_instances.shape[0],), CATEGORY_INDEX[category], dtype=np.uint8)
            save_path = os.path.join(split_dir, f"preprocessed_dataset_{CATEGORY_INDEX[category]}.p")
            with open(save_path, "wb") as f:
                pickle.dump({"instances": merged_instances, "labels": labels}, f)
            print(f"{split_type} - 类别 {category} 实例形状: {merged_instances.shape}")
            print(f"{split_type} - 类别 {category} 标签形状: {labels.shape}")
            print(f"{split_type} 数据已保存到: {save_path}\n")

            # 保存特征长度信息
            lengths_save_path = os.path.join(split_dir, f"feature_lengths_{CATEGORY_INDEX[category]}.p")
            with open(lengths_save_path, "wb") as f:
                pickle.dump(feature_lengths[split_type][category], f)
            print(f"{split_type} - 类别 {category} 特征长度: {feature_lengths[split_type][category]}")
            print(f"{split_type} 特征长度数据已保存到: {lengths_save_path}\n")


def load_data(directory="data"):
    datasets = {"Training": {}, "Validation": {}, "Test": {}}
    for split_type in datasets.keys():
        split_dir = os.path.join(directory, split_type)
        if not os.path.exists(split_dir):
            continue
        for filename in os.listdir(split_dir):
            if filename.endswith(".p"):
                category = int(filename.split('_')[-1].split('.')[0])
                with open(os.path.join(split_dir, filename), "rb") as f:
                    data = pickle.load(f)
                    datasets[split_type][category] = data
    return datasets

def load_feature_lengths(directory="data"):
    feature_lengths = {"Training": {}, "Validation": {}, "Test": {}}
    for split_type in feature_lengths.keys():
        split_dir = os.path.join(directory, split_type)
        if not os.path.exists(split_dir):
            continue
        for filename in os.listdir(split_dir):
            if filename.startswith("feature_lengths_") and filename.endswith(".p"):
                category = int(filename.split('_')[-1].split('.')[0])
                with open(os.path.join(split_dir, filename), "rb") as f:
                    data = pickle.load(f)
                    feature_lengths[split_type][category] = data
    return feature_lengths


# 主程序入口
if __name__ == "__main__":
    directory = "data_kth/"  # 设置数据集的根目录
    sequences_file = "00sequences.txt"  # 序列文件名
    frames_idx = parse_sequence_file(os.path.join(directory, sequences_file))  # 解析序列文件
    datasets, feature_lengths = preprocess_dataset(directory, frames_idx, dataset_split)  # 预处理数据集
    save_data(datasets, feature_lengths)  # 保存预处理后的数据

    # 打印特征长度信息
    for split_type, category_data in feature_lengths.items():
        print(f"{split_type} 特征长度信息：")
        for category, lengths in category_data.items():
            print(f"类别 {category}: {lengths}")
