import os
import cv2
# 假设 filepath 是你的视频文件路径
filepath = 'KTH\walking\person01_walking_d1_uncomp.avi'
if not os.path.exists(filepath):
    print(f"文件不存在: {filepath}")
else:
    print(f"文件存在: {filepath}")

# 尝试打开视频文件
cap = cv2.VideoCapture(filepath)
if not cap.isOpened():
    print(f"无法打开视频文件: {filepath}")
else:
    print(f"视频文件打开成功: {filepath}")

# 读取第一帧
ret, frame = cap.read()
if not ret:
    print(f"无法读取视频帧: {filepath}")
else:
    # 显示帧（如果需要）
    cv2.imshow('frame', frame)
    cv2.waitKey(0)  # 等待按键后关闭窗口
    cv2.destroyAllWindows()
    print(f"帧读取成功: {filepath}")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if frame_count == 0:
    print(f"视频文件没有帧: {filepath}")
else:
    print(f"视频文件包含 {frame_count} 帧")