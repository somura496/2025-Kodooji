# ItemQueue_Jun25.py
# 中心座標の距離に基づいてインデックスを並び替える関数
import numpy as np

def get_sorted_by_center(centers, frame_shape):
    """
    フレーム中心からのユークリッド距離順にインデックスを返す
    """
    center_frame = np.array([frame_shape[1] // 2, frame_shape[0] // 2])
    distances = [np.linalg.norm(np.array(c) - center_frame) for c in centers]
    return sorted(range(len(distances)), key=lambda i: distances[i])
