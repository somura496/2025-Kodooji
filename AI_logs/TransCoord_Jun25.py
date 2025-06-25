# TransCoord_Jun25.py
# カメラ座標からDOBOT座標への変換関数
import numpy as np
import cv2
from TransformationMatrix import MATRIX

MATRIX = np.array(MATRIX)

def transform_coordinate(pos_x, pos_y):
    """
    カメラ座標 (pos_x, pos_y) をDOBOT座標に変換する
    """
    pos = np.array([[[pos_x, pos_y]]], dtype='float32')
    result = cv2.perspectiveTransform(pos, MATRIX)
    return int(result[0][0][0]), int(result[0][0][1])
