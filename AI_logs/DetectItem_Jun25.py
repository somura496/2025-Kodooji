# DetectItem_Jun25.py
# 輪郭抽出と画像分類、座標変換を実行するモジュール

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

from common import *
from TransCoord_Jun25 import transform_coordinate

def rect_preprocess(img):
    """
    矩形画像を正方形にパディング（黒埋め）して整形する
    """
    h, w, _ = img.shape
    longest_edge = max(h, w)
    top = bottom = left = right = 0
    if h < longest_edge:
        diff = longest_edge - h
        top = diff // 2
        bottom = diff - top
    elif w < longest_edge:
        diff = longest_edge - w
        left = diff // 2
        right = diff - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def detect_and_classify(frame, model, pic_size, min_area, max_area, label_name):
    """
    入力フレームから対象物を検出して分類し、DOBOT座標に変換する

    Returns:
        result_num_all: ラベル番号リスト
        transform_pos_all: 座標リスト
        centers_all: 中心座標リスト
        edframe: 描画済みフレーム
    """
    edframe = frame.copy()
    gray = cv2.cvtColor(edframe, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_num_all = []
    centers_all = []
    transform_pos_all = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        center, size, angle = rect
        center = tuple(map(int, center))
        size = tuple(map(int, size))

        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        h, w = frame.shape[:2]
        rotated = cv2.warpAffine(frame, rot_mat, (w, h))
        cropped = cv2.getRectSubPix(rotated, size, center)
        img_src = rect_preprocess(cropped)

        image = cv2.resize(img_src, (pic_size, pic_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img_to_array(image).astype("float32") / 255.
        image = image[None, ...]

        result = model.predict(image)
        result_num = int(np.argmax(result, axis=-1)[0])
        mp_x, mp_y = center
        transform_pos = transform_coordinate(mp_x, mp_y)

        result_num_all.append(result_num)
        transform_pos_all.append(transform_pos)
        centers_all.append(center)

        x, y, w_, h_ = cv2.boundingRect(contour)
        label = f"{result_num} {label_name[result_num]}"
        cv2.rectangle(edframe, (x, y), (x+w_, y+h_), draw_white, 1)
        cv2.putText(edframe, label, (x, y-4), font, FONT_SIZE, draw_black, FONT_WIDTH)
        cv2.putText(edframe, f"DOBOT: {transform_pos}", (x+w_+5, y+10), font, FONT_SIZE, draw_green, FONT_WIDTH)

    return result_num_all, transform_pos_all, centers_all, edframe
