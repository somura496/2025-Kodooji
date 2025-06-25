# -*- coding: utf-8 -*-
'''
Project Name  : dobot_ai
File Name     : classifier_alter.py
File Encoding : UTF-8
Copyright © 2020 Afrel Co.,Ltd.
'''

import os
import sys, time
from datetime import datetime
from argparse import ArgumentParser

from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

import cameraSetting as camset
import dobotClassifier as dc

from common import *
from TransformationMatrix import MATRIX
MATRIX = np.array(MATRIX)

# ラベルごとの仕分け先座標
SORTING_POSITIONS = {
    0: (235, -120),
    1: (235, 0),
    2: (235, 120),
}

# 画像のプリプロセス

def rect_preprocess(img):
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
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    return img

# 座標変換

def transform_coordinate(pos_x, pos_y):
    pos = np.array([[[pos_x, pos_y]]], dtype='float32')
    result = cv2.perspectiveTransform(pos, MATRIX)
    return int(result[0][0][0]), int(result[0][0][1])

# 青色LED検出

def is_blue_led_detected(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([100, 150, 100])
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.countNonZero(mask) > 100

# 引数取得

def get_option():
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", dest="load_dir", default=DATA_DIR)
    parser.add_argument("-m", "--modelname", dest="modelname", default="model")
    parser.add_argument("-s", "--size", dest="pic_size", type=int, default=PIC_SIZE)
    parser.add_argument("--min", dest="min_area_size", type=int, default=MIN_AREA_SIZE)
    parser.add_argument("--max", dest="max_area_size", type=int, default=MAX_AREA_SIZE)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_option()
    PIC_SIZE = args.pic_size
    DATA_DIR = args.load_dir
    MIN_AREA_SIZE = args.min_area_size
    MAX_AREA_SIZE = args.max_area_size

    # モデル読み込み
    model_path = MODEL_DIR + args.modelname
    if not os.path.exists(model_path + ".json"):
        print("Model file not found.")
        sys.exit()

    model = model_from_json(open(model_path + ".json").read())
    model.load_weights(model_path + "_weights.hdf5")

    # カメラ
    cap = cv2.VideoCapture(0)
    camset.camera_get(cv2, cap)

    for name in os.listdir(DATA_DIR):
        if name not in LabelName and not name.startswith('.'):
            l = len(LabelName)
            LabelName.append(name)
            setattr(LabelNumber, name, l)

    dc.initialize()
    dc.move_home()

    while True:
        ret, frame = cap.read()
        ret, edframe = cap.read()
        cv2.imshow('Raw Frame', frame)

        gray = cv2.cvtColor(edframe, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_num_all = []
        centers_all = []  # 各輪郭の中心座標リスト（ユークリッド距離用）
        transform_pos_all = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA_SIZE or area > MAX_AREA_SIZE:
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

            image = cv2.resize(img_src, (PIC_SIZE, PIC_SIZE))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = img_to_array(image).astype("float32") / 255.
            image = image[None, ...]

            result = model.predict_classes(image)
            result_num = int(result[0])
            mp_x, mp_y = center
            transform_pos = transform_coordinate(mp_x, mp_y)

            result_num_all.append(result_num)
            transform_pos_all.append(transform_pos)
            centers_all.append(center)  # 中心を記録

            x, y, w_, h_ = cv2.boundingRect(contour)
            label = f"{result_num} {LabelName[result_num]}"
            cv2.rectangle(edframe, (x, y), (x+w_, y+h_), draw_white, 1)
            cv2.putText(edframe, label, (x, y-4), font, FONT_SIZE, draw_black, FONT_WIDTH)
            cv2.putText(edframe, f"DOBOT: {transform_pos}", (x+w_+5, y+10), font, FONT_SIZE, draw_green, FONT_WIDTH)

        cv2.imshow('Edited Frame', edframe)
        k = cv2.waitKey(1)

        if k == 27:
            break
        elif k == ord('c'):
            g = input("gain: ")
            e = input("exposure: ")
            camset.camera_set(cv2, cap, gain=float(g), exposure=float(e))
            camset.camera_get(cv2, cap)
        elif k == ord('h'):
            dc.move_home()
        elif k == ord('s') or is_blue_led_detected(frame):
            # 中心からの距離順に並べ替え
            distances = [np.linalg.norm(np.array(c) - np.array([frame.shape[1]//2, frame.shape[0]//2])) for c in centers_all]
            sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
            result_num_all = [result_num_all[i] for i in sorted_indices]
            transform_pos_all = [transform_pos_all[i] for i in sorted_indices]
            for result_num, pos in zip(result_num_all, transform_pos_all):
                print(f"{result_num} {LabelName[result_num]} -> {pos}")
                while not dc.dobot_classifier(result_num, pos[0], pos[1]):
                time.sleep(2)  # 動作後に2秒待機
                    pass

    dc.finalize()
    cap.release()
    cv2.destroyAllWindows()
