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
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2
import numpy as np

import cameraSetting as camset
import dobotClassifier as dc

from common import *
from TransformationMatrix import MATRIX
MATRIX = np.array(MATRIX)

# ラベルごとの仕分け先座標（ロボット座標系）
SORTING_POSITIONS = {
    0: (235, -120),  # 小型
    1: (235, 0),     # 中型
    2: (235, 120),   # 大型
}

# 画像のサイズ調整（切り取った矩形を正方形化）
def rect_preprocess(img):
    h, w, c = img.shape
    longest_edge = max(h, w)
    top = bottom = left = right = 0
    if h < longest_edge:
        diff_h = longest_edge - h
        top = diff_h // 2
        bottom = diff_h - top
    elif w < longest_edge:
        diff_w = longest_edge - w
        left = diff_w // 2
        right = diff_w - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

# カメラ座標系 -> ロボット座標系への変換
def transform_coordinate(pos_x, pos_y):
    global MATRIX
    pos = np.array([[[pos_x, pos_y]]], dtype='float32')
    transform_pos = cv2.perspectiveTransform(pos, MATRIX)
    return int(transform_pos[0][0][0]), int(transform_pos[0][0][1])

# 青色LEDの検出関数（検出されたら仕分け動作をトリガー）
def is_blue_led_detected(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 100])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_area = cv2.countNonZero(mask)
    return blue_area > 100

# 引数の取得
def get_option():
    argparser = ArgumentParser()
    argparser.add_argument("-d", "--directory", dest="load_dir", type=str, default=DATA_DIR)
    argparser.add_argument("-m", "--modelname", dest="modelname", type=str, default="model")
    argparser.add_argument("-s", "--size", dest="pic_size", type=int, default=PIC_SIZE)
    argparser.add_argument("--min", dest="min_area_size", type=int, default=MIN_AREA_SIZE)
    argparser.add_argument("--max", dest="max_area_size", type=int, default=MAX_AREA_SIZE)
    return argparser.parse_args()

if __name__ == '__main__':
    args = get_option()
    PIC_SIZE      = args.pic_size
    DATA_DIR      = args.load_dir
    MIN_AREA_SIZE = args.min_area_size
    MAX_AREA_SIZE = args.max_area_size

    # モデルの読み込み
    if not os.path.isfile(MODEL_DIR + args.modelname + ".json"):
        print("\n No such model file.")
        sys.exit()

    print("\n - - - model loading - - -")
    model = model_from_json(open(MODEL_DIR + args.modelname + ".json").read())
    model.load_weights(MODEL_DIR + args.modelname + "_weights.hdf5")
    model.summary()

    cap = cv2.VideoCapture(0)
    camset.camera_get(cv2, cap)

    for dir in os.listdir(DATA_DIR):
        if dir not in LabelName and not dir.startswith('.'):
            l = len(LabelName)
            LabelName.append(dir)
            setattr(LabelNumber, dir, l)
            print(f"{l} : {dir}")

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

            image = cv2.resize(img_src, dsize=(PIC_SIZE, PIC_SIZE))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = img_to_array(image).astype("float32") / 255.
            image = image[None, ...]

            result = model.predict_classes(image)
            result_num = int(result[0])
            result_num_all.append(result_num)

            mp_x, mp_y = center
            transform_pos = transform_coordinate(mp_x, mp_y)
            transform_pos_all.append(transform_pos)

            # ラベルと座標の表示
            x, y, width, height = cv2.boundingRect(contour)
            label = f"{result_num} {LabelName[result_num]}"
            cv2.rectangle(edframe, (x, y), (x+width, y+height), draw_white, 1)
            cv2.putText(edframe, label, (x, y-4), font, FONT_SIZE, draw_black, FONT_WIDTH, cv2.LINE_AA)
            cv2.putText(edframe, f"DOBOT: {transform_pos}", (x+width+5, y+10), font, FONT_SIZE, draw_green, FONT_WIDTH, cv2.LINE_AA)

        cv2.imshow('Edited Frame', edframe)
        k = cv2.waitKey(1)

        if k == 27:
            break
        elif k == ord('c'):
            g = input("gain     : ")
            e = input("exposure : ")
            camset.camera_set(cv2, cap, gain=float(g), exposure=float(e))
            camset.camera_get(cv2, cap)
        elif k == ord('h'):
            dc.move_home()
        elif k == ord('s') or is_blue_led_detected(frame):
            for result_num, transform_pos in zip(result_num_all, transform_pos_all):
                print(f"{result_num} {LabelName[result_num]} -> {transform_pos}")
                while dc.dobot_classifier(result_num, transform_pos[0], transform_pos[1]) != True:
                    pass

    dc.finalize()
    cap.release()
    cv2.destroyAllWindows()
