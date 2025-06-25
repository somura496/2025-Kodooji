# main_classifier_Jun25.py
# -*- coding: utf-8 -*-
"""
画像認識 → 座標変換 → ソート → LEDトリガー → dobot仕分け
"""

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import sys, time
import cv2
import numpy as np

from tensorflow.keras.models import model_from_json
from argparse import ArgumentParser

import cameraSetting as camset
import dobotClassifier as dc
from common import *

from ItemQueue_Jun25 import get_sorted_by_center
from DetectLED_Jun25 import is_blue_led_detected
from DetectItem_Jun25 import detect_and_classify

# オプション引数
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

    model_path = MODEL_DIR + args.modelname
    if not os.path.exists(model_path + ".json"):
        print("Model file not found.")
        sys.exit()

    model = model_from_json(open(model_path + ".json").read())
    model.load_weights(model_path + "_weights.hdf5")

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
        if not ret:
            print("カメラからフレームを取得できませんでした")
            break

        result_num_all, transform_pos_all, centers_all, edframe = detect_and_classify(
            frame, model, PIC_SIZE, MIN_AREA_SIZE, MAX_AREA_SIZE, LabelName)

        cv2.imshow('Raw Frame', frame)
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
            sorted_indices = get_sorted_by_center(centers_all, frame.shape)
            result_num_all = [result_num_all[i] for i in sorted_indices]
            transform_pos_all = [transform_pos_all[i] for i in sorted_indices]
            for result_num, pos in zip(result_num_all, transform_pos_all):
                print(f"{result_num} {LabelName[result_num]} -> {pos}")
                while not dc.dobot_classifier(result_num, pos[0], pos[1]):
                    time.sleep(2)

    dc.finalize()
    cap.release()
    cv2.destroyAllWindows()
