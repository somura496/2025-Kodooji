# DetectLED_Jun25.py
# 青色LEDを検出するモジュール
import cv2
import numpy as np

def is_blue_led_detected(frame):
    """
    フレーム内に青色LEDが検出されたかどうかを返す
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([100, 150, 100])
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.countNonZero(mask) > 100
