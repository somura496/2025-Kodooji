'''
Project Name  : dobot_ai
File Name     : dobotClassifier.py
File Encoding : UTF-8
Copyright © 2020 Afrel Co.,Ltd.
'''

import os
import myDobotModule as dobot

# Z座標
# オブジェクトの大きさ、環境に合わせて変更する
z = -60#-58
z_offset = -30
def initialize():
    '''
    DOBOTの初期化処理を呼び出し、カメラに映らない場所に移動
    '''
    dobot.initialize()
    dobot.move(150, 100, z_offset, 0)

def finalize():
    '''
    DOBOTの終了処理を呼び出す
    '''
    dobot.finalize()

def move_home():
    '''
    右側にカメラがあることを考慮し、アームを移動させ、DOBOTをホームポジションに移動させる
    '''
    dobot.move(150, 100, z_offset, 0)
    dobot.move_home()
    dobot.move(150, 100, z_offset, 0)

def dobot_classifier(label, pos_x, pos_y):
    '''
    指定された座標のオブジェクトを取り、ラベルごとに仕分ける
    '''
    # オブジェクトの真上に移動
    dobot.move(pos_x, pos_y, z_offset, 0)
    dobot.wait(1)
    # オブジェクトを取れる位置まで移動し、オブジェクトを取る
    dobot.move(pos_x, pos_y, z, 0)
    dobot.suctioncup(True, True)
    dobot.wait(1)
    # オブジェクトの真上に移動
    dobot.move(pos_x, pos_y, z_offset, 0)
    dobot.wait(1)

    # ラベルに合わせた座標を指定
    if label <= 3:
        x = 150 + 50 * label
        y = 100
    elif label <= 6:
        x = 150 + 50 * (label - 4)
        y = 150
    elif label <= 8:
        x = 150 + 50 * (label - 7)
        y = 200
    else:
        x = 300
        y = 0

    # ラベルごとに仕分ける位置の真上に移動
    dobot.move(x, y, z_offset, 0)
    dobot.wait(2)

    # オブジェクトを置く
    dobot.move(x, y, -30, 0)
    dobot.suctioncup(False, False)
    dobot.wait(2)

    print("move -> (%d, %d)" % (x, y))

    # カメラに映らない場所に移動
    dobot.move(150, 100, z_offset, 0)

    return True

