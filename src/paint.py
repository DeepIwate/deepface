
# coding: utf-8

# In[2]:

import cv2
import numpy as np
import tkinter
import tkinter.filedialog

# OpenCVのイベントリストの出力
def printEvents():
    events = [i for i in dir(cv2) if 'EVENT' in i]
    print (events)

# OpenCVのマウスイベントを扱うためのクラス
class CVMouseEvent:
    def __init__(self, press_func=None, drag_func=None, release_func=None):
        self._press_func = press_func
        self._drag_func = drag_func
        self._release_func = release_func

        self._is_drag = False

    # Callback登録関数
    def setCallBack(self, win_name):
        cv2.setMouseCallback(win_name, self._callBack)

    def _doEvent(self, event_func, x, y):
        if event_func is not None:
            event_func(x, y)

    def _callBack(self, event, x, y, flags, param):
        # マウス左ボタンが押された時の処理
        if event == cv2.EVENT_LBUTTONDOWN:
            self._doEvent(self._press_func, x, y)
            self._is_drag = True

        # マウス左ドラッグ時の処理
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._is_drag:
                self._doEvent(self._drag_func, x, y)

        # マウス左ボタンが離された時の処理
        elif event == cv2.EVENT_LBUTTONUP:
            self._doEvent(self._release_func, x, y)
            self._is_drag = False


# 描画用の空画像作成
def emptyImage():
    return np.zeros((512, 512, 3), np.uint8)

# シンプルなマウス描画のデモ
def simplePaint(filename):
    defaultFace = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = defaultFace

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),(0,0,0),(255,255,255)]
    color = colors[0]

    # ドラッグ時に描画する関数の定義
    def brushPaint(x, y):
        cv2.circle(img, (x, y), 20, color, -1)

    win_name = 'Face Editor'
    cv2.namedWindow(win_name)

    # CVMouseEventクラスによるドラッグ描画関数の登録
    mouse_event = CVMouseEvent(drag_func=brushPaint)
    mouse_event.setCallBack(win_name)

    while(True):
        cv2.imshow(win_name, img)

        key = cv2.waitKey(30) & 0xFF

        # 色切り替えの実装
        if key == ord('1'):
            color = colors[0]
        elif key == ord('2'):
            color = colors[1]
        elif key == ord('3'):
            color = colors[2]
        elif key == ord('4'):
            color = colors[3]
        elif key == ord('5'):
            color = colors[4]

        # 画像のリセット
        elif key == ord('r'):
            img = defaultFace

        elif key == ord('q'):
            cv2.imwrite("result.jpg",img)
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    #編集したいファイルを選ぶ。jpg限定
    tk = tkinter.Tk()
    tk.withdraw()
    args = { "initialdir" : "","filetypes" : [("image", "*.jpg")],"title" : "編集したい画像を選んでください"}
    filename= tkinter.filedialog.askopenfilename(**args)
    #編集する qで終了
    printEvents()
    simplePaint(filename)


# In[ ]:

#プロトタイプ　9/20
#編集したいファイルを選択できるように変更(jpg only) 9/27
#太さを変更できるようにする予定

