#coding:utf-8

'''
Author:kataoka
顔抽出した画像を RESULT_FILE の下に "0.jpg" から連番で保存していきます
'''

import cv2
import sys
import os


FILE_NAME = "lfw"				#画像ディレクトリ
RESULT_FILE = "detected"		#結果保存先ディレクトリ
CASCADE_PATH = 	"/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml"


def main():
	i = 0			#ナンバリング用
	
	#ディレクトリの作成
	if os.path.exists(RESULT_FILE) == False:
		os.mkdir(RESULT_FILE)

	files = os.listdir("./" + FILE_NAME)     							#ディレクトリ先のリスト取得

	for file in files:
		images = os.listdir("./" + FILE_NAME + "/" + file)
		for im in images:
			image_path = "./" + FILE_NAME + "/" + file + "/" + str(im)	#画像path取得

			image = cv2.imread(image_path)						#画像読み込み
			image_gray = cv2.cvtColor(image,cv2.cv.CV_BGR2GRAY)	#グレースケール変換
			cascade = cv2.CascadeClassifier(CASCADE_PATH)		
			facerect = cascade.detectMultiScale(image_gray,scaleFactor=1.1,minNeighbors=2,minSize=(10,10))

			#抽出
			if len(facerect) > 0:
				x = facerect[0][0]
				y = facerect[0][1]
				width = facerect[0][2]
				height = facerect[0][3]
				dst = image[y:y+height,x:x+width]
				cv2.imwrite('./' + RESULT_FILE + '/' + str(i) + '.jpg',dst)
				i += 1

if __name__ == '__main__':
	main()
