# 画像の部分領域に検出器を適用
# 画像のすべての部分領域に検出器を適用して，その結果の船らしさを表示

import os
import numpy as np
from skimage import data, color, exposure
from skimage.feature import hog
from skimage.transform import resize
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn import svm
import joblib
import math
import imageio
import cv2 

# pathは適宜変更してください
path = 'C:/Users/Akama/Desktop/Engineering/Python/PJ/sat_ship_detector/'
test_img_name = 'uraga.jpeg'

# SVMの学習結果を読み込む
print('Start loading SVM...')
detector = joblib.load(path + 'best_ship_detector.pkl')
print('Finished loading SVM.')

# 検出器の大きさの指定
PERSON_WIDTH = 80
PERSON_HEIGHT = 80

# 検出対象画像の指定
test_img_path = path + 'test/' + test_img_name
test_img = imageio.imread(test_img_path,as_gray=True)
test_img_to_show = imageio.imread(test_img_path,as_gray=False)

# 画像のサイズの変更の大きさと探索の細かさの指定
img_w = test_img.shape[1]
img_h = test_img.shape[0]
img_size = (img_h,img_w)

step_w = 50
step_h = 50

# 画像のサイズ変更
test_img = resize(test_img, img_size)
test_img_to_show = resize(test_img_to_show, img_size)

# 画像をスキャンして，各領域の船らしさを計算
likelihood_list = []
for i in range(5):
	for x in range(0,img_w-step_w-PERSON_WIDTH,step_w):
	    for y in range(0,img_h-step_h-PERSON_HEIGHT,step_h):
	        window = test_img[y:y+PERSON_HEIGHT,x:x+PERSON_WIDTH]
	        fd = hog(window, orientations=9, pixels_per_cell=(6,6),cells_per_block=(3,3), visualize=False) ## 領域内のHOG特徴量を取り出し、SVMに入力して学習データと比較
	        estimated_class = 1/(1+(math.exp(-1*detector.decision_function(fd.reshape(1,-1))))) ##SVMの出力値をシグモイド関数で0~1に正規化
	        if estimated_class >= 0.7: ## 領域内の船らしさが7割を超えた場合のみ座標を保持
	            likelihood_list.append([x,y,x+PERSON_WIDTH,y+PERSON_HEIGHT])

if len(likelihood_list) > 0:
	for rect in likelihood_list:
		cv2.rectangle(test_img_to_show, tuple(rect[0:2]), tuple(rect[2:4]), (255,0,0), 4)

# 歩行者らしさの表示
plt.subplot(111).set_axis_off()
plt.imshow(test_img_to_show)
#plt.imshow(test_img, cmap=plt.cm.gray)
plt.title('Ship Detection Result at ' + test_img_name)
plt.savefig(path + 'results/' + test_img_name + '.png')
plt.show()