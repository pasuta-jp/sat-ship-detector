import os
from skimage import data, color, exposure
import imageio
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# pathは適宜変更してください
path = 'C:/Users/Akama/Desktop/Engineering/Python/PJ/sat_ship_detector/'

pos_img_dir = path + 'train_positive/'
pos_img_files = os.listdir(pos_img_dir)

neg_img_dir = path + 'train_negative/'
neg_img_files = os.listdir(neg_img_dir)

PERSON_WIDTH = 80
PERSON_HEIGHT = 80
leftop = [0,0]
rightbottom =  [0+PERSON_WIDTH,0+PERSON_HEIGHT]

X = []
y = []

## ポジティブ画像からのHOG特徴量の取り出し
print('Loading ' + str(len(pos_img_files)) + ' positive files...')
for pos_img_file in pos_img_files:
    pos_filepath = pos_img_dir + pos_img_file
    pos_img = imageio.imread(pos_filepath,as_gray=True)
    pos_roi = pos_img[leftop[1]:rightbottom[1],leftop[0]:rightbottom[0]]
    fd = hog(pos_roi, orientations=9, pixels_per_cell=(6,6),cells_per_block=(3,3), visualize=False)
    X.append(fd)
    y.append(1)

## ネガティブ画像からのHOG特徴量の取り出し
print('Loading ' + str(len(neg_img_files)) + ' negative files...')
for neg_img_file in neg_img_files:
    neg_filepath = neg_img_dir + neg_img_file
    neg_img = imageio.imread(neg_filepath,as_gray=True)
    neg_roi = neg_img[leftop[1]:rightbottom[1],leftop[0]:rightbottom[0]]
    fd = hog(neg_roi, orientations=9, pixels_per_cell=(6,6),cells_per_block=(3,3), visualize=False)
    X.append(fd)
    y.append(0)
 
## リストをnp.array型に変換
X = np.array(X)
y = np.array(y)

# 特徴量の書き出し
np.savetxt(path + 'HOG_ship_data.csv', X, fmt="%f", delimiter=",")
np.savetxt(path + 'HOG_ship_target.csv', y, fmt="%.0f", delimiter=",")
print('HOG export completed.')