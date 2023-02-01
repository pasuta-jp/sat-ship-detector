import numpy as np
from sklearn import svm
import joblib
from sklearn import model_selection as cross_validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# pathは適宜変更してください
path = 'C:/Users/Akama/Desktop/Engineering/Python/PJ/sat_ship_detector/'

# 特徴量の読み込み
X = np.loadtxt(path + 'HOG_ship_data.csv', delimiter=",")
y = np.loadtxt(path + 'HOG_ship_target.csv', delimiter=",")

# 線形SVMの学習パラメータを格子点探索で求める
tuned_parameters = [{'C': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}]

print('Starting Grid Search...')
gscv = GridSearchCV(svm.LinearSVC(), tuned_parameters, cv=5)
gscv.fit(X, y)
svm_best = gscv.best_estimator_

print('Searched result of  C =', svm_best.C)

# 最適(?)なパラメータを用いたSVMの再学習
print('Re-learning SVM with best parameter set...')
svm_best.fit(X, y)

# 学習結果の保存
print('Finished learning SVM　with Grid-search.')
joblib.dump(svm_best, path + 'best_ship_detector.pkl', compress=9)