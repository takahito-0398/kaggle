# import keras
import os
import random
import matplotlib.pyplot as plt
import pickle as pkl
import random

import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from keras.utils import np_utils
import lightgbm as lgb


def roc_auc(y_true, y_pred):
    roc_auc = tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)
    return roc_auc
# classes = ['1','2','3','4','5']
# nb_classes = len(classes)

# samplewise_std_normalization: 真理値．各入力をその標準偏差で正規化
# datagen = ImageDataGenerator(samplewise_std_normalization=True)
# train_datagen = ImageDataGenerator(samplewise_std_normalization=True)
# train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         # target_size=(150, 150),
#         batch_size=32,
#         class_mode='binary')
def prep_lgbm(img):
    # 計算量縮小のためグレースケール変換
    # 顔認識なら色はそんなに関係ないだろうと想像
    # =>処理時間が1/3になった

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 一次元にする
    flatten_img = img.flatten()

    scaler = StandardScaler()
    # StandardScalerは二次元じゃないと使えないらしい
    flatten_img_scaled = scaler.fit_transform(flatten_img.reshape(-1,1))
    # もともとの一次元の形状に戻す
    flatten_img_scaled = flatten_img_scaled.reshape(len(flatten_img_scaled))
    # print(flatten_img)
    # print(flatten_img_scaled)

    return flatten_img_scaled

def make_input(df_csv):
    # df_csv["path"]:画像のパスが入ってる
    # パスの画像に前処理関数(prep_lgbm)を適用
    flatten_imgs = df_csv["path"].map(cv2.imread).map(prep_lgbm)
    # img = cv2.imread(p)

    # shape:(sample数, 3channel*100px*100px)
    shape = (len(df_csv),flatten_imgs[0].shape[0])

    # モデルに入力するために形を整える
    X = np.empty(shape)
    for idx in range(len((flatten_imgs))):

        flatten_img = flatten_imgs[idx]
        X[idx] = flatten_img

    y = df_csv[["label"]].copy()


    # train:(3468, 30000)
    # test:(54, 30000)
    # print(X.shape)
    return X, y

# np_input  :ndarray(sample_size, 9, 128, 3)
# np_output :ndarray(sample_size, 9*128*3)
def prep(np_input):
    sample_size = len(np_input)
    np_output = np_input.reshape(sample_size, -1)
    # print(np_output.shape)

    return np_output

# input  :path
# output :ndarray(h,w,c)
def path2npy(path):
    output = np.load(path)

    return output


# Xは.npyのパス
# yはone-hot encodingされたラベル
def get_batch(X, y, batch_size):
    """
    batchを取得する関数
    """
    SIZE = len(X)

    # n_batchs
    n_batchs = SIZE//batch_size
    # for でyield
    i = 0
    while (i < n_batchs):
        print("doing", i, "/", n_batchs)

        idx_start = i * batch_size
        idx_stop = idx_start + batch_size

        Y_batch = y[idx_start:idx_stop]

        #あるbatchのfilenameの配列を持っておく
        X_batch_name = X[idx_start:idx_stop]
        if len(Y_batch) != len(X_batch_name):
            print(i)

        files_X = [np.load(file) for file in X_batch_name]
        X_batch = np.array(files_X).reshape(len(X_batch_name),-1)
        # print(Y_batch)
        i += 1

        # filenameにしたがってバッチのtensorを構築
        # これで(batch_size, 28, 28, 1)のtrainのテンソルが作られる
        # print(X_batch.shape, Y_batch.shape)
        yield X_batch, Y_batch

if __name__ == '__main__':
    train_dir = "./train_3channel"
    test_dir = "./test_3channel"
    model_dir = "./models"
    INPUT_SHAPE = (9, 128, 3)


    train = pd.read_csv('../input/g2net-gravitational-wave-detection/training_labels.csv')
    # test = pd.read_csv('../input/g2net-gravitational-wave-detection/sample_submission.csv')

    train["path"] = train_dir + "/" + train["id"] + ".npy"
    # print(train.tail())
    # raise
    y = train["target"]
    # y = np_utils.to_categorical(y, 2)
    X_train, X_test, y_train, y_test = train_test_split(train["path"], y, test_size=0.15)

    indices_train = list(X_train.index)

    N_train = 128
    indices_train_sample = random.sample(indices_train, N_train)
    a = X_train[indices_train_sample]

    # パス一覧からloadして、arrayを取得
    # それをN_train個保持しておく
    b = a.map(path2npy).values

    # 各arrayを取り出して新しい軸方向に重ねる
    # x.shape :(N_train, 9, 128, 3)
    c = np.array([list(array) for array in b])

    # flattenを含めた前処理を行う
    X_train_sample = prep(c)
    y_train_sample = prep(np.array(y_train[indices_train_sample]))

    lgb_train_dataset = lgb.Dataset(X_train_sample, y_train_sample)

    num_iter = 200
    params = {
            "task":"train",
            "objective":"multiclass",
            "metrics":"multi_logloss",
            "num_class":2,
            'random_state':1234,
            'verbose':0,
            'learning_rate': 0.1,         # 学習率
            'num_leaves': 21,             # ノードの数
            'min_data_in_leaf': 3,        # 決定木ノードの最小データ数
            'num_iteration': num_iter
            }         # 予測器(決定木)の数:イテレーション
    num_round = 200

    model = lgb.train(params, lgb_train_dataset, num_boost_round=num_round)


    indices_test = list(X_test.index)
    N_test = 512
    indices_test_sample = random.sample(indices_test, N_test)

    a = X_test[indices_test_sample]

    # パス一覧からloadして、arrayを取得
    # それをN_train個保持しておく
    b = a.map(path2npy).values

    # 各arrayを取り出して新しい軸方向に重ねる
    # x.shape :(N_train, 9, 128, 3)
    c = np.array([list(array) for array in b])

    X_test_sample = prep(c)
    y_test_sample = prep(np.array(y_test[indices_test_sample]))


    y_pred = model.predict(X_test_sample)[:,1]
    acc = None
    # acc = accuracy_score(y_test_sample, y_pred)
    roc = roc_auc_score(y_test_sample, y_pred)

    print(acc, roc)

