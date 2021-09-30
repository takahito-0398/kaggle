import os
import numpy as np
import datetime
from time import time
from datetime import datetime

import pandas as pd
import random
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import torch
from nnAudio.Spectrogram import CQT1992v2
# from nnAudio.Spectrogram import CQT

from scipy.stats import boxcox

from functools import partial
import joblib
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
import tensorflow as tf
from keras.utils import np_utils


# todo:
# いらないimportを消す

# util関数

def set_seed(seed=200):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def add_dt(p_file):
    if os.path.exists(p_file):

        dt_now = datetime.now()
        dt_str = dt_now.strftime("%Y%m%d%H%M%S")

        p_dir = os.path.dirname(p_file)
        basename = os.path.splitext(os.path.basename(p_file))[0]
        extension = os.path.splitext(p_file)[-1]

        p_file_new = f"{p_dir}/{basename}_{dt_str}{extension}"

    else:
        p_file_new = p_file

    return p_file_new

def min_max(x, axis=None):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min)

def standardization(x, axis=None, ddof=0):
    x_mean = x.mean(axis=axis, keepdims=True)
    x_std = x.std(axis=axis, keepdims=True, ddof=ddof)
    return (x - x_mean) / x_std

# sig.shape: (4096, )
# sr = 2048
# F: complex
def myfft(sig, sr):
    # サンプリング周波数から測定間隔に変換
    dt = 1 / sr
    # 何点測定したのか
    N = len(sig)

    # フーリエ変換
    F = np.fft.fft(sig)
    # 周波数軸（横軸）を作成
    # fq = np.linspace(0, dt, N)
    fq = np.fft.fftfreq(N, dt)

    return fq, F


def bandpassfilter(fq, F, fcut_low, fcut_high):
    mask = (fcut_low < fq) & (fq < fcut_high)
    F_masked = mask * F

    return fq, F_masked


def filtering(data, sr, fcut_low=30, fcut_high=900):
    fq, F = myfft(data, sr)
    fq, F_masked = bandpassfilter(fq, F, fcut_low, fcut_high)
    data_filtered = np.fft.ifft(F_masked).real

    return data_filtered

def myroc(*args, **kwargs):
    try:
        score = roc_auc_score(*args, **kwargs)
    except ValueError:
        score = 0

    return score

def roc_auc(y_true, y_pred):
    roc_auc = tf.py_function(myroc, (y_true, y_pred), tf.double)
    return roc_auc

def get_train_file_path(image_id):
    return "../input/g2net-gravitational-wave-detection/train/{}/{}/{}/{}.npy".format(
        image_id[0], image_id[1], image_id[2], image_id)

def get_test_file_path(image_id):
    return "../input/g2net-gravitational-wave-detection/test/{}/{}/{}/{}.npy".format(
        image_id[0], image_id[1], image_id[2], image_id)


def get_list_of_train_path():
    train = pd.read_csv('../input/g2net-gravitational-wave-detection/training_labels.csv')
    train['file_path'] = train['id'].apply(get_train_file_path)
    list_path_train = train['file_path'].values

    del train
    return list_path_train

def get_list_of_test_path():
    test = pd.read_csv('../input/g2net-gravitational-wave-detection/sample_submission.csv')
    test['file_path'] = test['id'].apply(get_test_file_path)
    list_path_test = test['file_path'].values

    del test
    return list_path_test

def get_df_of_train_path(n_type):
    df_train = pd.read_csv('../input/g2net-gravitational-wave-detection/training_labels.csv')
    p_parent = f"./train_type{n_type}"
    df_train['path'] = p_parent + "/" + df_train["id"] + ".npy"

    return df_train

def get_df_of_test_path(n_type):
    df_test = pd.read_csv('../input/g2net-gravitational-wave-detection/sample_submission.csv')
    p_parent = f"./test_type{n_type}"
    df_test['path'] = p_parent + "/" + df_test["id"] + ".npy"

    return df_test


# 1チャネル用
# arr.shape: (4096, )
def myboxcox_1channel(arr):
    arr_bc = arr.copy()

    eps = 1e-7
    target_shape = arr.shape
    target = arr.flatten()
    minimum = target.min()

    arr_bc = boxcox(target + minimum + eps)[0].reshape(target_shape)

    return arr_bc

# 3チャネル用
# arr.shape: (4096, 3)
def myboxcox(arr):
    n_channel = arr.shape[-1]
    arr_bc = arr.copy()

    eps = 1e-7
    for i in range(n_channel):
        target_shape = arr[:,:,i].shape
        target = arr[:,:,i].flatten()
        minimum = target.min()

        arr_bc[:,:,i] = boxcox(target + minimum + eps)[0].reshape(target_shape)

    return arr_bc


# shapeチェック用のデコレータ
# funcをassertではさむ
def check_shape(func):
    def mywrapper(**kwargs):
        waves = kwargs["waves"]
        input_shape = kwargs["input_shape"]
        output_shape = kwargs["output_shape"]

        assert waves.shape == input_shape, f"shapes are incompatible. Expected input_shape: {input_shape}, but {waves.shape}."

        converted_waves = func(**kwargs)

        assert converted_waves.shape == output_shape, f"shapes are incompatible. Expected output_shape: {output_shape}, but {converted_waves.shape}."

        return converted_waves

    return mywrapper


# cfg_transform = {
#     "sr":2048,
#     "fmin":20,
#     "hop_length":32,
#     "freq_bins":64
# }

# wave(1サイトのシグナル)をmelspecに変換
# wave: (1, 4096) -> melspec: (freq, time) (128, 9)
def convert2mel(wave, cfg_transform):
    sr = cfg_transform["sr"]
    fmin = cfg_transform["fmin"]
    hop_length = cfg_transform["hop_length"]
    freq_bins = cfg_transform["freq_bins"]

    melspec = librosa.feature.melspectrogram(wave/wave.max(),
                                                sr=sr, fmin=fmin,hop_length=hop_length,n_mels=freq_bins)

    melspec = librosa.power_to_db(melspec, ref=np.max)
    # melspec = melspec.transpose((1, 0))

    del wave
    return melspec


def convert2cqt(wave, func_transform):

    wave = wave / np.max(wave)
    wave = torch.from_numpy(wave).float()

    cqt = func_transform(wave)
    cqt = np.array(cqt)
    cqt = np.squeeze(cqt)

    del wave
    return cqt


# Xは.npyのパス
# yはone-hot encodingされたラベル
def get_batch(X, y, batch_size, INPUT_SHAPE, offset=0, func=lambda x:x):
    """
    batchを取得する関数
    """
    SIZE = len(X)

    n_batchs = SIZE//batch_size
    i = 0 + offset
    # i = 59495
    t = time()
    while (i < n_batchs):
        t_prev = t
        t = time()
        dt = round((t - t_prev),1)
        _n = n_batchs - i + 1
        eta_s = t + _n * dt
        eta_dt = datetime.fromtimestamp(eta_s)
        eta_dt = eta_dt.strftime("%Y/%m/%d %H:%M:%S")


        print(f"doing {i} / {n_batchs}\t\tused {dt}[s]\t\tETA:{eta_dt}")

        idx_start = i * batch_size
        idx_stop = idx_start + batch_size

        Y_batch = y[idx_start:idx_stop]

        #あるbatchのfilenameの配列を持っておく
        X_batch_name = X[idx_start:idx_stop]
        if len(Y_batch) != len(X_batch_name):
            print(i)

        files_X = [func(np.load(file)) for file in X_batch_name]
        X_batch = np.array(files_X).reshape(-1, *INPUT_SHAPE)
        i += 1

        yield X_batch, Y_batch



# 汎用性が著しく低くなってしまった。
def get_batch_DA(X, y, batch_size, INPUT_SHAPE, offset=0, func=lambda x:x, da_ratio=1.5, train_dir="./train_type5"):
    """
    batchを取得する関数
    """
    df_cm = pd.read_csv("./pred_by_traindata/pred_train_ResNet50V2_type5_3epochs.csv ")
    df_cm["path"] = train_dir + "/" + df_cm["id"] + ".npy"

    assert da_ratio >= 1, "da_ratio must be >= 1"

    # 最小で1
    n_da_per_batch = int(batch_size * (da_ratio - 1))

    SIZE = len(X)

    n_batchs = SIZE//batch_size
    i = 0 + offset
    # i = 59495
    t = time()
    while (i < n_batchs):
        t_prev = t
        t = time()
        dt = round((t - t_prev),1)
        _n = n_batchs - i + 1
        eta_s = t + _n * dt
        eta_dt = datetime.fromtimestamp(eta_s)
        eta_dt = eta_dt.strftime("%Y/%m/%d %H:%M:%S")


        print(f"doing {i} / {n_batchs}\t\tused {dt}[s]\t\tETA:{eta_dt}")

        idx_start = i * batch_size
        idx_stop = idx_start + batch_size

        y_batch = y[idx_start:idx_stop]

        #あるbatchのfilenameの配列を持っておく
        X_batch_name = X[idx_start:idx_stop]


        files_X = [np.load(file) for file in X_batch_name]
        X_batch = np.array(files_X).reshape(-1, *INPUT_SHAPE)


        ##########
        # data aug分を追加する

        df_aug = df_cm[df_cm["cm"] == "FN"].sample(n_da_per_batch)

        files_X_aug = [func(np.load(file)) for file in df_aug.path.values]
        X_batch_aug = np.array(files_X_aug).reshape(-1, *INPUT_SHAPE)
        X_batch_shape = X_batch_aug.shape
        med = pd.Series(X_batch_aug.flatten()).median()
        noise_scale = med * 0.1
        X_noise = (np.random.rand(*X_batch_shape) * 2 - 1) * noise_scale
        X_batch_aug += X_noise

        y_batch_aug = df_aug.true_label.values
        y_batch_aug = np_utils.to_categorical(y_batch_aug, 2)


        X_batch = np.vstack([X_batch, X_batch_aug])
        y_batch = np.vstack([y_batch, y_batch_aug])
        ##########

        i += 1

        yield X_batch, y_batch