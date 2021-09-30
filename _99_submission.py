import pandas as pd
import keras
import efficientnet.keras
import os
from glob import glob
import numpy as np
from functools import partial
import joblib
from time import time
from datetime import datetime

import re
from _00_utils import add_dt, myboxcox

P_TRAIN = "../input/g2net-gravitational-wave-detection/training_labels.csv"
P_TEST = "../input/g2net-gravitational-wave-detection/sample_submission.csv"


# def get_batch_test(X, batch_size, func=lambda x:x):
def get_batch_test(X, batch_size, func=myboxcox):

    """
    batchを取得する関数
    """
    SIZE = len(X)

    # n_batchs
    n_batchs = SIZE//batch_size
    i = 0
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


        #あるbatchのfilenameの配列を持っておく
        X_batch_name = X[idx_start:idx_stop]

        INPUT_SHAPE = np.load(X_batch_name[0]).shape
        files_X = [func(np.load(file)) for file in X_batch_name]
        X_batch = np.array(files_X).reshape(-1, *INPUT_SHAPE)
        # print(Y_batch)
        i += 1

        # 一応idxも返したい
        dict_idx = {"idx_start":idx_start, "idx_stop":idx_stop}

        # filenameにしたがってバッチのtensorを構築
        # これで(batch_size, 28, 28, 1)のtrainのテンソルが作られる
        # print(X_batch.shape, Y_batch.shape)
        yield X_batch, dict_idx

def make_prediction(model, df_pred, n_type):
    # batch_size=1000でHDDからバッチを取得する
    BATCH_SIZE = 256
    df_submit = df_pred[["id", "target"]].copy()
    X_test = list(df_pred["path"])
    for X_batch, dict_idx in get_batch_test(X_test, BATCH_SIZE):
        pred = model.predict(X_batch)
        # pred = pred.max(axis = 1)
        pred = pred[:,1]
        # print(pred)
        df_submit.iloc[dict_idx["idx_start"]:dict_idx["idx_stop"], 1] = pred
        del pred

        ##########################
    return df_submit

def _make_prediction(model, df_pred):
    # batch_size=1000でHDDからバッチを取得する
    BATCH_SIZE = 256
    df_submit = df_pred[["id", "target"]].copy()
    X_test = list(df_pred["path"])

    def _pred(X_batch, dict_idx):
        pred = model.predict(X_batch)
        pred = pred[:,1]
        df_submit.iloc[dict_idx["idx_start"]:dict_idx["idx_stop"], 1] = pred

    # 並列化したが速度にあまり変化なし
    _ = joblib.Parallel(n_jobs=32)(
    joblib.delayed(_pred)(X_batch, dict_idx) for X_batch, dict_idx in get_batch_test(X_test, BATCH_SIZE)
    )

    return df_submit


def make_submission(P_MODEL):
    # P_MODEL = "./models/ResNeXt101_type5_2epochs.h5"

    # "type"が含まれているか確認
    assert "type" in P_MODEL, "couldn't find a type in P_MODEL..."

    # type?の部分を抽出
    m = re.search(r'type\d+', P_MODEL)
    n_type = m.group()

    # "EfficientNetB0_3channel_5epochs"の部分だけ取り出す
    # FILENAME = P_MODEL.split("/")[-1].split(".")[-2]
    # a = './models\\ResNet50V2_type5_2th-3epochs_1th-4epochs_cv0.7985.h5'
    FILENAME = P_MODEL.split("\\")[-1]
    FILENAME = FILENAME.rsplit(".h5")[0]

    # ファイルパスを追加
    p_parent_test = f"./test_{n_type}"
    df_pred = pd.read_csv(P_TEST)
    df_pred["path"] = p_parent_test + "/" + df_pred["id"] + ".npy"

    model = keras.models.load_model(P_MODEL, compile=False)
    df_submit = make_prediction(model, df_pred, n_type)

    p_save = f"./submission/submission_{FILENAME}.csv"
    p_save = add_dt(p_save)

    df_submit.to_csv(p_save, index=False)

if __name__ == '__main__':
    # P_MODEL = "./models/EfficientNetB0_3channel_50epochs.h5"
    # P_MODEL = "./models/EfficientNetB0_3channel_5epochs.h5"
    # P_MODEL = "./models/ResNeXt101_3channel_type2_5epochs.h5"
    # P_MODEL = "./models/EfficientNetB7_3channel_type2_5epochs.h5"
    # P_MODEL = "./models/ResNeXt101_type3_5epochs.h5"
    # P_MODEL = "./models/ResNeXt101_type4_5epochs.h5"
    # P_MODEL = "./models/ResNet50V2_type5_10epochs.h5"
    # P_MODEL = "./models/ResNet50V2_type5_3epochs.h5"
    # P_MODEL = "./models/ResNet50V2_type5_4epochs.h5"
    # P_MODEL = "./models/ResNeXt101_type5_3epochs.h5"
    # P_MODEL = "./models/ResNeXt101_type5_2epochs.h5"
    # P_MODEL = "./models/ResNeXt101_type5_2epochs.h5"

    p_parent_model = "./models"
    p_models = glob(f"{p_parent_model}/*.h5")
    # regex = "ResNet50V2_type5_(2|3)th"
    # regex = "ResNet50V2_type5_\dth"
    # regex = "ResNeXt101_type5_\dth-\dcv"
    # regex = "ResNet50V2_noiseda_type5_\dth"
    # regex = "ResNet50V2_boxcox_noda_type5_\dth"
    regex = "ResNet50V2_bp30_900_noda_type6_\dth"

    p_models = [p_model for p_model in p_models if re.search(regex, p_model)]
    for P_MODEL in p_models:
        # print(P_MODEL)
        make_submission(P_MODEL)