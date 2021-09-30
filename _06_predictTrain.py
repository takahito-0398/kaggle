import pandas as pd
import keras
import efficientnet.keras
import os
from glob import glob
import numpy as np
from functools import partial
import joblib
from time import time

from _00_utils import get_df_of_train_path


def get_batch_test(X, batch_size):

    """
    batchを取得する関数
    """
    SIZE = len(X)

    # n_batchs
    n_batchs = SIZE//batch_size
    i = 0
    t = time()
    while (i < n_batchs):
        t_old = t
        t = time()
        t_used_per_batch = round((t - t_old), 2)
        # print("doing", i, "/", n_batchs)
        print(f"doing {i}/{n_batchs}\t  used {t_used_per_batch}[s]")

        idx_start = i * batch_size
        idx_stop = idx_start + batch_size


        #あるbatchのfilenameの配列を持っておく
        X_batch_name = X[idx_start:idx_stop]

        INPUT_SHAPE = np.load(X_batch_name[0]).shape
        files_X = [np.load(file) for file in X_batch_name]
        X_batch = np.array(files_X).reshape(-1, *INPUT_SHAPE)
        # print(Y_batch)
        i += 1

        # 一応idxも返したい
        dict_idx = {"idx_start":idx_start, "idx_stop":idx_stop}

        del files_X, X_batch_name

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

# df_pred : DataFrame(columns=["id", "target", "path"])
def make_prediction_parallel(model, df_pred):
    # batch_size=1000でHDDからバッチを取得する
    BATCH_SIZE = 256
    df_submit = df_pred[["id", "target"]].copy()
    X_test = list(df_pred["path"])

    def _pred(X_batch, dict_idx):
        pred = model.predict(X_batch)
        pred = pred[:,1]
        df_submit.iloc[dict_idx["idx_start"]:dict_idx["idx_stop"], 1] = pred

    # 並列化したが速度にあまり変化なし
    _ = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(_pred)(X_batch, dict_idx) for  X_batch, dict_idx in get_batch_test(X_test, BATCH_SIZE)
    )

    return df_submit


def make_save_prediction(cfg_predict_train):
    P_MODEL = cfg_predict_train["P_MODEL"]
    n_type = cfg_predict_train["n_type"]
    model = keras.models.load_model(P_MODEL, compile=False)
    # p_parent_train = f"./train_type{n_type}"

    df_train = get_df_of_train_path(n_type)
    # print(df_train.head())

    df_pred_train = make_prediction(model=model, df_pred=df_train, n_type=n_type)
    # df_pred_train = make_prediction_parallel(model=model, df_pred=df_train)

    df_pred_train["true_label"] = df_train["target"]

    # "EfficientNetB0_3channel_5epochs"の部分だけ取り出す
    FILENAME = P_MODEL.split("/")[-1].split(".")[-2]
    p_save = f"./pred_by_traindata/pred_train_{FILENAME}.csv"
    os.makedirs(p_save, exist_ok=True)

    df_pred_train.to_csv(p_save, index=False)

if __name__ == '__main__':
    cfg_predict_train = {
        "P_MODEL" : "./models/ResNet50V2_type5_4epochs.h5",
        "n_type" : "5"
    }
    make_save_prediction(cfg_predict_train)

    cfg_predict_train = {
        "P_MODEL" : "./models/ResNeXt101_type5_2epochs.h5",
        "n_type" : "5"
    }
    make_save_prediction(cfg_predict_train)

    cfg_predict_train = {
        "P_MODEL" : "./models/ResNeXt101_type5_3epochs.h5",
        "n_type" : "5"
    }
    make_save_prediction(cfg_predict_train)



