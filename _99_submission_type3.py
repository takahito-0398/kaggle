import pandas as pd
import keras
import efficientnet.keras
import os
from glob import glob
import numpy as np
from functools import partial
import joblib




def get_batch_test(X, batch_size):

    """
    batchを取得する関数
    """
    SIZE = len(X)

    # n_batchs
    n_batchs = SIZE//batch_size
    i = 0
    while (i < n_batchs):
        print("doing", i, "/", n_batchs)

        idx_start = i * batch_size
        idx_stop = idx_start + batch_size


        #あるbatchのfilenameの配列を持っておく
        X_batch_name = X[idx_start:idx_stop]

        # INPUT_SHAPE = np.load(X_batch_name[0]).shape
        # files_X = [np.load(file) for file in X_batch_name]
        files_X = [np.repeat(np.load(file),3,axis=-1) for file in X_batch_name]

        INPUT_SHAPE = (69, 193, 3)
        X_batch = np.array(files_X).reshape(-1, *INPUT_SHAPE)
        # print(Y_batch)
        i += 1

        # 一応idxも返したい
        dict_idx = {"idx_start":idx_start, "idx_stop":idx_stop}

        # filenameにしたがってバッチのtensorを構築
        # これで(batch_size, 28, 28, 1)のtrainのテンソルが作られる
        # print(X_batch.shape, Y_batch.shape)
        yield X_batch, dict_idx

def make_prediction(model, df_pred):
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
    joblib.delayed(_pred)(X_batch, dict_idx) for  X_batch, dict_idx in get_batch_test(X_test, BATCH_SIZE)
    )

    return df_submit



if __name__ == '__main__':
    # P_MODEL = "./models/EfficientNetB0_3channel_50epochs.h5"
    # P_MODEL = "./models/EfficientNetB0_3channel_5epochs.h5"
    # P_MODEL = "./models/ResNeXt101_3channel_type2_5epochs.h5"
    # P_MODEL = "./models/EfficientNetB7_3channel_type2_5epochs.h5"
    P_MODEL = "./models/ResNeXt101_type3_5epochs.h5"

    # "EfficientNetB0_3channel_5epochs"の部分だけ取り出す
    FILENAME = P_MODEL.split("/")[-1].split(".")[-2]

    model = keras.models.load_model(P_MODEL, compile=False)
    p_parent_test = "./test_type3"

    p_test = "../input/g2net-gravitational-wave-detection/sample_submission.csv"
    df_pred = pd.read_csv(p_test)
    df_pred["path"] = p_parent_test + "/" + df_pred["id"] + ".npy"


    n_type = "3"
    df_submit = make_prediction(model, df_pred)

    p_save = f"./submission/submission_{FILENAME}.csv"
    df_submit.to_csv(p_save, index=False)