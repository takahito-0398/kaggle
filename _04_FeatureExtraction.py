import keras
import efficientnet.keras
import numpy as np
# import tensorflow as tf
from keras.layers import Dense, Activation
from keras.models import Model
from matplotlib import pyplot as plt
import random
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import time
from functools import partial

def path2feature(p, model_fe):
    x = np.load(p).astype("float32")
    img_shape = x.shape
    x = x.reshape(-1, *img_shape)
    pred = model_fe.predict(x)
    pred = list(pred.squeeze())
    del x

    return pred

def printAggregation(features):
    n_all = features.size
    n_non_zero = np.count_nonzero(features)
    n_zero = n_all - n_non_zero
    print(f"features shape : {features.shape}")
    print(f"# of 0     : {n_zero}/{n_all} ratio:{n_zero/n_all}")
    print(f"# of not 0 : {n_non_zero}/{n_all} ratio:{n_non_zero/n_all}")

    return features

def forTrain():
    P_MODEL = "./models/EfficientNetB0_3channel_5epochs.h5"
    model =  keras.models.load_model(P_MODEL, compile=False)
    model_name = P_MODEL.split("/")[-1].split(".")[-2]

    layer_name = "dense_1"
    model_fe = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    print(model_fe.summary())


    label_csv_test = "../input/g2net-gravitational-wave-detection/training_labels.csv"
    df_label_csv = pd.read_csv(label_csv_test)
    p_parent = "./train_3channel"
    df_label_csv["path"] = p_parent + "/" + df_label_csv["id"] + ".npy"

    n_sample = 560000
    indices = random.sample(range(len(df_label_csv)), n_sample)
    ps = list(df_label_csv.iloc[indices]["path"])
    ids = df_label_csv.iloc[indices]["id"].to_numpy(dtype=object)
    targets = df_label_csv.iloc[indices]["target"].to_numpy(dtype="i8")


    t_start = time.time()

    path2feature_partial = partial(path2feature, model_fe=model_fe)
    features = [path2feature_partial(p) for p in tqdm(ps)]


    features = np.array(features, dtype="float32")
    t_stop = time.time()
    print("time to convert: {:.3f} [s]".format(t_stop-t_start))

    # print(features)

    printAggregation(features)

    dim = features.shape[1]
    p_save = f"./extracted_features/features_dim{dim}_sampleSize{n_sample}.npz"

    np.savez(p_save, ids=ids, features=features, targets=targets)


def forTest():
    P_MODEL = "./models/EfficientNetB0_3channel_5epochs.h5"
    model =  keras.models.load_model(P_MODEL, compile=False)
    model_name = P_MODEL.split("/")[-1].split(".")[-2]

    layer_name = "dense_1"
    model_fe = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    print(model_fe.summary())


    label_csv_test = "../input/g2net-gravitational-wave-detection/sample_submission.csv"
    df_label_csv = pd.read_csv(label_csv_test)
    p_parent = "./test_3channel"
    df_label_csv["path"] = p_parent + "/" + df_label_csv["id"] + ".npy"

    n_sample = 226000
    indices = random.sample(range(len(df_label_csv)), n_sample)
    ps = list(df_label_csv.iloc[indices]["path"])
    ids = df_label_csv.iloc[indices]["id"].to_numpy(dtype=object)
    # targets = df_label_csv.iloc[indices]["target"].to_numpy(dtype="i8")


    t_start = time.time()

    path2feature_partial = partial(path2feature, model_fe=model_fe)
    features = [path2feature_partial(p) for p in tqdm(ps)]


    features = np.array(features, dtype="float32")
    t_stop = time.time()
    print("time to convert: {:.3f} [s]".format(t_stop-t_start))

    # print(features)

    printAggregation(features)

    dim = features.shape[1]
    p_save = f"./extracted_features/test_features_dim{dim}_sampleSize{n_sample}.npz"

    np.savez(p_save, ids=ids, features=features)



if __name__ == '__main__':
    # forTest()
    getScore()

    # plt.spy(features, markersize=3)
    # plt.show()