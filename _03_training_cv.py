# todo
# kerasとtf.kerasを分離する（片方のみ使う）
# modelやsubmissionを上書きしないようなロジックにする
# cvの処理入れる
# 時間計測する
# コールバック足す
# 不要なimport消す


# import keras
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
from time import time

from operator import itemgetter
import itertools

# from sklearn.model_selection import train_val_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


# from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
# from keras.models import Sequential, Model
# from keras.layers import Flatten, Dense,Input, Dropout
# import keras
# from keras_applications.resnext import ResNeXt101

from keras import optimizers
from tensorflow.keras.metrics import binary_crossentropy, Accuracy, categorical_crossentropy
import efficientnet.keras
from tensorflow.python.client import device_lib
import tensorflow as tf


from _00_utils import set_seed, add_dt, get_batch, get_batch_DA, myroc, myboxcox
# from _02_models import myResNeXt101
from _02_models import myResNet50V2
# from _02_models import myEfficientNetV2B0

# 1cvラウンドごとの学習を行う関数
# 学習した関数を返す
def train_per_cv(model, cfg_training, X_train, y_train):
    INPUT_SHAPE     = cfg_training["INPUT_SHAPE"]
    batch_size      = cfg_training["batch_size"]
    offset          = cfg_training["offset"]

    acc_train, loss_train, roc_train = [], [], []

    # batch_size=1000でHDDからバッチを取得する
    for X_batch, Y_batch in get_batch(X_train, y_train, batch_size, INPUT_SHAPE, offset=offset, func=myboxcox):
    # for X_batch, Y_batch in get_batch_DA(X_train, y_train, batch_size, INPUT_SHAPE, offset=offset, func=myboxcox):

        model.train_on_batch(X_batch, Y_batch)
        score = model.evaluate(X_batch, Y_batch)
        print("batch loss:", score[0])
        print("batch accuracy:", score[1])
        print("batch roc:", score[2])
        loss_train.append(score[0])
        acc_train.append(score[1])
        roc_train.append(score[2])

        trained_model = model
    return trained_model, acc_train, loss_train, roc_train


def val_per_cv(trained_model, cfg_training, X_val, y_val):
    INPUT_SHAPE     = cfg_training["INPUT_SHAPE"]
    N_val_sample    = cfg_training["N_val_sample"]

    if N_val_sample == -1:
        indices = [i for i in range(len(X_val))]

    else:
        indices = random.sample(range(len(X_val)), N_val_sample)


    y_pred = []
    y_true = []


    for X_batch_val, Y_batch_val in get_batch(X=X_val, y=y_val, batch_size=256, INPUT_SHAPE=INPUT_SHAPE, offset=0, func=myboxcox):

        y_pred_batch = model.predict(X_batch_val)
        y_pred_label_batch = np.argmax(y_pred_batch, 1)

        y_true_label_batch = Y_batch_val[:,1]


        y_pred.append(list(y_pred_label_batch))
        y_true.append(list(y_true_label_batch))

    y_pred = list(itertools.chain.from_iterable(y_pred))
    y_true = list(itertools.chain.from_iterable(y_true))
    auc_val = myroc(y_true, y_pred)
    cm_val = confusion_matrix(y_true, y_pred)


    ####

    return auc_val, cm_val

def save_result_graph(cfg_training, result):
    MODEL_NAME      = cfg_training["MODEL_NAME"]
    n_epochs        = cfg_training["n_epochs"]
    n_type          = cfg_training["type"]

    accuracy_train  = result["accuracy_train"]
    accuracy_val    = result["accuracy_val"]
    loss_train      = result["loss_train"]
    loss_val        = result["loss_val"]
    roc_train       = result["roc_train"]
    roc_val         = result["roc_val"]
    TNs             = result["TNs"]
    FPs             = result["FPs"]
    FNs             = result["FNs"]
    TPs             = result["TPs"]

    x = [i+1 for i in range(n_epochs)]
    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(221)
    ax1.plot(x, loss_train, label='Train')
    ax1.plot(x, loss_val, label='val')
    ax1.set_title('Categorical Crossentropy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2 = fig.add_subplot(222)
    ax2.plot(x, accuracy_train, label='Train')
    ax2.plot(x, accuracy_val, label='val')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylim(0,1)
    ax2.legend()

    ax3 = fig.add_subplot(223)
    ax3.plot(x, roc_train, label='Train')
    ax3.plot(x, roc_val, label='val')
    ax3.set_title('ROC AUC')
    ax3.set_xlabel('Epoch')
    ax3.set_ylim(0,1)
    ax3.legend()

    ax4 = fig.add_subplot(224)
    ax4.plot(x, TNs, label='TN', linestyle = "solid")
    ax4.plot(x, FPs, label='FP', linestyle = "dotted")
    ax4.plot(x, FNs, label='FN', linestyle = "dotted")
    ax4.plot(x, TPs, label='TP', linestyle = "solid")
    ax4.set_title('Confusion Matrix')
    ax4.set_xlabel('Epoch')
    ax4.legend()

    p_file = f"./result_img/{MODEL_NAME}_type{n_type}_{n_epochs}epochs.png"
    p_file = add_dt(p_file)
    fig.savefig(p_file)
    print(f"The learning process has been saved to {p_file}!")

def training(model, cfg_training):
    MODEL_NAME      = cfg_training["MODEL_NAME"]
    n_type          = cfg_training["type"]
    INPUT_SHAPE      = cfg_training["INPUT_SHAPE"]
    n_epochs        = cfg_training["n_epochs"]
    batch_size        = cfg_training["batch_size"]
    val_size        = cfg_training["val_size"]
    offset        = cfg_training["offset"]
    N_val_sample        = cfg_training["N_val_sample"]
    N_FOLD          = cfg_training["N_FOLD"]

    with open("./models/training.log","a") as f:
        f.write("\n")
        f.write("="*100)
        f.write("\nCONFIG:")
        s = f"\nMODEL_NAME\tn_type\tINPUT_SHAPE\tn_epochs\tbatch_size\tval_size\toffset\tN_val_sample\tN_FOLD"
        f.write(s)
        s = f"\n{MODEL_NAME}\t{n_type}\t{INPUT_SHAPE}\t{n_epochs}\t{batch_size}\t{val_size}\t{offset}\t{N_val_sample}\t{N_FOLD}"
        f.write(s)
        s = "\n\ndatetime\tcv_round\tN_FOLD\tepoch\tn_epochs\tauc_roc\tTN\tFP\tFN\tTP\taccuracy\tprecision\trecall\tf1"
        f.write(s)



    train_dir = f"./train_type{n_type}"

    train = pd.read_csv('../input/g2net-gravitational-wave-detection/training_labels.csv')

    # X = list(train_dir + "/" + train["id"] + ".npy")
    X = train_dir + "/" + train["id"] + ".npy"
    y = train["target"]
    # train["target_categorical"] = np_utils.to_categorical(train["target"], 2)
    # y = train["target_categorical"]


    accuracy_train, accuracy_val = [], []
    loss_train, loss_val = [], []
    roc_train, roc_val = [], []
    TNs, FPs, FNs, TPs = [], [], [], []


    kf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=42)
    ###########
    for fold, (train_indices, valid_indices) in enumerate(kf.split(X, y)):
        print("=" * 50)
        print(f"cv: {fold} / {N_FOLD}")

        for epoch in range(n_epochs):
            print(f"epoch: {epoch} / {n_epochs}")

            X_train, X_val = X.iloc[train_indices], X.iloc[valid_indices]
            y_train, y_val = y.iloc[train_indices], y.iloc[valid_indices]

            y_train, y_val = np_utils.to_categorical(y_train, 2), np_utils.to_categorical(y_val, 2)


            model, acc_train, loss_train, roc_train = train_per_cv(model, cfg_training, X_train, y_train)
            auc_val, cm_val = val_per_cv(model, cfg_training, X_val, y_val)
            tn, fp, fn, tp = cm_val.flatten()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            ###########

            print("Train ROC", np.mean(roc_train))
            print("val ROC:", auc_val)
            print(cm_val)
            loss_train.append(np.mean(loss_train))
            accuracy_train.append(np.mean(acc_train))
            roc_train.append(np.mean(roc_train))
            roc_val.append(auc_val)

            TNs.append(tn)
            FPs.append(fp)
            FNs.append(fn)
            TPs.append(tp)

            cv_roc = round(auc_val, 4)
            p_model = f"./models/{MODEL_NAME}_type{n_type}_{fold+1}th-{N_FOLD}cv_{epoch+1}th-{n_epochs}epochs_cv{cv_roc}.h5"

            p_model = add_dt(p_model)
            model.save(p_model)
            print(f"The trained model has been saved to {p_model}!")

            with open("./models/training.log","a") as f:
                now = datetime.datetime.now()
                now = now.strftime("%Y/%m/%d %H:%M:%S")
                s = f"\n{now}\t{fold+1}\t{N_FOLD}\t{epoch+1}\t{n_epochs}\t{cv_roc}\t{tn}\t{fp}\t{fn}\t{tp}\t{accuracy}\t{precision}\t{recall}\t{f1}"
                f.write(s)
                # s = f"\n\nThe trained model has been saved to {p_model}"



    # result = {
    #     "accuracy_train" : accuracy_train,
    #     "accuracy_val"   : accuracy_val,
    #     "loss_train"     : loss_train,
    #     "loss_val"       : loss_val,
    #     "roc_train"      : roc_train,
    #     "roc_val"        : roc_val,
    #     "TNs"            : TNs,
    #     "FPs"            : FPs,
    #     "FNs"            : FNs,
    #     "TPs"            : TPs
    # }
    # save_result_graph(cfg_training, result)


if __name__ == '__main__':
    set_seed()
    print(device_lib.list_local_devices())
    print("="*100)


    cfg_training = {
        "MODEL_NAME"    : "ResNet50V2_boxcox_noda",
        "type"          : "5",
        "INPUT_SHAPE"   : (64, 129, 3),
        "n_epochs"      : 3,
        "batch_size"    : 100,
        "val_size"      : 0.15,
        "offset"        : 0,
        "N_val_sample"  : -1,
        "N_FOLD"  : 4
    }

    # cfg_training = {
    #     "MODEL_NAME"    : "ResNeXt101",
    #     "type"          : "5",
    #     "INPUT_SHAPE"   : (64, 129, 3),
    #     "n_epochs"      : 3,
    #     "batch_size"    : 16,
    #     "val_size"      : 0.15,
    #     "offset"        : 0,
    #     "N_val_sample"  : -1,
    #     "N_FOLD"  : 4
    # }

    cfg_training = {
        "MODEL_NAME"    : "ResNet50V2_bp30_900_noda",
        "type"          : "6",
        "INPUT_SHAPE"   : (64, 129, 3),
        "n_epochs"      : 3,
        "batch_size"    : 100,
        "val_size"      : 0.15,
        "offset"        : 0, 
        "N_val_sample"  : -1,
        "N_FOLD"  : 4
    }

    # model = myResNeXt101(cfg_training["INPUT_SHAPE"])
    model = myResNet50V2(cfg_training["INPUT_SHAPE"])
    # model = myEfficientNetV2B0(cfg_training["INPUT_SHAPE"])
    training(model, cfg_training)