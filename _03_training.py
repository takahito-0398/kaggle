# todo
# kerasとtf.kerasを分離する（片方のみ使う）
# modelやsubmissionを上書きしないようなロジックにする


# import keras
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense,Input, Dropout
import keras
from keras_applications.resnext import ResNeXt101

from keras import optimizers
from tensorflow.python.client import device_lib
import tensorflow as tf

# from varname import nameof

from _00_utils import set_seed, add_dt, get_batch
from _02_models import myResNeXt101
# from _02_models import myResNet50V2


def training(model, cfg_training):
    MODEL_NAME      = cfg_training["MODEL_NAME"]
    n_type          = cfg_training["type"]
    INPUT_SHAPE     = cfg_training["INPUT_SHAPE"]
    n_epochs        = cfg_training["n_epochs"]
    batch_size      = cfg_training["batch_size"]
    test_size       = cfg_training["test_size"]
    N_test_sample   = cfg_training["N_test_sample"]

    train_dir = f"./train_type{n_type}"
    # test_dir = f"./test_type{n_type}"
    # model_dir = "./models"


    train = pd.read_csv('../input/g2net-gravitational-wave-detection/training_labels.csv')
    # test = pd.read_csv('../input/g2net-gravitational-wave-detection/sample_submission.csv')

    X = list(train_dir + "/" + train["id"] + ".npy")
    y = train["target"]
    y = np_utils.to_categorical(y, 2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    accuracy_train, accuracy_test = [], []
    loss_train, loss_test = [], []
    roc_train, roc_test = [], []
    TNs, FPs, FNs, TPs = [], [], [], []

    for epoch in range(n_epochs):
        print("=" * 50)
        print(epoch, "/", n_epochs)
        acc, loss, roc = [], [], []

        # batch_size=1000でHDDからバッチを取得する
        for X_batch, Y_batch in get_batch(X_train, y_train, batch_size, INPUT_SHAPE):
            model.train_on_batch(X_batch, Y_batch)
            score = model.evaluate(X_batch, Y_batch)
            print("batch loss:", score[0])
            print("batch accuracy:", score[1])
            print("batch roc:", score[2])
            loss.append(score[0])
            acc.append(score[1])
            roc.append(score[2])

        indices = random.sample(range(len(X_test)), N_test_sample)

        p_samples_test = [X_test[idx] for idx in indices]
        X_batch_test = np.array([np.load(file) for file in p_samples_test]).reshape(-1, *INPUT_SHAPE)
        score = model.evaluate(X_batch_test, y_test[indices])
        y_pred = model.predict(X_batch_test)
        y_pred_label = np.argmax(y_pred, 1)
        y_true_label = y_test[indices][:,1]
        cm = confusion_matrix(y_true_label, y_pred_label)
        tn, fp, fn, tp = cm.flatten()

        print("Train loss", np.mean(loss))
        print("Test loss:", score[0])
        print("Train accuracy", np.mean(acc))
        print("Test accuracy:", score[1])
        print("Train ROC", np.mean(roc))
        print("Test ROC:", score[2])
        print(cm)
        loss_train.append(np.mean(loss))
        loss_test.append(score[0])
        accuracy_train.append(np.mean(acc))
        accuracy_test.append(score[1])
        roc_train.append(np.mean(roc))
        roc_test.append(score[2])

        TNs.append(tn)
        FPs.append(fp)
        FNs.append(fn)
        TPs.append(tp)


    p_model = f"./models/{MODEL_NAME}_type{n_type}_{n_epochs}epochs.h5"
    p_model = add_dt(p_model)
    model.save(p_model)
    print(f"The trained model has been saved to {p_model}!")


    x = range(n_epochs)
    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(221)
    ax1.plot(x, loss_train, label='Train')
    ax1.plot(x, loss_test, label='Test')
    ax1.set_title('Categorical Crossentropy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2 = fig.add_subplot(222)
    ax2.plot(x, accuracy_train, label='Train')
    ax2.plot(x, accuracy_test, label='Test')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylim(0,1)
    ax2.legend()

    ax3 = fig.add_subplot(223)
    ax3.plot(x, roc_train, label='Train')
    ax3.plot(x, roc_test, label='Test')
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

if __name__ == '__main__':
    set_seed()
    print(device_lib.list_local_devices())
    print("="*100)


    cfg_training = {
        "MODEL_NAME"    : "ResNet50V2",
        "type"          : "5",
        "INPUT_SHAPE"   : (64, 129, 3),
        "n_epochs"      : 4,
        "batch_size"    : 64,
        "test_size"     : 0.15,
        "N_test_sample" : 4096
    }
    cfg_training = {
        "MODEL_NAME"    : "ResNeXt101",
        "type"          : "5",
        "INPUT_SHAPE"   : (64, 129, 3),
        "n_epochs"      : 2,
        "batch_size"    : 8,
        "test_size"     : 0.15,
        "N_test_sample" : 4096
    }

    model = myResNeXt101(cfg_training["INPUT_SHAPE"])
    # model = myResNet50V2(cfg_training["INPUT_SHAPE"])
    training(model, cfg_training)