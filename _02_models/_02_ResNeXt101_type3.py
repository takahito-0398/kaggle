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


train_dir = "./train_type3"
test_dir = "./test_type3"
model_dir = "./models"
INPUT_SHAPE = (69, 193, 3)



def set_seed(seed=200):
    tf.random.set_seed(seed)

    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)



def catch_ValueError(function):
    def inner(*args):
        "function の前後で呼び出しログをプリントする"
        print("{0}{1}".format(function.__name__, args))
        ret = function(*args)
        print("{0}{1} ==> {2}".format(function.__name__, args, ret))
        return ret
    return inner

def myroc(*args, **kwargs):
    try:
        score = roc_auc_score(*args, **kwargs)
    except ValueError:
        score = 0

    return score



def roc_auc(y_true, y_pred):
    roc_auc = tf.py_function(myroc, (y_true, y_pred), tf.double)
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
    # i = 7434
    i = 0
    # i = 59490
    while (i < n_batchs):
        print("doing", i, "/", n_batchs)

        idx_start = i * batch_size
        idx_stop = idx_start + batch_size

        Y_batch = y[idx_start:idx_stop]

        #あるbatchのfilenameの配列を持っておく
        X_batch_name = X[idx_start:idx_stop]
        if len(Y_batch) != len(X_batch_name):
            print(i)

        # files_X = [np.load(file) for file in X_batch_name]
        files_X = [np.repeat(np.load(file),3,axis=-1) for file in X_batch_name]
        # print(X_batch_name[0])
        # print(np.array(files_X).shape)
        X_batch = np.array(files_X).reshape(-1, *INPUT_SHAPE)
        # print(Y_batch)
        i += 1

        # filenameにしたがってバッチのtensorを構築
        # これで(batch_size, 28, 28, 1)のtrainのテンソルが作られる
        # print(X_batch.shape, Y_batch.shape)
        yield X_batch, Y_batch

def setModel(input_shape):
    from keras_applications.resnext import ResNeXt101

    input_tensor = Input(shape=input_shape)
    main_model = ResNeXt101(weights='imagenet',
                    backend=keras.backend,
                    layers=keras.layers,
                    models=keras.models,
                    utils=keras.utils,
                    include_top=False,
                    input_tensor=input_tensor)
    # ResNet50 = ResNet50(include_top=False, weights='imagenet',input_tensor=input_tensor)
    # main_model = ResNeXt101(include_top=False, weights='imagenet',input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=main_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    # 2値分類なので2
    top_model.add(Dense(2, activation='softmax'))

    top_model = Model(input=main_model.input, output=top_model(main_model.output))
    top_model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),metrics=['accuracy', roc_auc])

    return top_model

if __name__ == '__main__':
    set_seed()
    print(device_lib.list_local_devices())
    print("="*100)

    N_EPOCHS = 5
    train = pd.read_csv('../input/g2net-gravitational-wave-detection/training_labels.csv')
    # test = pd.read_csv('../input/g2net-gravitational-wave-detection/sample_submission.csv')

    X = list(train_dir + "/" + train["id"] + ".npy")
    y = train["target"]
    y = np_utils.to_categorical(y, 2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    # modelへ入力のため、次元を整形する
    # input_shape:(27, 128) => (27, 128, 1)
    # input_shape_org = np.load(X[0]).shape
    # input_shape = input_shape_org
    # input_shape = np.load(X[0]).reshape(*input_shape_org,1).shape
    print(INPUT_SHAPE)
    # raise

    model = setModel(INPUT_SHAPE)
    accuracy_train = []
    loss_train = []
    loss_test = []
    accuracy_test = []
    roc_train = []
    roc_test = []
    TNs = []
    FPs = []
    FNs = []
    TPs = []

    for epoch in range(N_EPOCHS):
        print("=" * 50)
        print(epoch, "/", N_EPOCHS)
        acc = []
        loss = []
        roc = []

        # batch_size=1000でHDDからバッチを取得する
        for X_batch, Y_batch in get_batch(X_train, y_train, 8):
            model.train_on_batch(X_batch, Y_batch)
            score = model.evaluate(X_batch, Y_batch)
            print("batch loss:", score[0])
            print("batch accuracy:", score[1])
            print("batch roc:", score[2])
            loss.append(score[0])
            acc.append(score[1])
            roc.append(score[2])

        N_test = 2048
        indices = random.sample(range(len(X_test)), N_test)

        p_samples_test = [X_test[idx] for idx in indices]
        X_batch_test = np.array([np.repeat(np.load(file),3,axis=-1) for file in p_samples_test]).reshape(-1, *INPUT_SHAPE)
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

    # json_string = model.to_json()
    MODEL_NAME = "ResNeXt101"
    p_model = f"./models/{MODEL_NAME}_type3_{N_EPOCHS}epochs.h5"
    model.save(p_model)

    x = range(N_EPOCHS)
    # fig = plt.figure()
    # fig, axes = plt.subplots(nrows=row, ncols=col)
    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(221)
    ax1.plot(x, loss_train, label='Train')
    ax1.plot(x, loss_test, label='Test')
    ax1.set_title('Categorical Crossentropy')
    ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Loss')
    ax1.legend()

    ax2 = fig.add_subplot(222)
    ax2.plot(x, accuracy_train, label='Train')
    ax2.plot(x, accuracy_test, label='Test')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    # ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0,1)
    ax2.legend()

    ax3 = fig.add_subplot(223)
    ax3.plot(x, roc_train, label='Train')
    ax3.plot(x, roc_test, label='Test')
    ax3.set_title('ROC AUC')
    ax3.set_xlabel('Epoch')
    # ax3.set_ylabel('Accuracy')
    ax3.set_ylim(0,1)
    ax3.legend()

    ax4 = fig.add_subplot(224)
    ax4.plot(x, TNs, label='TN', linestyle = "solid")
    ax4.plot(x, FPs, label='FP', linestyle = "dotted")
    ax4.plot(x, FNs, label='FN', linestyle = "dotted")
    ax4.plot(x, TPs, label='TP', linestyle = "solid")
    ax4.set_title('Confusion Matrix')
    ax4.set_xlabel('Epoch')
    # ax4.set_ylabel('Accuracy')
    # ax4.set_ylim(0,1)
    ax4.legend()

    p_file = f"./result_img/{MODEL_NAME}_3channel_{N_EPOCHS}epochs.png"
    fig.savefig(p_file)
    # plt.show()

