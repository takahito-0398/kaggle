import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from glob import glob
import os
import random
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
# from scipy import sparse





def roc_auc(y_true, y_pred):
    roc_auc = tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)
    return roc_auc


def train():
    p = "./extracted_features/features_dim256_sampleSize560000.npz"
    matrix = np.load(p, allow_pickle=True)
    ids = matrix["ids"]
    X_train = matrix["features"]
    y_train = matrix["targets"]
    print(ids.shape, X_train.shape, y_train.shape)


    N_sample = 560000
    indices = [i for i in range(N_sample)]
    X_train_sample = X_train[indices]
    y_train_sample = y_train[indices]
    lgb_train_dataset = lgb.Dataset(X_train_sample, y_train_sample)

    num_iter = 200
    params = {
            "task":"train",
            "objective":"regression",
            "metrics": {"l2"},
            'random_state':1234,
            'verbose':1,
            'learning_rate': 0.1,         # 学習率
            'num_leaves': 31,             # ノードの数
            'min_data_in_leaf': 3,        # 決定木ノードの最小データ数
            'num_iteration': num_iter
            }         # 予測器(決定木)の数:イテレーション
    # params = {
    #         "task":"train",
    #         "objective":"multiclass",
    #         "metrics":"multi_logloss",
    #         "num_class":2,
    #         'random_state':1234,
    #         'verbose':0,
    #         'learning_rate': 0.1,         # 学習率
    #         'num_leaves': 21,             # ノードの数
    #         'min_data_in_leaf': 3,        # 決定木ノードの最小データ数
    #         'num_iteration': num_iter
    #         }         # 予測器(決定木)の数:イテレーション
    num_round = 200

    model = lgb.train(params, lgb_train_dataset, num_boost_round=num_round)

    name = f"LGBM_{num_round}round_{N_sample}samples"
    p_save = f"./models/{name}.txt"
    model.save_model(p_save)


def predict():
    p_test = "./extracted_features/test_features_dim256_sampleSize226000.npz"
    info_features = p_test.split("/")[-1].split(".")[-2]
    matrix = np.load(p_test, allow_pickle=True)
    ids = matrix["ids"]
    X_test = matrix["features"]

    # p_model = "C:/Users/takah/Desktop/main/programming/kaggle/g2net-gravitational-wave-detection/models/LGBM_200round_560000samples.txt"
    p_model = "./models/LGBM_200round_560000samples.txt"
    info_model = p_model.split("/")[-1].split(".")[-2]

    model = lgb.Booster(model_file=p_model)
    y_pred = model.predict(X_test)

    df_submit = pd.DataFrame(columns=["id", "target"])
    df_submit["id"] = ids
    df_submit["target"] = y_pred

    # a = df_submit["target"]
    # a.hist(bins=50, alpha=0.8, color="blue")
    # plt.show()

    p_save = f"./submission/{info_model}_{info_features}.csv"
    print(p_save)
    df_submit.to_csv(p_save, index=False)

    # y_pred = model.predict(X_test_sample)[:,1]
    # acc = None
    # # acc = accuracy_score(y_test_sample, y_pred)
    # roc = roc_auc_score(y_test_sample, y_pred)

    # print(acc, roc)

if __name__ == '__main__':
    # train()
    predict()