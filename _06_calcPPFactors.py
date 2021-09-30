import numpy as np
import pandas as pd
import seaborn as sns
from random import randint
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import roc_curve
from tqdm import tqdm
import warnings
from _00_utils import myroc

warnings.simplefilter('ignore', RuntimeWarning)

def BCE_col(df):

    df["BCE"] = 1.0
    for i in range(len(df)):
        y = df["target"][i]
        y = y.reshape(1, y.size)
        t = df["true_label"][i]
        t = t.reshape(1, t.size)
        bce = -(t * np.log(y + 1e-7))[0][0]
        df["BCE"][i] = bce
        print(f"{i}/{len(df)}\tBCE:{bce}")

    print(df)
    df.to_csv("./_dummy.csv")

def plot_violin():
    p = "./pred_by_traindata/pred_train_ResNet50V2_type5_3epochs.csv"
    df = pd.read_csv(p)
    print(df.head())

    sns.set()
    sns.violinplot(data=df, x='true_label', y='target')
    plt.show()


def scale(probs, factors):
    a, b, c = factors[0], factors[1], factors[2]

    probs = probs.copy()
    idx = np.where((probs!=1) & (probs!=0))[0]
    odds = probs[idx] / (1-probs[idx])
    odds_new = a * (odds ** b) + c
    probs[idx] =  odds_new / (1 + odds_new)

    return probs

def cross_entropy_error(y, t):
    if y.ndim == 1: # 次元が 1 の場合
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.size
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def show_roc_curve(y_pred, y_true):
    fpr_all, tpr_all, _ = roc_curve(y_true, y_pred, drop_intermediate=False)

    sns.set()
    # plt.plot(fpr_all, tpr_all, marker='o', size=0.2)
    plt.plot(fpr_all, tpr_all)
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    # plt.show()
    # plt.savefig('data/dst/sklearn_roc_curve_all.png')

def getPercentiles(probs):
    pass

# 四分位範囲ごとにことなるスケーリング
def scaling(df, params_dict, target_col="target"):
    # print(df.head())
    # n = len(params_dict)
    perc = [0.25, 0.5, 0.75]
    # n = len(perc)
    a = df[target_col].quantile(perc)

    # 第一四分位数、第二四分位数、第三四分位数
    a1, a2, a3 = a.values[0], a.values[1], a.values[2]

    params_1 = params_dict["params_1"]
    params_2 = params_dict["params_2"]
    params_3 = params_dict["params_3"]
    params_4 = params_dict["params_4"]


    # print(df[df[target_col] < a1])
    # print(df[df[target_col] == 0.0])
    # df[target_col].mask(df[target_col] < a1, scale(df[target_col].values, params_1), inplace=True)
    # print(df[df[target_col].isnull()])
    # raise


    df[target_col].mask(df[target_col] < a1, scale(df[target_col].values, params_1), inplace=True)
    df[target_col].mask((a1 <= df[target_col]) & (df[target_col] < a2), scale(df[target_col].values, params_2), inplace=True)
    df[target_col].mask((a2 <= df[target_col]) & (df[target_col] < a3), scale(df[target_col].values, params_3), inplace=True)
    df[target_col].mask(a3 <= df[target_col], scale(df[target_col].values, params_4), inplace=True)

    # print(df.head())
    return df
    # print(a1)




if __name__ == '__main__':
    a = [10**i for i in range(10)]
    b = [i/20 for i in range(-20,20)]
    c = [10**i for i in range(10)]
    print(f"{a}\n{b}\n{c}")
    iter_1 = itertools.product(a, b, c)
    iter_2 = itertools.product(a, b, c)
    iter_3 = itertools.product(a, b, c)
    iter_4 = itertools.product(a, b, c)

    # print(len(iter_X))
    p = "./pred_by_traindata/pred_train_ResNet50V2_type5_3epochs.csv"
    df = pd.read_csv(p)

    y_true = df.true_label.values
    y_pred = df.target.values
    auc_init = myroc(y_true, y_pred)

    auc_best = auc_init

    params_best_1, params_best_2, params_best_3, params_best_4 = (1, 1, 0),(1, 1, 0), (1, 1, 0), (1, 1, 0)

    for params_1 in iter_1:
        for params_2 in iter_2:
            for params_3 in iter_3:
                for params_4 in iter_4:

                    params_dict = {
                        "params_1" : params_1,
                        "params_2" : params_2,
                        "params_3" : params_3,
                        "params_4" : params_4
                    }

                    df = scaling(df, params_dict)
                    y_true = df.true_label.values
                    y_pred_pp = df.target.values

                    auc = myroc(y_true, y_pred_pp)
                    print(f"auc:{auc}\t ratio:{auc/auc_init}")

                    if auc > auc_best:
                        auc_best = auc
                        params_best_1 = params_1
                        params_best_2 = params_2
                        params_best_3 = params_3
                        params_best_4 = params_4
                        print("="*10 + "BEST!" + "="*50)
                        print(f"auc:{auc_best}\t ratio:{auc_best/auc_init}")
                        print("f{params_best_1}\n{params_best_2}\n{params_best_3}\n{params_best_4}")
                        print("="*65)

    print("="*10 + "BEST!" + "="*50)
    print(f"auc:{auc_best}\t ratio:{auc_best/auc_init}")
    print("f{params_best_1}\n{params_best_2}\n{params_best_3}\n{params_best_4}")