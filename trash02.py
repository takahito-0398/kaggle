import numpy as np
from numpy.random import *
import pandas as pd
import seaborn as sns
import pandas as pd
import os
from functools import partial
import joblib
from tqdm import tqdm
import random

import matplotlib.pyplot as plt


from sklearn.cluster import KMeans


def makeSamples(N_samples):
    # pos_center = [(0,0), (0,1), (1,0), (1,1)]
    mu = [0, 0]
    sigma = [[30, 20], [20, 50]]

    # 2次元正規乱数を1万個生成
    values_a = multivariate_normal(mu, sigma, 10000)
    df_a = pd.DataFrame(columns=["x", "y", "type"])
    df_a["x"] = values_a[:,0]
    df_a["y"] = values_a[:,1]
    df_a["type"] = "A"

    values_b = multivariate_normal(mu, sigma, 10000) + [30,30]
    df_b = pd.DataFrame(columns=["x", "y", "type"])
    df_b["x"] = values_b[:,0]
    df_b["y"] = values_b[:,1]
    df_b["type"] = "B"

    values_c = multivariate_normal(mu, sigma, 10000) + [50,0]
    df_c = pd.DataFrame(columns=["x", "y", "type"])
    df_c["x"] = values_c[:,0]
    df_c["y"] = values_c[:,1]
    df_c["type"] = "C"


    df = pd.concat([df_a, df_b, df_c])
    # 散布図

    # plt.show()

    df2 = df[["x", "y"]]
    pred = KMeans(n_clusters=3).fit_predict(df2)
    df["pred_KMeans"] = pred
    print(pred)
    print(type(pred))
    print(len(pred))
    print(df)
    sns.jointplot(x=df.columns[0], y=df.columns[1], data=df, alpha=0.1, hue=df.columns[2], marginal_kws={'hist_kws': {'edgecolor': 'red'}})
    sns.jointplot(x=df.columns[0], y=df.columns[1], data=df, alpha=0.1, hue=df.columns[3], marginal_kws={'hist_kws': {'edgecolor': 'red'}})
    plt.show()

if __name__ == '__main__':
    makeSamples(10)