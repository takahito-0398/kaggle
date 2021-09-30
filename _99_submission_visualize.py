import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.figure_factory as ff
from glob import glob

def show_all():
    # p = "./submission/submission_ResNeXt50_3channel_type2_5epochs.csv"
    p_parent = "./submission"
    ps = glob(f"{p_parent}/*.csv")
    labels = [p.lstrip(f"{p_parent}\\submission_").rstrip(".csv") for p in ps]

    # a = pd.read_csv(p)
    hist_data = [list(pd.read_csv(p)["target"]) for p in ps]
    # print(hist_data)
    # group_labels = ["ResNeXt50_3channel_type2_5epochs.csv"]
    group_labels = labels
    fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_hist=False, show_rug=False)
    fig.show()


if __name__ == '__main__':
    show_all()