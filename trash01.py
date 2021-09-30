import numpy as np
import pandas as pd
import pandas as pd
import os
from functools import partial
import joblib
from tqdm import tqdm
# import matplotlib.pyplot as plt
# import librosa
# from nnAudio.Spectrogram import CQT1992v2
# import seaborn as sns

# from _00_utils import convert2mel, convert2cqt, standardization, min_max
# from _00_utils import get_batch


def add_data_path(p_parent):
    is_train = "train" in p_parent

    if is_train:
        df = pd.read_csv('../input/g2net-gravitational-wave-detection/training_labels.csv')

    else:
        df = pd.read_csv('../input/g2net-gravitational-wave-detection/sample_submission.csv')


    df["path"] = p_parent + "/" + df["id"] + ".npy"
    return df["path"].values


def get_save_path(p, is_train):
    if is_train:
        old_label = "_train"
        new_label = "_train"
    else:
        old_label = "_test"
        new_label = "_test"

    p_save = p.replace(old_label, new_label)
    return p_save




def tiling(p):
    is_train = "train" in p
    INPUT_SHAPE = (69, 193, 3)
    tmp = np.load(p)
    tmp = np.tile(tmp,3)

    assert tmp.shape == INPUT_SHAPE, f"Shape mismatch, expected {INPUT_SHAPE} but {tmp.shape}"

    p_save = get_save_path(p, is_train)
    p_parent_save = os.path.dirname(p_save)
    os.makedirs(p_parent_save, exist_ok=True)
    np.save(p_save, tmp)



def do_parallel(p_parent):

    list_path = add_data_path(p_parent)
    _ = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(tiling)(file_path) for file_path in tqdm(list_path)
    )


if __name__ == '__main__':
    p_parent = "./_train_type3"
    do_parallel(p_parent)
    p_parent = "./_test_type3"
    do_parallel(p_parent)
    # p_parent = "./train_type3/000b077d34.npy"
    # s = np.load(p_parent).shape
    # print(s)